from pathlib import Path
from datetime import datetime
import os 
import copy
from typing import Any

import torch
import torchio as tio
from torch.utils.data import random_split, DataLoader
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter
import torchvision

import pandas as pd
from nilearn import plotting
import nibabel as nib
import monai
import json


from tqdm.auto import tqdm

import matplotlib.pyplot as plt 

import sys
sys.path.append("/home/emma/Projets/stroke_lesion_segmentation_v2/")
sys.path.append("/home/emma/Projets/stroke_lesion_segmentation_v2/config_files/")
sys.path.append("/home/emma/Projets/stroke_lesion_segmentation_v2/utils/")
from transforms import augment, preprocess
from get_subjects import get_subjects
from load_model import load
from subject_dataloader import SubjectDataLoader

#################
#   Parameters  #
#################

# Solution from https://github.com/fepegar/torchio/issues/1179 to fix problem of compatibilities btwn torchio and pytorch 2.3 (See to_dict() function from MySubject in get_subjects file)
class SubjectsDataset(tio.SubjectsDataset):
    def __init__(self, *args, **kwargs: dict[str, Any]):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int):
        try:
            index = int(index)
        except (RuntimeError, TypeError):
            message = (
                f'Index "{index}" must be int or compatible dtype,'
                f' but an object of type "{type(index)}" was passed'
            )
            raise ValueError(message)

        subject = self._subjects[index]
        subject = copy.deepcopy(subject)  # cheap since images not loaded yet
        if self.load_getitem:
            subject.load()

        # Apply transform (this is usually the bottleneck)
        if self._transform is not None:
            subject = self._transform(subject)
        # Here I've changed the return to a dictionary rather than a Subject
        return subject.to_dict()
    
    def get_subjects(self):

        return self._subjects


config_files = ["config_unet_atlas"]
data_infos_files = ["dataset_subsample"]

for i in range(len(config_files)):
    config_file = config_files[i]
    data_infos = data_infos_files[i]
    print(f"######## training with config {config_file} and data {data_infos}")

    with open('./config_files/'+config_file+".json") as f:
            ctx = json.load(f)
            num_workers = ctx["num_workers"]
            num_epochs = ctx["num_epochs"]
            task = ctx["experiment_name"]
            lr = ctx["initial_lr"]
            seed = ctx["seed"]
            net_model = ctx["net_model"]
            batch_size = ctx["batch_size"]
            dropout = ctx["dropout"]
            loss_type = ctx['loss_type']
            channels = ctx["channels"]
            n_layers = len(channels)
            train_val_ratio = ctx["train_val_ratio"]
            if ctx["patch"]:
                patch_size = ctx["patch_size"]
                queue_length = ctx["queue_length"]
                samples_per_volume = ctx["samples_per_volume"] 

    with open('./config_files/'+data_infos+".json") as f:
        data_info = json.load(f)
        channel = data_info["channel_names"]["0"]
        rootdir_train_img = ctx["rootdir_train_img"]
        rootdir_train_labels = ctx["rootdir_train_labels"]
        rootdir_test_img = ctx["rootdir_test-cap"]
        rootdir_test_labels = ctx["rootdir_test_labels-cap"]
        suffixe_img_train = data_info["suffixe_img-train"]
        suffixe_labels_train = data_info["suffixe_labels-train"]
        suffixe_img_test = data_info["suffixe_img-test"]
        suffixe_labels_test = data_info["suffixe_labels-test"]
        num_classes = len(data_info["labels"])
        file_ending = data_info["file_ending"]
        subsample = data_info["subset"]
        labels_names = list(data_info["labels"].keys())
    print(f"{num_classes} classes : {labels_names}")

    sample = ""
    if subsample:
        sample = "_subset"

    current_dateTime = datetime.now()
    id_run = config_file + sample + "_" + str(current_dateTime.day) + "-" + str(current_dateTime.month) + "-" + str(current_dateTime.year) + "-" + str(current_dateTime.hour) + str(current_dateTime.minute) + str(current_dateTime.second) 
    save_model_weights_path = "./weights/" + id_run + ".pth"


    #################
    #   MONITORING  #
    #################

    logger = TensorBoardLogger("unet_logs", name=config_file)
    writer = SummaryWriter("unet_logs/"+config_file)

    ##############
    #   Devices  #
    ##############

    if torch.cuda.is_available():
        [print() for i in range(torch.cuda.device_count())]
        [print(f"Available GPUs : \n{i} : {torch.cuda.get_device_name(i)}") for i in range(torch.cuda.device_count())]
        device = "cuda" 
    else:
        device = "cpu"
    print(f"device used : {device}")


    ################
    #   DATA PATH  #
    ################
    print("\n# DATA PATH : \n")

    img_dir=Path(rootdir_train_img)
    print(f"Training images in : {img_dir}")
    labels_dir=Path(rootdir_train_labels)
    print(f"Labels in : {labels_dir}")
    img_test_dir=Path(rootdir_test_img)
    print(f"Test images in : {img_test_dir}")


    ####################
    #   TRAINING DATA  #
    ####################
    print(f"\n# TRAINING DATA : \n")

    train_image_paths = sorted(img_dir.glob('**/*'+suffixe_img_train+file_ending))
    train_label_paths = sorted(labels_dir.glob('**/*'+suffixe_labels_train+file_ending))
    test_img_paths = sorted(img_test_dir.glob('**/*_'+suffixe_img_test+file_ending))
    test_label_paths = sorted(img_test_dir.glob('**/*_'+suffixe_labels_test+file_ending))

    assert len(train_image_paths) == len(train_label_paths)
    assert len(test_img_paths) == len(test_label_paths)

    train_subjects = get_subjects(train_image_paths, train_label_paths, subsample)

    test_subjects = get_subjects(test_img_paths, test_label_paths)

    train_subjects_dataset = SubjectsDataset(train_subjects)
    print('training dataset size: ', len(train_subjects), ' subjects')

    #######################
    #   ONE SUBJECT PLOT  #
    #######################
    # print("\n# ONE SUBJECT PLOT\n")

    # sub_tmp = nib.Nifti1Image(test_healthy_subjects_dataset[1]["t1"]['data'].numpy().squeeze(), test_healthy_subjects_dataset[1]["t1"]['affine'])
    # plotting.plot_img(sub_tmp, display_mode='mosaic')
    # plt.show()


    ##########################
    #   DATA TRANFORMATION   #
    ##########################
    print("\n# DATA TRANFORMATION\n")

    transform_train = tio.Compose([preprocess(num_classes), augment()])
    transform_val = tio.Compose([preprocess(num_classes)])
    transform_test = tio.Compose([preprocess(num_classes)])

    #######################
    #   TRAIN VAL SPLIT   #
    #######################

    print("\n# TRAIN VAL SPLIT\n")
    num_subjects = len(train_subjects)

    num_train_subjects = int(round(num_subjects * train_val_ratio))
    num_val_subjects = num_subjects - num_train_subjects
    splits = num_train_subjects, num_val_subjects
    generator = torch.Generator().manual_seed(seed)
    train_subjects, val_subjects = random_split(train_subjects, splits, generator=generator)

    train_subjects_dataset = SubjectsDataset(train_subjects, transform=transform_train)
    val_subjects_dataset = SubjectsDataset(val_subjects, transform=transform_val)
    test_subjects_dataset = SubjectsDataset(test_subjects, transform=transform_test)

    print(f"Training: {len(train_subjects_dataset)}")
    print(f"Validation: {len(val_subjects_dataset)}")     
    print(f"Test: {len(test_subjects_dataset)}")    

    if ctx["patch"]:
        print("Patch")
        patch_sampler = tio.data.LabelSampler(
            patch_size=patch_size,
            label_name='seg',
            # label_probabilities=probabilities,
        )

        train_set = tio.Queue(
            train_subjects_dataset,
            queue_length,
            samples_per_volume,
            patch_sampler,
            num_workers=num_workers,
        )

        val_set = tio.Queue(
            val_subjects_dataset,
            queue_length,
            samples_per_volume,
            patch_sampler,
            num_workers=num_workers,
        )

        generator_train = patch_sampler(train_subjects[0])
        generator_test = patch_sampler(test_subjects[0])
        img_train=next(iter(generator_train))
        img_test=next(iter(generator_test))
        img_train.plot()
        plt.show()
        img_test.plot()
        plt.show()
        # img_grid = torchvision.utils.make_grid(img_train["t1"]['data'][:,:,:,49])
        writer.add_image('patch_image_train', img_train["t1"]['data'][:,:,:,49])
        # img_grid = torchvision.utils.make_grid(img_test["t1"]['data'][:,:,:,49])
        writer.add_image('patch_image_test', img_test["t1"]['data'][:,:,:,49])
    else:
        print("No patch")
        val_set = val_subjects_dataset
        train_set = train_subjects_dataset



    #################
    #   LOAD DATA   #
    #################
    print("\n# LOAD DATA\n")

    train_dataloader = DataLoader(dataset=train_set, batch_size= batch_size, num_workers=7, pin_memory=True)
    val_dataloader = DataLoader(dataset=val_set, batch_size=batch_size, num_workers=7, pin_memory=True)
    test_dataloader = DataLoader(dataset=test_subjects_dataset, batch_size=batch_size, num_workers=7)

    batch = next(iter(train_dataloader))
    print(type(batch))  # Verif see comment line 39
    assert batch.__class__ is dict

    img = next(iter(train_dataloader))
    print(img['t1']['data'].shape)
    img_grid = torchvision.utils.make_grid(img['t1']['data'][:,:,:,:,80])
    writer.add_image('images_train', img_grid)
    img = next(iter(test_dataloader))
    print(img['t1']['data'].shape)
    img_grid = torchvision.utils.make_grid(img['t1']['data'][:,:,:,:,112])
    writer.add_image('images_test', img_grid)
    #############
    #   MODEL   #
    #############
    print(f"\n# MODEL : {net_model}\n")

    # model = load("/home/emma/Projets/stroke_lesion_segmentation_v2/weights/config_unet_atlas_subset_23-8-2024-153932.pth",net_model, lr, dropout, loss_type, num_classes, channels, num_epochs)

    model = load(None, net_model, lr, dropout, loss_type, num_classes, channels, num_epochs)


    ## Trainer 

    # early_stopping = pl.callbacks.early_stopping.EarlyStopping(
    #     monitor="val_loss",
    #     patience = 5,
    # )

    # lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        max_epochs=num_epochs, # Number of pass of the entire training set to the network
        accelerator=device, 
        devices=1,
        precision=16,
        logger=logger,
        log_every_n_steps=1,
        # callbacks=[lr_monitor],
        # limit_train_batches=0.2 # For fast training
    )

    # #################
    # #   TRAINING    #
    # #################
    print("\n# TRAINING\n")


    start = datetime.now()
    print("Training started at", start)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders = val_dataloader)
    print("Training duration:", datetime.now() - start)


    #################
    #   SAVE MODEL  #
    #################
    # print("\n# SAVE MODEL\n")

    # torch.save(model.state_dict(), save_model_weights_path)

    # print("model saved in : " + save_model_weights_path)

    #################
    #   INFERENCE   #
    #################
    print("# INFERENCE")

    model.eval()
    def inference(img_set, save = True, metric = False, df = None, data = "", patch = False):
        get_dice = monai.metrics.DiceMetric(include_background=False, reduction="none")
        get_hd = monai.metrics.HausdorffDistanceMetric(include_background=False, reduction="none", percentile=95)
        subjects_list = []
    
        for subject in img_set._subjects:
            if img_set._transform is not None:
                subject = img_set._transform(subject)
            print(type(subject))
            if patch: 
                grid_sampler = tio.inference.GridSampler(
                    subject,
                    patch_size,
                    patch_overlap=10,
                )
                aggregator = tio.inference.GridAggregator(grid_sampler)
                loader = DataLoader(grid_sampler)
                subject.clear_history()
            else:
                loader = subject

            with torch.no_grad():
                if patch: 
                    for batch in tqdm(loader, unit='batch'):
                        input = batch['t1']['data']
                        
                        logits = model.net(input.type(torch.FloatTensor))
                        labels = logits.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True)
                        locations = batch[tio.LOCATION]
                        aggregator.add_batch(labels, locations)                     
                else:
                    input = subject['t1']['data'].unsqueeze(axis=0)
                    print(input.shape)
                    logits = model.net(input.type(torch.FloatTensor))
                    labels = logits.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True)
                        
            if patch:
                output_tensor = aggregator.get_output_tensor()
            else:
                output_tensor = labels.squeeze(axis=0)


            print(f"output tensor shape : {output_tensor.shape}")
            pred = tio.LabelMap(tensor=output_tensor, affine=subject['t1']['affine'])
            # plotting.plot_img(nib.Nifti1Image(input.cpu().numpy().squeeze(), subject['t1']['affine']), display_mode='tiled')
            if metric:
                if data == "train" or data == "val":
                    gt_tensor = subject['seg']['data'].unsqueeze(axis=0)
                else:
                    gt_tensor = torch.nn.functional.one_hot(subject['seg']['data'].long(), num_classes=num_classes).permute(0, 4, 1, 2, 3)
                print(f"gt tensor shape : {gt_tensor.shape}")
                outputs_one_hot = torch.nn.functional.one_hot(output_tensor.long(), num_classes=num_classes)
                print(f"outputs_one_hot shape : {outputs_one_hot.shape}")
                outputs_one_hot = outputs_one_hot.permute(0, 4, 1, 2, 3)
                print(f"outputs_one_hot shape permuted: {outputs_one_hot.shape}")
                # if data == "test":
                #     outputs_one_hot = outputs_one_hot[:,5,:,:,:].unsqueeze(axis=0)
                get_dice(outputs_one_hot.to(model.device), gt_tensor.to(model.device))
                get_hd(outputs_one_hot.to(model.device), gt_tensor.to(model.device))

                print(f"subjects : {subjects_list}")
                dice = get_dice.aggregate().cpu().numpy()[0]
                print(f"DICE : {dice[0]}")
                get_dice.reset()
                hd = get_hd.aggregate().cpu().numpy()[0]
                print(f"HD : {hd[0]}")
                get_hd.reset()
                df = pd.concat([df, pd.DataFrame(df.columns,[subject['subject']] + [dice[0]] + [hd[0]])], ignore_index=True)

            subject["prediction"] = pred
            new_subject = subject.apply_inverse_transform()

        
            if save :
                out_file = "./out-predictions/"+id_run+"/"+data+"/"+subject['subject']
                if not os.path.exists(out_file):
                    os.makedirs(out_file)
                pred = nib.Nifti1Image(new_subject['prediction']['data'].numpy().astype(float).squeeze(), new_subject['t1']['affine'])
                gt = nib.Nifti1Image(new_subject['seg']['data'].numpy().astype(float).squeeze(), new_subject['t1']['affine'])
                t1 = nib.Nifti1Image(new_subject['t1']['data'].numpy().astype(float).squeeze(), new_subject['t1']['affine'])
                nib.save(pred,f"{out_file}/{subject['subject']}_pred.nii.gz")
                nib.save(t1,f"{out_file}/{subject['subject']}_t1.nii.gz")
                nib.save(gt,f"{out_file}/{subject['subject']}_seg.nii.gz")
        if metric:
            return df

    df_labels = pd.DataFrame(columns=["Subjects"] + ["DICE"] + ["HD 95"])

    print("# Validation")
    print(val_subjects_dataset)
    print(type(val_subjects_dataset))
    df_val = inference(val_subjects_dataset, metric = True, df = df_labels, data="val", patch = ctx["patch"])
    df_val.to_csv("./out-predictions/"+id_run+"/scores_val.csv", index=False)

    print("# Training")
    df_train = inference(train_subjects_dataset, metric = True, df = df_labels, data="train", patch = ctx["patch"])
    df_train.to_csv("./out-predictions/"+id_run+"/scores_train.csv", index=False)

    print("# Test")
    df_labels_test = pd.DataFrame(columns=["Subjects", "DICE", "HD 95"])
    df_test = inference(test_subjects_dataset, metric = True, df = df_labels_test, data="test", patch = ctx["patch"])
    df_test.to_csv("./out-predictions/"+id_run+"/scores_test.csv", index=False)



#'''