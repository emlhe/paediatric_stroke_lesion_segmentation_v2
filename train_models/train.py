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
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import torchvision

import pandas as pd
from nilearn import plotting
import nibabel as nib
import monai
import json


from tqdm.auto import tqdm

import matplotlib.pyplot as plt 

from utils.transforms import augment, preprocess
from utils.get_subjects import get_subjects
from utils.load_model import load

#################
#   Parameters  #
#################

# Solution from https://github.com/fepegar/torchio/issues/1179 to fix problem of compatibilities btwn torchio and pytorch 2.3 (See to_dict() function from MySubject in get_subjects file)
# class SubjectsDataset(tio.SubjectsDataset):
#     def __init__(self, *args, **kwargs: dict[str, Any]):
#         super().__init__(*args, **kwargs)

#     def __getitem__(self, index: int):
#         try:
#             index = int(index)
#         except (RuntimeError, TypeError):
#             message = (
#                 f'Index "{index}" must be int or compatible dtype,'
#                 f' but an object of type "{type(index)}" was passed'
#             )
#             raise ValueError(message)

#         subject = self._subjects[index]
#         subject = copy.deepcopy(subject)  # cheap since images not loaded yet
#         if self.load_getitem:
#             subject.load()

#         # Apply transform (this is usually the bottleneck)
#         if self._transform is not None:
#             subject = self._transform(subject)
#         # Here I've changed the return to a dictionary rather than a Subject
#         return subject.to_dict()
    
#     def get_subjects(self):

#         return self._subjects


config_files = ["config_unet_atlas"]
data_infos_files = ["dataset"]

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
            adaptative_lr=ctx["adaptative_lr"]
            seed = ctx["seed"]
            net_model = ctx["net_model"]
            batch_size = ctx["batch_size"]
            dropout = ctx["dropout"]
            loss_type = ctx['loss_type']
            channels = ctx["channels"]
            n_layers = len(channels)
            train_val_ratio = ctx["train_val_ratio"]
            overfit_batch = ctx["overfit_batch"]
            if ctx["patch"]:
                patch_size = ctx["patch_size"]
                queue_length = ctx["queue_length"]
                samples_per_volume = ctx["samples_per_volume"] 

    with open('./config_files/'+data_infos+".json") as f:
        data_info = json.load(f)
        channel = data_info["channel_names"]["0"]
        rootdir_train_img = data_info["rootdir_train_img"]
        rootdir_train_labels = data_info["rootdir_train_labels"]
        # rootdir_test_img = data_info["rootdir_test-cap"]
        # rootdir_test_labels = data_info["rootdir_test_labels-cap"]
        # rootdir_test_brain_mask = data_info["rootdir_test_brain_mask"]
        rootdir_train_brain_mask = data_info["rootdir_train_brain_mask"]
        suffixe_img_train = data_info["suffixe_img-train"]
        suffixe_labels_train = data_info["suffixe_labels-train"]
        # suffixe_img_test = data_info["suffixe_img-test"]
        # suffixe_labels_test = data_info["suffixe_labels-test"]
        suffixe_brain_mask_train = data_info["suffixe_brain_mask-train"]
        # suffixe_brain_mask_test = data_info["suffixe_brain_mask-test"]
        num_classes = len(data_info["labels"])
        file_ending = data_info["file_ending"]
        subsample = data_info["subset"]
        labels_names = list(data_info["labels"].keys())
        train_subjects_id = data_info['ft_subjects']
    print(f"{num_classes} classes : {labels_names}")

    sample = ""
    if subsample:
        sample = "_subset"

    current_dateTime = datetime.now()
    id_run = config_file + sample + "_" + str(current_dateTime.day) + "-" + str(current_dateTime.month) + "-" + str(current_dateTime.year) + "-" + str(current_dateTime.hour) + str(current_dateTime.minute) + str(current_dateTime.second) 
    save_model_weights = "./weights/" + id_run
    if not os.path.exists(save_model_weights):
        os.makedirs(save_model_weights)
    os.system('cp ./config_files/'+data_infos+'.json '+save_model_weights)  
    os.system('cp ./config_files/'+config_file+'.json '+save_model_weights)  
    save_model_weights_path = save_model_weights +"/"+ id_run + ".pth"

    #################
    #   MONITORING  #
    #################

    logger = TensorBoardLogger("logs", name=id_run)
    writer = SummaryWriter("logs/"+id_run)

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
    # img_test_dir=Path(rootdir_test_img)
    # print(f"Test images in : {img_test_dir}")
    brain_mask_train_dir=Path(rootdir_train_brain_mask)
    print(f"Train brain mask in : {brain_mask_train_dir}")



    ####################
    #   TRAINING DATA  #
    ####################
    print(f"\n# TRAINING DATA : \n")

    train_image_paths = sorted(img_dir.glob('**/*'+suffixe_img_train+file_ending))
    train_label_paths = sorted(labels_dir.glob('**/*'+suffixe_labels_train+file_ending))
    train_brain_mask_paths = sorted(brain_mask_train_dir.glob('**/*'+suffixe_brain_mask_train+file_ending))

    assert len(train_image_paths) == len(train_label_paths)
    assert len(train_image_paths) == len(train_brain_mask_paths)

    train_subjects = get_subjects(train_image_paths, train_label_paths, subsample, brain_mask_paths = train_brain_mask_paths)

    train_subjects_dataset = tio.SubjectsDataset(train_subjects)
    print('training dataset size: ', len(train_subjects), ' subjects')

    # train_subjects_ids = []
    # for s in train_subjects_dataset:
    #     train_subjects_ids.append(s['subject'])
    # import pandas
    # df = pandas.DataFrame(data={"sub_ids": train_subjects_ids})
    # df.to_csv("./file.csv", sep=',',index=False)

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

    transform_train = tio.Compose([preprocess(brain_mask="brain_mask"), augment()])
    transform_val = tio.Compose([preprocess(brain_mask="brain_mask")])
    transform_test = tio.Compose([preprocess(brain_mask="brain_mask")])

    #######################
    #   TRAIN VAL SPLIT   #
    #######################

    print("\n# TRAIN VAL SPLIT\n")

    if train_subjects_id==False:
        num_subjects = len(train_subjects)

        num_train_subjects = int(round(num_subjects * train_val_ratio))
        num_val_subjects = num_subjects - num_train_subjects
        splits = num_train_subjects, num_val_subjects
        generator = torch.Generator().manual_seed(seed)
        train_subjects, val_subjects = random_split(train_subjects, splits, generator=generator)

        train_subjects_dataset = tio.SubjectsDataset(train_subjects, transform=transform_train)
        val_subjects_dataset = tio.SubjectsDataset(val_subjects, transform=transform_val)
    
    else:
        train_subjects = []
        val_subjects = []
        for train_subject in train_subjects_dataset:
            sub=train_subject["subject"].split("sub-")[-1][:3]
            ses=train_subject["subject"].split("ses-")[-1][:2]
            if sub == '012' and ses == '01':
                val_subjects.append(train_subject)
            else:
                if sub in train_subjects_id.keys():
                    if ses == train_subjects_id[sub]:
                        train_subjects.append(train_subject)
                        print(f"train : {train_subject.subject}")
        transform_train = tio.Compose([preprocess(brain_mask='brain_mask'), tio.RandomFlip(axes=('LR',), flip_probability=0.2)])
        transform_val = tio.Compose([preprocess(brain_mask='brain_mask')])

    train_subjects_dataset = tio.SubjectsDataset(train_subjects, transform=transform_train)
    val_subjects_dataset = tio.SubjectsDataset(val_subjects, transform=transform_val)

    print(f"Training: {len(train_subjects_dataset)}")
    print(f"Validation: {len(val_subjects_dataset)}")     

    if ctx["patch"]:
        print("Patch")
        patch_sampler = tio.data.UniformSampler(
            patch_size=patch_size,
            # label_name='seg',
            # label_probabilities={0: 1, 1: 3},
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

        # generator_train = patch_sampler(train_subjects[0])
        # # generator_test = patch_sampler(test_subjects[0])
        # img_train=next(iter(generator_train))
        # # img_test=next(iter(generator_test))
        # img_train.plot()
        # plt.show()
        # # img_test.plot()
        # plt.show()
        # img_grid = torchvision.utils.make_grid(img_train["t1"]['data'][:,:,:,49])
        # writer.add_image('patch_image_train', img_train["t1"]['data'][:,:,:,49])
        # img_grid = torchvision.utils.make_grid(img_test["t1"]['data'][:,:,:,49])
        # writer.add_image('patch_image_test', img_test["t1"]['data'][:,:,:,49])
    else:
        print("No patch")
        val_set = val_subjects_dataset
        train_set = train_subjects_dataset



    #################
    #   LOAD DATA   #
    #################
    print("\n# LOAD DATA\n")

    train_dataloader = tio.SubjectsLoader(dataset=train_set, batch_size= batch_size, num_workers=0, pin_memory=True)
    val_dataloader = tio.SubjectsLoader(dataset=val_set, batch_size=batch_size, num_workers=0, pin_memory=True)

    batch = next(iter(train_dataloader))
    print(type(batch))  # Verif see comment line 39
    assert batch.__class__ is dict

    img = next(iter(train_dataloader))
    print(img['t1']['data'].shape)
    img_grid = torchvision.utils.make_grid(img['t1']['data'][:,:,:,:,64])
    writer.add_image('images_train', img_grid)
  
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

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        max_epochs=num_epochs, # Number of pass of the entire training set to the network
        # deterministic=True, #Might make your system slower, but ensures reproducibility
        accelerator=device, 
        devices=1,
        precision=16,
        logger=logger,
        log_every_n_steps=10,
        overfit_batches=overfit_batch,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", min_delta=0.0001, patience = 10)], #lr_monitor
        # limit_train_batches=0.1 # For fast training
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
    print("\n# SAVE MODEL\n")

    torch.save(model.state_dict(), save_model_weights_path)

    print("model saved in : " + save_model_weights_path)

    os.system('cp ./out.log ' +save_model_weights)

    print("txt logs saved in : " + save_model_weights + "/out.log")


#'''