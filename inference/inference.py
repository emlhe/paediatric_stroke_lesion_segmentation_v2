from pathlib import Path
from datetime import datetime
import os 

import torch
import torchio as tio
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import numpy as np
import pandas as pd
from nilearn import plotting
import nibabel as nib
import monai
import json
import sys


from tqdm.auto import tqdm

import matplotlib.pyplot as plt 

from utils.transforms import augment, preprocess
from utils.get_subjects import get_subjects
from utils.load_model import load

#################
#   Parameters  #
#################

weights_ids = [str(sys.argv[1])]

for i in range(len(weights_ids)):
    weights_id = weights_ids[i]
    weigts_dir = Path("./weights")
    data_infos_path = str(sorted(weigts_dir.glob('**/*'+weights_id+'/dataset*.json'))[0])
    config_file_path = str(sorted(weigts_dir.glob('**/*'+weights_id+'/config*.json'))[0])
    config_file = config_file_path.split("/")[-1].split(".json")[0]
    data_info_file = data_infos_path.split("/")[-1].split(".json")[0]
    print(f"######## Inference with config {config_file_path} and data {data_infos_path} for weights {weights_id}")
    with open(config_file_path) as f:
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

    with open(data_infos_path) as f:
        data_info = json.load(f)
        channel = data_info["channel_names"]["0"]
        rootdir_train_img = data_info["rootdir_train_img"]
        rootdir_train_labels = data_info["rootdir_train_labels"]
        rootdir_test_img = data_info["rootdir_test-cap"]
        rootdir_test_labels = data_info["rootdir_test_labels-cap"]
        rootdir_test_brain_mask = data_info["rootdir_test_brain_mask"]
        rootdir_train_brain_mask = data_info["rootdir_train_brain_mask"]
        suffixe_img_train = data_info["suffixe_img-train"]
        suffixe_labels_train = data_info["suffixe_labels-train"]
        suffixe_img_test = data_info["suffixe_img-test"]
        suffixe_labels_test = data_info["suffixe_labels-test"]
        suffixe_brain_mask_train = data_info["suffixe_brain_mask-train"]
        suffixe_brain_mask_test = data_info["suffixe_brain_mask-test"]
        num_classes = len(data_info["labels"])
        file_ending = data_info["file_ending"]
        subsample = data_info["subset"]
        labels_names = list(data_info["labels"].keys())
    print(f"{num_classes} classes : {labels_names}")

    sample = ""
    if subsample:
        sample = "_subset"

    current_dateTime = datetime.now()
    id_run = config_file + sample + "_weights-" + weights_id + "_" + str(current_dateTime.day) + "-" + str(current_dateTime.month) + "-" + str(current_dateTime.year) + "-" + str(current_dateTime.hour) + str(current_dateTime.minute) + str(current_dateTime.second) 
    model_weights_path = str(sorted(weigts_dir.glob('**/*'+weights_id+'.pth'))[0])
    print(f"Inference with weights : {model_weights_path}")


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
    brain_masks_train_dir=Path(rootdir_train_brain_mask)
    print(f"Train brain mask in : {brain_masks_train_dir}")
    brain_masks_test_dir = Path("/home/emma/data/IRM-CAP/derivatives/Brain_extraction/FS-SynthStrip/CAP-stroke")
    print(f"Test brain mask in : {brain_masks_test_dir}")
    


    ####################
    #   TRAINING DATA  #
    ####################
    print(f"\n# TRAINING DATA : \n")

    train_image_paths = sorted(img_dir.glob('**/*'+suffixe_img_train+file_ending))
    train_label_paths = sorted(labels_dir.glob('**/*'+suffixe_labels_train+file_ending))
    train_brain_mask_paths = sorted(brain_masks_train_dir.glob('**/*'+suffixe_brain_mask_train+file_ending))

    test_img_paths = sorted(img_test_dir.glob('**/*'+suffixe_img_test+file_ending))
    test_label_paths = sorted(img_test_dir.glob('**/*'+suffixe_labels_test+file_ending))
    test_brain_mask_paths = sorted(brain_masks_test_dir.glob('**/*'+suffixe_brain_mask_test+file_ending))

    assert len(train_image_paths) == len(train_label_paths)
    assert len(test_img_paths) == len(test_label_paths)

    train_subjects = get_subjects(train_image_paths, train_label_paths, subsample, brain_mask_paths = train_brain_mask_paths)

    test_subjects = get_subjects(test_img_paths, test_label_paths, subsample=False, brain_mask_paths = test_brain_mask_paths)

    train_subjects_dataset = tio.SubjectsDataset(train_subjects)

    ##########################
    #   DATA TRANFORMATION   #
    ##########################
    print("\n# DATA TRANFORMATION\n")

    transform_train = tio.Compose([preprocess(brain_mask='brain_mask'), augment()])
    transform_val = tio.Compose([preprocess(brain_mask='brain_mask')])
    transform_test = tio.Compose([preprocess(brain_mask='brain_mask')])

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

    train_subjects_dataset = tio.SubjectsDataset(train_subjects)#, transform=transform_train)
    val_subjects_dataset = tio.SubjectsDataset(val_subjects)#, transform=transform_val)
    test_subjects_dataset = tio.SubjectsDataset(test_subjects)#, transform=transform_test)

    for subject in test_subjects_dataset._subjects:
        print("####")
        print(subject)

    print(f"Training: {len(train_subjects_dataset)}")
    print(f"Validation: {len(val_subjects_dataset)}")     
    print(f"Test: {len(test_subjects_dataset)}")    

    #############
    #   MODEL   #
    #############
    print(f"\n# MODEL : {net_model}\n")

    model = load(model_weights_path,net_model, lr, dropout, loss_type, num_classes, channels, num_epochs)

    #################
    #   INFERENCE   #
    #################
    print("# INFERENCE")

    model.eval()        
    def inference(img_set, save = True, metric = False, df = None, data = "", patch = False, transform = None):
        get_dice = monai.metrics.DiceMetric(include_background=False, reduction="none")
        get_hd = monai.metrics.HausdorffDistanceMetric(include_background=False, reduction="none", percentile=95)
        subjects_list = []

        # names = [subject.subject for subject in img_set.dry_iter()]
        # print(names)
        print(len(img_set))
    
        for subject in img_set.dry_iter(): # Suggested in https://github.com/fepegar/torchio/issues/256
            print(subject)
            new_sub = transform(subject)
            shape = new_sub['t1'][tio.DATA].shape
            print(shape)
            new_shape = []
            for s in shape[1:]:
                if s<128:
                    new_shape.append(128)
                else:
                    new_shape.append(s) 
            print(new_shape)
            transform_test = tio.Compose([preprocess(brain_mask='brain_mask', crop=new_shape)])
            print(subject)
            subject = transform_test(subject)
            print(subject['t1'][tio.DATA].shape)
            
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
                    print("Patch")      
                    for batch in tqdm(loader, unit='batch'):
                        input = batch['t1'][tio.DATA]
                        print(input.shape)
                        logits = model.net(input)
                        labels = logits.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True)
                        locations = batch[tio.LOCATION]
                        aggregator.add_batch(labels, locations)                     
                else:
                    print("No patch")
                    input = subject['t1'][tio.DATA].unsqueeze(axis=0)
                    print(input.shape)
                    logits = model.net(input)
                    labels = logits.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True) 
                        
            if patch:
                output_tensor = aggregator.get_output_tensor().to(torch.int64).unsqueeze(axis=0)
            else:
                output_tensor = labels

            
            print(f"output tensor shape : {output_tensor.shape}")
            # plotting.plot_t1(nib.Nifti1Image(input.cpu().numpy().squeeze(), subject.t1.affine), display_mode='tiled')
            if metric:
                gt_tensor = subject['seg'][tio.DATA].unsqueeze(axis=0) # GT tensor must be one hot encoding ==> preprocess transform the data + 5D to get metrics (batch size = 1)
                print(f"gt tensor shape : {gt_tensor.shape}")
                outputs_one_hot = torch.nn.functional.one_hot(output_tensor, num_classes=num_classes).squeeze(axis=1)
                print(f"outputs_one_hot shape : {outputs_one_hot.shape}")
                outputs_one_hot = outputs_one_hot.permute(0, 4, 1, 2, 3)
                print(f"outputs_one_hot shape permuted: {outputs_one_hot.shape}")

                get_dice(outputs_one_hot.to(model.device), gt_tensor.to(model.device))
                get_hd(outputs_one_hot.to(model.device), gt_tensor.to(model.device))

                print(f"subjects : {subjects_list}")
                dice = get_dice.aggregate().cpu().numpy()[0]
                print(f"DICE : {dice[0]}")
                get_dice.reset()
                hd = get_hd.aggregate().cpu().numpy()[0]
                print(f"HD : {hd[0]}")
                get_hd.reset()
                new_data = [[subject['subject']] + [dice[0]] + [hd[0]]]
                print(pd.DataFrame(data=new_data, columns = df.columns))
                df = pd.concat([df, pd.DataFrame(data=new_data, columns = df.columns)], ignore_index=True)
               
            pred = tio.LabelMap(tensor=outputs_one_hot.squeeze(axis=0), affine=subject.t1.affine, type=tio.LABEL) # tensor must be 4D
            # Nécessaire de Crop à la taille du cerveau car on a fait un pad pour ensure shape multiple ce qui a ajouté une bordure dans laquelle le modèle segmente
            subject.add_image(pred, "prediction")
            new_subject = subject.apply_inverse_transform()

        
            if save :
                out_file = "./out-predictions/"+id_run+"/"+data+"/"+subject.subject
                if not os.path.exists(out_file):
                    os.makedirs(out_file)
                print(f"prediction shape : {new_subject.prediction.data.to(torch.float).squeeze().numpy().shape}")
                print(f"output tensor shape {output_tensor.shape}")
                print(f"GT shape {new_subject.seg.data.to(torch.float).numpy().squeeze().shape}")
                print(f"T1w shape {new_subject.t1.data.to(torch.float).numpy().squeeze().shape}")
                pred = nib.Nifti1Image(new_subject.prediction.data.to(torch.float).numpy()[1,...].squeeze(), new_subject.t1.affine)
                gt = nib.Nifti1Image(new_subject.seg.data.to(torch.float).numpy().squeeze()[1,...], new_subject.t1.affine)
                t1 = nib.Nifti1Image(new_subject.t1.data.to(torch.float).numpy().squeeze(), new_subject.t1.affine)
                nib.save(pred,f"{out_file}/{subject.subject}_pred.nii.gz")
                nib.save(t1,f"{out_file}/{subject.subject}_t1w.nii.gz")
                nib.save(gt,f"{out_file}/{subject.subject}_seg.nii.gz")
        if metric:
            return df

    df_labels = pd.DataFrame(columns=["Subjects"] + ["DICE"] + ["HD 95"])

    # print("# Test")
    # df_labels_test = pd.DataFrame(columns=["Subjects", "DICE", "HD 95"])
    # df_test = inference(test_subjects_dataset, metric = True, df = df_labels_test, data="test", patch = ctx["patch"], transform = transform_test)
    # df_test.to_csv("./out-predictions/"+id_run+"/scores_test.csv", index=False)
    # print(df_test)
    # print(f"\t# Mean DICE = {df_test.loc[:, 'DICE'].mean()} +- {df_test.loc[:, 'DICE'].std()}")
    # print(f"\t# Mean HD95 = {df_test.loc[:, 'HD 95'].mean()} +- {df_test.loc[:, 'HD 95'].std()}")

    # print("# Validation")
    df_val = inference(val_subjects_dataset, metric = True, df = df_labels, data="val", patch = ctx["patch"], transform = transform_val)
    df_val.to_csv("./out-predictions/"+id_run+"/scores_val.csv", index=False)
    print(df_val)
    print(f"\t# Mean DICE = {df_val.loc[:, 'DICE'].mean()} +- {df_val.loc[:, 'DICE'].std()}")
    print(f"\t# Mean HD95 = {df_val.loc[:, 'HD 95'].mean()} +- {df_val.loc[:, 'HD 95'].std()}")
    
    # print("# Training")
    # df_train = inference(train_subjects_dataset, metric = True, df = df_labels, data="train", patch = ctx["patch"], transform = transform_train)
    # df_train.to_csv("./out-predictions/"+id_run+"/scores_train.csv", index=False)

#'''
