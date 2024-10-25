from pathlib import Path
from datetime import datetime
import os 

import torch
import torchio as tio
from torch.utils.data import random_split, DataLoader
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import torchvision

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

config_file = "config_resunet_FT_CAP"
data_infos = "dataset_FT_CAP"

weights_dir = Path("./weights")

print(f"######## Finetuning model {config_file} and trained on data {data_infos}")
with open(f"./config_files/{config_file}.json") as f:
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
        weights_id = ctx["ft_weights"]
        overfit_batch = ctx["overfit_batch"]
        if ctx["patch"]:
            patch_size = ctx["patch_size"]
            queue_length = ctx["queue_length"]
            samples_per_volume = ctx["samples_per_volume"] 

with open(f"config_files/{data_infos}.json") as f:
    data_info = json.load(f)
    channel = data_info["channel_names"]["0"]
    rootdir_ft_img = data_info["rootdir-cap"]
    rootdir_ft_labels = data_info["rootdir_labels-cap"]
    rootdir_ft_brain_mask = data_info["rootdir_brain_mask-cap"]
    suffixe_img_ft = data_info["suffixe_img-cap"]
    suffixe_labels_ft = data_info["suffixe_labels-cap"]
    suffixe_brain_mask_ft = data_info["suffixe_brain_mask-cap"]
    num_classes = len(data_info["labels"])
    file_ending = data_info["file_ending"]
    subsample = data_info["subset"]
    labels_names = list(data_info["labels"].keys())
    train_subjects_id = data_info['ft_subjects']
print(f"{num_classes} classes : {labels_names}")
print(train_subjects_id)

sample = ""
if subsample:
    sample = "_subset"

current_dateTime = datetime.now()
id_run = config_file + sample + "_weights-" + weights_id + "_" + str(current_dateTime.day) + "-" + str(current_dateTime.month) + "-" + str(current_dateTime.year) + "-" + str(current_dateTime.hour) + str(current_dateTime.minute) + str(current_dateTime.second) 
model_weights_path = str(sorted(weights_dir.glob('**/*'+weights_id+'*.pth'))[0])
print(model_weights_path)
save_model_weights = "./weights/" + id_run
if not os.path.exists(save_model_weights):
    os.makedirs(save_model_weights)
os.system('cp ./config_files/'+data_infos+'.json '+save_model_weights)  
os.system('cp ./config_files/'+config_file+'.json '+save_model_weights)  
save_model_weights_path = save_model_weights +"/"+ id_run + ".pth"


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

#################
#   MONITORING  #
#################

logger = TensorBoardLogger("logs", name=id_run)
writer = SummaryWriter("logs/"+id_run)


################
#   DATA PATH  #
################
print("\n# DATA PATH : \n")

img_dir=Path(rootdir_ft_img)
print(f"Training images in : {img_dir}")
labels_dir=Path(rootdir_ft_labels)
print(f"Labels in : {labels_dir}")
brain_masks_dir=Path(rootdir_ft_brain_mask)
print(f"Train brain mask in : {brain_masks_dir}")


####################
#   TRAINING DATA  #
####################
print(f"\n# TRAINING DATA : \n")

ft_img_paths = sorted(img_dir.glob('**/*'+suffixe_img_ft+file_ending))
ft_label_paths = sorted(labels_dir.glob('**/*'+suffixe_labels_ft+file_ending))
ft_brain_mask_paths = sorted(brain_masks_dir.glob('**/*'+suffixe_brain_mask_ft+file_ending))

assert len(ft_img_paths) == len(ft_label_paths)

ft_subjects = get_subjects(ft_img_paths, ft_label_paths, subsample=False, brain_mask_paths = ft_brain_mask_paths)

ft_subjects_dataset = tio.SubjectsDataset(ft_subjects)

##########################
#   DATA TRANFORMATION   #
##########################
print("\n# DATA TRANFORMATION\n")

transform_ft = tio.Compose([preprocess(brain_mask='brain_mask'), tio.RandomFlip(axes=('LR',), flip_probability=0.2)])

#######################
#   TRAIN VAL SPLIT   #
#######################

print("\n# TRAIN VAL SPLIT\n")


train_subjects = []
val_subjects = []
test_subjects = []
for ft_subject in ft_subjects_dataset:
    sub=ft_subject["subject"].split("sub-")[-1][:3]
    ses=ft_subject["subject"].split("ses-")[-1][:2]
    if sub == '012' and ses == '01':
        val_subjects.append(ft_subject)
    else:
        if sub in train_subjects_id.keys():
            if ses == train_subjects_id[sub]:
                train_subjects.append(ft_subject)
                print(f"train : {ft_subject.subject}")
            else:
                test_subjects.append(ft_subject)
                print(f"test : {ft_subject.subject}")
        else:
            test_subjects.append(ft_subject)
            print(f"test : {ft_subject.subject}")
                
train_subjects_dataset = tio.SubjectsDataset(train_subjects, transform=transform_ft)
val_subjects_dataset = tio.SubjectsDataset(val_subjects, transform=transform_ft)
test_subjects_dataset = tio.SubjectsDataset(test_subjects, transform=transform_ft)

print(f"Training: {len(train_subjects_dataset)}")
print(f"Validation: {len(val_subjects_dataset)}")     
print(f"Test: {len(test_subjects_dataset)}")    

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
img_grid = torchvision.utils.make_grid(img['t1']['data'][:,:,:,:,32])
writer.add_image('images_train', img_grid)

#############
#   MODEL   #
#############
print(f"\n# MODEL : {net_model}\n")

model = load(model_weights_path,net_model, lr, dropout, loss_type, num_classes, channels, num_epochs)

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
    # callbacks=[EarlyStopping(monitor="val_loss", mode="min", min_delta=0.0001, patience = 10)], #lr_monitor
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