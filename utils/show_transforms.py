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


from tqdm.auto import tqdm

import matplotlib.pyplot as plt 

import sys
sys.path.append("/home/emma/Projets/stroke_lesion_segmentation_v2/")
sys.path.append("/home/emma/Projets/stroke_lesion_segmentation_v2/config_files/")
sys.path.append("/home/emma/Projets/stroke_lesion_segmentation_v2/utils/")
from transforms import augment, preprocess
from get_subjects import get_subjects
from load_model import load

#################
#   Parameters  #
#################

data_infos = "dataset"
print(f"######## data {data_infos}")

with open('./config_files/'+data_infos+".json") as f:
    data_info = json.load(f)
    suffixe_img = data_info["suffixe_img-train"]
    rootdir_img = data_info["rootdir_train_img"]
    num_classes = len(data_info["labels"])
    file_ending = data_info["file_ending"]
    labels_names = list(data_info["labels"].keys())
print(f"{num_classes} classes : {labels_names}")


current_dateTime = datetime.now()
id_run = rootdir_img.split("/")[-2] + "_" + str(current_dateTime.day) + "-" + str(current_dateTime.month) + "-" + str(current_dateTime.year) + "-" + str(current_dateTime.hour) + str(current_dateTime.minute) + str(current_dateTime.second) 


################
#   DATA PATH  #
################
print("\n# DATA PATH : \n")

img_dir=Path(rootdir_img)
print(f"Images in : {img_dir}")


####################
#   TRAINING DATA  #
####################
print(f"\n# TRAINING DATA : \n")

image_paths = sorted(img_dir.glob('**/*'+suffixe_img+file_ending))


subjects = get_subjects(image_paths)

##########################
#   DATA TRANFORMATION   #
##########################
print("\n# DATA TRANFORMATION\n")

transform = tio.Compose([preprocess(num_classes), augment()])

subjects_dataset = tio.SubjectsDataset(subjects, transform=transform)

print(f"Length: {len(subjects_dataset)}")


############
#   SAVE   #
############
print("# SAVE")

def save_img(img_set, data=""):
    
    for subject in img_set:
        print(subject)
        out_file = "./out-predictions/"+id_run+"/"+data+"/"+subject.subject
        if not os.path.exists(out_file):
            os.makedirs(out_file)
        t1 = nib.Nifti1Image(subject.t1.data.numpy().astype(float).squeeze(), subject.t1.affine)
        print(f"{subject.subject} : {subject.get_inverse_transform()}")
        nib.save(t1,f"{out_file}/{subject.subject}_t1.nii.gz")


save_img(subjects_dataset, data="train")

#'''