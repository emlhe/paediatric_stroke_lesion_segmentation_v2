import monai 
from monai.networks.nets import UNet, ResNetFeatures, FlexibleUNet, ResNetEncoder
import torch 
from monai.networks.layers.factories import Norm
import pytorch_lightning as pl
import torchio as tio
import numpy as np

import sys
from Model.ResUNet import Res_UNet
from Model.Model import Model

def load(weights_path, model, lr, dropout, loss_type, n_class, channels, epochs):
    if model == "unet":
        net = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=n_class,
            channels=channels, #(24, 48, 96, 192, 384),#(32, 64, 128, 256, 320, 320),#
            strides=np.ones(len(channels)-1, dtype=np.int8)*2,#(2, 2, 2, 2),
            norm = Norm.BATCH,
            dropout=dropout
        )
        optim=torch.optim.AdamW

    elif model == "resunet":
        # net = Res_UNet(num_classes=n_class, pretrained = True)
        # features = ResNetFeatures("resnet10", pretrained=True, spatial_dims=3, in_channels=1)
        net = FlexibleUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            backbone="resnet10", 
            pretrained=True, # "MedicalNet weights are available for residual networks if spatial_dims=3 and in_channels=1"
            norm = Norm.BATCH 
            )
        
        optim=torch.optim.SGD

    if loss_type == "Dice":
        crit = monai.losses.DiceLoss(include_background=True,
        to_onehot_y=False,
        sigmoid=False,
        softmax=True,
        other_act=None,
        squared_pred=False,
        jaccard=False,
        reduction="mean",
        smooth_nr=1e-05,
        smooth_dr=1e-05,
        batch=True)# monai.losses.GeneralizedWassersteinDiceLoss
    elif loss_type == "DiceCE":
        crit = monai.losses.DiceCELoss(include_background=True,
        to_onehot_y=False,
        sigmoid=False,
        softmax=True,
        other_act=None,
        squared_pred=False,
        jaccard=False,
        reduction="mean",
        smooth_nr=1e-05,
        smooth_dr=1e-05,
        batch=True)# monai.losses.GeneralizedWassersteinDiceLoss
        
    model = Model(
        net=net,
        criterion= crit,
        learning_rate=lr,
        optimizer_class=optim,
        epochs = epochs,
    )

    if weights_path != None:
        model.load_state_dict(torch.load(weights_path))
        # model.eval() # deactivate dropout layers https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323
    return model

