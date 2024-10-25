import torch 
from torch import nn 
from torch.nn import functional as F
from torch.autograd import Variable
from functools import partial
from monai.networks.nets import resnet10
import matplotlib.pyplot as plt 

import warnings

from collections.abc import Sequence

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection

#####################
#       SOURCE      #  
#####################

# https://github.com/Tencent/MedicalNet
# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
# https://github.com/ternaus/TernausNet
# https://github.com/ternaus/angiodysplasia-segmentation/blob/master/models.py


#####################
#       UNET        #
#####################

# With resnet encoder

 

class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.conv = nn.Conv3d(in_, out, 3, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x
    
def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)
     
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels
        self.middle_channels = middle_channels
        self.out_channels = out_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(self.in_channels, self.middle_channels),
                nn.ConvTranspose3d(self.middle_channels, self.out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear'),
                ConvRelu(self.in_channels, self.middle_channels),
                ConvRelu(self.middle_channels, self.out_channels),
            )

    def forward(self, x):

        return self.block(x)
    

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    


class Res_UNet(nn.Module):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder
        Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
        """

    def __init__(self, num_classes=1, num_filters=32, pretrained=False, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet10
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()     

        self.num_classes = num_classes

        self.pool = nn.MaxPool3d(2)

        # self.model = 
        
        # if pretrained :
        #     # model = monai.networks.nets.ResNet('basic' , [2,2,2,2], [2,2,2,2], spatial_dims=3, n_input_channels=1 )
        #     net_dict = self.model.state_dict() 
        #     pretrain = torch.load("/home/emma/Projets/dl_brain_MRI_segmentation/models_weights/MedNet/resnet_10_23dataset.pth")           
        #     pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
        #     net_dict.update(pretrain_dict)
        #     self.model.load_state_dict(net_dict)
        #     self.model.eval()
            # fmodel = torch.jit.freeze(model) # On freeze le modèle pour ne pas entraîner les paramètres
            # for param in self.model.parameters():
            #     param.requires_grad = False
        
        self.encoder = resnet10(pretrained=True, n_input_channels=1, feed_forward=False, spatial_dims=3, bias_downsample=False)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.act, # In monai v 1.3.2 instead of relu 
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlock(256, num_filters * 8 * 2, num_filters * 8, is_deconv)#[4,8,16,32,64]

        self.dec5 = DecoderBlock(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlock(128 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlock(64 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlock(32 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv3d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))

        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out = self.final(dec0)

        return x_out

