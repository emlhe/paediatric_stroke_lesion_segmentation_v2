import torch 
from torch import nn 
from torch.nn import functional as F
from torch.autograd import Variable
from functools import partial
import monai 
import matplotlib.pyplot as plt 

#####################
#       SOURCE      #  
#####################

# https://github.com/Tencent/MedicalNet
# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
# https://github.com/ternaus/TernausNet
# https://github.com/ternaus/angiodysplasia-segmentation/blob/master/models.py


############################
#       RESNET (ENCODER)   #
############################

class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_seg_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
            
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilation=4)

        self.conv_seg = nn.Sequential(
                                        nn.ConvTranspose3d(
                                        512 * block.expansion,
                                        32,
                                        2,
                                        stride=2
                                        ),
                                        nn.BatchNorm3d(32),
                                        nn.ReLU(inplace=True),
                                        nn.Conv3d(
                                        32,
                                        32,
                                        kernel_size=3,
                                        stride=(1, 1, 1),
                                        padding=(1, 1, 1),
                                        bias=False), 
                                        nn.BatchNorm3d(32),
                                        nn.ReLU(inplace=True),
                                        nn.Conv3d(
                                        32,
                                        num_seg_classes,
                                        kernel_size=1,
                                        stride=(1, 1, 1),
                                        bias=False) 
                                        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False), 
                nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv_seg(x)

        return x

#------------------------------------------------------------------------------#

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

        self.model = ResNet(
                BasicBlock,
                [1,1,1,1], # Pour 10 couches, [2,2,2,2] pour 18 couches, [3, 4, 6, 3] pour 34, [3, 4, 23, 3] pour 101 ... voir : https://github.com/Tencent/MedicalNet/blob/master/models/resnet.py
                num_seg_classes=2)
        
        if pretrained :
            # model = monai.networks.nets.ResNet('basic' , [2,2,2,2], [2,2,2,2], spatial_dims=3, n_input_channels=1 )
            net_dict = self.model.state_dict() 
            pretrain = torch.load("/home/emma/Projets/dl_brain_MRI_segmentation/models_weights/MedNet/resnet_10_23dataset.pth")           
            pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
            net_dict.update(pretrain_dict)
            self.model.load_state_dict(net_dict)
            self.model.eval()
            # fmodel = torch.jit.freeze(model) # On freeze le modèle pour ne pas entraîner les paramètres
            # for param in self.model.parameters():
            #     param.requires_grad = False
        
        self.encoder = self.model

        # self.encoder = torchvision.models.resnet34(pretrained=pretrained) # marche pas car 2D

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlock(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlock(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlock(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
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

#------------------------------------------------------------------------------#

#####################
#       UNET        #
#####################

# With resnet encoder and attention blocks


class UnetConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1), init_stride=(1,1,1)):
        super(UnetConv3, self).__init__()

        # self.up = nn.Upsample(scale_factor=2, mode='trilinear')

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True), )
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True), )            
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.ReLU(inplace=True), )
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True), )
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.up(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, g_in_channels, x_in_channels, spatial_dims = 5):
        super().__init__()
        self.g_in_channels = g_in_channels
        self.x_in_channels = x_in_channels
        self.out_channels = g_in_channels // 2

        self.upconv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear'),
                # ConvRelu(self.in_channels, self.middle_channels),
                ConvRelu(self.g_in_channels, self.out_channels),
            )

        self.W_g  = nn.Sequential(
            nn.Conv3d(self.g_in_channels // 2, self.out_channels, kernel_size = 1, stride = 1, padding=0),
            nn.BatchNorm3d(self.g_in_channels // 2),
        ) 

        self.W_x = nn.Sequential(
            nn.Conv3d(self.x_in_channels, self.out_channels, kernel_size = 1, stride = 1, padding=0),
            nn.BatchNorm3d(self.x_in_channels),
        )

        self.psi = nn.Sequential(
            nn.Conv3d(self.out_channels, 1, kernel_size = 1, stride = 1, padding=0),
            nn.BatchNorm3d(1),
        )

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.merge = nn.Conv3d(self.g_in_channels, self.x_in_channels, kernel_size = 1)

    def forward(self, g, x1):

        fromlower = self.upconv(g)
        x1 = self.W_x(x1)
        g = self.W_g(fromlower)
        psi = self.relu(x1+g)
        psi = self.psi(psi)
        psi = self.sigmoid(psi)
        att = x1 * psi # si psi = 0, x1 = 0, sinon x1 = x1
        att_m = torch.cat((att, fromlower), dim=1)
        att_m = self.merge(att_m)
        att_m = self.relu(att_m)

        return att_m
    

class Attention_Res_UNet(nn.Module):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder
        Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
        """

    def __init__(self, num_classes=1, num_filters=32, pretrained=True, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool3d(2)

        self.model = ResNet(
                BasicBlock,
                [1,1,1,1], # Pour 10 couches, [2,2,2,2] pour 18 couches, [3, 4, 6, 3] pour 34, [3, 4, 23, 3] pour 101 ... voir : https://github.com/Tencent/MedicalNet/blob/master/models/resnet.py
                num_seg_classes=2)
        
        if pretrained:
            net_dict = self.model.state_dict() 
            pretrain = torch.load("../models_weights/MedNet/resnet_10_23dataset.pth")
            pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
            net_dict.update(pretrain_dict)
            self.model.load_state_dict(net_dict)
            self.model.eval()
            # fmodel = torch.jit.freeze(model) # On freeze le modèle pour ne pas entraîner les paramètres
            # for param in self.model.parameters():
            #     param.requires_grad = False
    
        
        self.encoder = self.model

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = UnetConv3(512, 512 + num_filters * 8 * 2, is_batchnorm = True) #1024

        
        self.dec5 = AttentionBlock(1024, 512) # 512
        self.dec4 = AttentionBlock(512, 256)  # 256
        self.dec3 = AttentionBlock(256, 128)  # 128
        self.dec2 = AttentionBlock(128, 64)  # 64
        self.dec1 = DecoderBlock(num_filters * 2, num_filters * 2, num_filters, is_deconv)
        self.dec0 = DecoderBlock(num_filters, num_filters, num_filters, is_deconv)
        
        self.convRelu = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv3d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))
        
        dec5 = self.dec5(center, conv5)
        dec4 = self.dec4(dec5, conv4)
        dec3 = self.dec3(dec4, conv3)
        dec2 = self.dec2(dec3, conv2)
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        dec0 = self.convRelu(dec0)


        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out = self.final(dec0)

        return x_out
