import torch
import torch.nn as nn
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class build_unet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = conv_block(512, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)

        return outputs




#---------------------------MODIFY
class build_unetpp(nn.Module):
    def __init__(self):
        super().__init__()









# class EarlyStopping:
#     def __init__(self, tolerance=5, min_delta=0):

#         self.tolerance = tolerance
#         self.min_delta = min_delta
#         self.counter = 0
#         self.early_stop = False

#     def __call__(self, train_loss, validation_loss):
#         if (validation_loss - train_loss) > self.min_delta:
#             self.counter +=1
#             if self.counter >= self.tolerance:  
#                 self.early_stop = True








""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)



class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

import segmentation_models_pytorch
from torchsummary import summary

def get_model(MODEL_ARCH, ENCODER_NAME, classes, in_channels = 3):
    print(MODEL_ARCH)
    if  MODEL_ARCH == "Unet":
        model = build_unet()

    elif MODEL_ARCH == "unet":
        model = segmentation_models_pytorch.Unet(encoder_name = ENCODER_NAME,
                                             encoder_weights='imagenet',
                                             decoder_use_batchnorm=True,
                                             decoder_channels=(1024, 512, 256, 128, 64),
                                             decoder_attention_type=None,
                                             in_channels=in_channels,
                                             classes=classes,
                                             activation=None,
                                             aux_params=None)
        # print(summary(model, (3, 512, 512)))
    elif MODEL_ARCH == "UnetPlusPlus":

        model = segmentation_models_pytorch.UnetPlusPlus(encoder_name = ENCODER_NAME,
                                             encoder_weights='imagenet',
                                             decoder_use_batchnorm=True,
                                             decoder_channels=(1024, 512, 256, 128, 64),
                                             decoder_attention_type=None,
                                             in_channels=in_channels,
                                             classes=classes,
                                             activation=None,
                                             aux_params=None)

    elif MODEL_ARCH == "FPN":
        model = segmentation_models_pytorch.FPN(
            encoder_name = ENCODER_NAME,
            encoder_depth=5,
            encoder_weights='imagenet',
            decoder_pyramid_channels=256,
            decoder_segmentation_channels=128,
            decoder_merge_policy='add',
            decoder_dropout=0.2,
            in_channels=in_channels,
            classes=classes,
            activation=None,
            upsampling=4,
            aux_params=None)


    elif MODEL_ARCH == "PSPNet":

        model = segmentation_models_pytorch.PSPNet(
            encoder_name=ENCODER_NAME,
            encoder_weights='imagenet',
            encoder_depth=3,
            psp_out_channels=512,
            psp_use_batchnorm=True,
            psp_dropout=0.2,
            in_channels=in_channels,
            classes=classes,
            activation=None,
            upsampling=8,
            aux_params=None)

    elif MODEL_ARCH == "Linknet":
        model  = segmentation_models_pytorch.Linknet(
            encoder_name=ENCODER_NAME, 
            encoder_depth=5, 
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            in_channels=in_channels,
            classes=classes,
            activation=None,
            aux_params=None)        

    elif MODEL_ARCH == "MAnet":
        model = segmentation_models_pytorch.MAnet(
            encoder_name=ENCODER_NAME,
            encoder_depth=5,
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            decoder_channels=(1024, 512, 256, 128, 64),
            decoder_pab_channels=64,
            in_channels=in_channels,
            classes=classes,
            activation=None,
            aux_params=None)

    # elif MODEL_ARCH == "PAN":
    #     model = segmentation_models_pytorch.PAN(
    #         encoder_name=ENCODER_NAME,
    #         encoder_weights='imagenet',
    #         encoder_output_stride=16,
    #         decoder_channels=1024,
    #         in_channels=in_channels,
    #         classes=classes,
    #         activation=None,
    #         upsampling=4,
    #         aux_params=None)
    # elif MODEL_ARCH == "UPerNet":
    #     model = segmentation_models_pytorch.UPerNet(
    #         encoder_name=ENCODER_NAME,
    #         encoder_depth=5,
    #         encoder_weights='imagenet',
    #         decoder_pyramid_channels=256,
    #         decoder_segmentation_channels=64,
    #         in_channels=in_channels,
    #         classes=classes,
    #         activation=None,
    #         aux_params=None)

    return model
if __name__ == "__main__":
    x = torch.randn((2, 3, 512, 512))
    f = build_unet()
    f = UNet(3, 2)
    y = f(x)
    print(y.shape)


