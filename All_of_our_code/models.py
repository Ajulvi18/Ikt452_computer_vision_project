import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

torch.manual_seed(42)
np.random.seed(42)
#from post_processing import MorphologicalOperations, MorphologicalOperations_
#from torchvision import models
# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
# https://github.com/zhixuhao/unet/blob/master/model.py

#https://github.com/Sjyhne/MapAI-Competition/blob/master/team_fundator/src/ensemble_model.py

class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models
        self.weights = nn.Parameter(torch.ones(len(self.models)))
        self.softmax = nn.Softmax(dim=1)
    def forward(self, input):
        model_predictions = []
        for i, model in enumerate(self.models):
            model_prediction = model(input)["out"]
            model_predictions.append(self.weights[i] * model_prediction)
        model_predictions = torch.stack(model_predictions)
        summed_predictions = torch.sum(model_predictions, dim=0)
        #summed_predictions = self.softmax(summed_predictions)
        return summed_predictions


class Unet(nn.Module):
    def __init__(self, in_channels=3):
        super(Unet, self).__init__()

        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.down5 = DoubleConv(512, 1024)

        self.up4 = Up(1024, 512)
        self.up3 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up1 = Up(128, 64)

        self.post1 = nn.Conv2d(64, 2, kernel_size=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # post processing
        #self.morph_kernel = torch.nn.Parameter(torch.ones(2, 5, 5))
        #self.morphological_operations = MorphologicalOperations(kernel_size=5)

    def forward(self, input):
        # encode
        down1 = self.down1(input)
        pool1 = self.pool(down1)

        down2 = self.down2(pool1)
        pool2 = self.pool(down2)

        down3 = self.down3(pool2)
        pool3 = self.pool(down3)

        down4 = self.down4(pool3)
        pool4 = self.pool(down4)

        down5 = self.down5(pool4)

        # decode
        up4 = self.up4(down5, down4)
        up3 = self.up3(up4, down3)
        up2 = self.up2(up3, down2)
        up1 = self.up1(up2, down1)

        output = self.post1(up1)
        #output = self.softmax(output)

        # post-processing
        #if not self.training:
        #    output = self.morphological_operations(output)#self.morphological_operations(output).to(input.device)

        return {'out': output}


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=True)  # nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=2)
        self.conv2 = DoubleConv(in_channels, out_channels)
        self.relu = nn.ReLU()
        self.batch_normalization = nn.BatchNorm2d(out_channels)

    def forward(self, input, merging_layer):
        x = self.upsample(input)
        values = merging_layer.size(-1) - x.size(-1)
        padded_x = F.pad(x, pad=(values, 1, values, 1), mode='constant', value=0)

        x = self.conv1(padded_x)
        x = self.relu(x)

        merged = torch.cat([x, merging_layer], dim=1)
        x = self.conv2(merged)
        x = self.batch_normalization(x)

        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Dropout(0.5)
        )

    def forward(self, input):
        x = self.seq(input)

        return x

class Unet2(nn.Module):
    def __init__(self, in_channels=3):
        super(Unet2, self).__init__()

        self.down1 = DoubleConv(in_channels, 32)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128)
        self.down4 = DoubleConv(128, 256)
        self.down5 = DoubleConv(256, 512)

        self.up4 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up1 = Up(64, 32)

        self.post1 = nn.Conv2d(32, 1, kernel_size=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # post processing
        #self.morph_kernel = torch.nn.Parameter(torch.ones(2, 5, 5))
        #self.morphological_operations = MorphologicalOperations(kernel_size=5)

    def forward(self, input):
        # encode
        down1 = self.down1(input)
        pool1 = self.pool(down1)

        down2 = self.down2(pool1)
        pool2 = self.pool(down2)

        down3 = self.down3(pool2)
        pool3 = self.pool(down3)

        down4 = self.down4(pool3)
        pool4 = self.pool(down4)

        down5 = self.down5(pool4)

        # decode
        up4 = self.up4(down5, down4)
        up3 = self.up3(up4, down3)
        up2 = self.up2(up3, down2)
        up1 = self.up1(up2, down1)

        output = self.post1(up1)
        return {'out': output}

class Unet3(nn.Module):
    def __init__(self, in_channels=3):
        super(Unet3, self).__init__()

        self.down1 = DoubleConv(in_channels, 16)
        self.down2 = DoubleConv(16, 32)
        self.down3 = DoubleConv(32, 64)
        self.down4 = DoubleConv(64, 128)

        self.up3 = Up(128, 64)
        self.up2 = Up(64, 32)
        self.up1 = Up(32, 16)

        self.post1 = nn.Conv2d(16, 1, kernel_size=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.sigmoid = nn.Sigmoid()
        # post processing
        # self.morph_kernel = torch.nn.Parameter(torch.ones(2, 5, 5))
        # self.morphological_operations = MorphologicalOperations(kernel_size=5)

    def forward(self, input):
        # encode
        down1 = self.down1(input)
        pool1 = self.pool(down1)

        down2 = self.down2(pool1)
        pool2 = self.pool(down2)

        down3 = self.down3(pool2)
        pool3 = self.pool(down3)

        down4 = self.down4(pool3)

        up3 = self.up3(down4, down3)
        up2 = self.up2(up3, down2)
        up1 = self.up1(up2, down1)

        output = self.post1(up1)

        output = self.sigmoid(output)
        return {'out': output}