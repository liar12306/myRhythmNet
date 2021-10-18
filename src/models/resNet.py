from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
class Residual_Block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(Residual_Block, self).__init__()
        self.same_shape = same_shape
        stride = 1 if self.same_shape else 2
        self.conv1 = nn.Sequential(
            self._conv3x3(in_channel, out_channel, stride=stride),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            self._conv3x3(out_channel, out_channel),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )
        if not self.same_shape:
            self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride,bias=False)

    def forward(self, x):
        out = self.conv1(x)
        if not self.same_shape:
            x = self.conv2(x)
        return F.relu(x + out, True)

    def _conv3x3(self, in_channel, out_channel, stride=1, padding=1):
        return nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=padding, bias=False)


class ResNet18(nn.Module):

    def __init__(self, num_classes=1):
        super(ResNet18, self).__init__()
        self.block1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2)
        self.block2 = nn.Sequential(
            nn.MaxPool2d(3, 2),
            Residual_Block(64, 64),
            Residual_Block(64, 64),

        )
        self.block3 = nn.Sequential(
            Residual_Block(64, 128, False),
            Residual_Block(128, 128),
        )
        self.block4 = nn.Sequential(
            Residual_Block(128, 256, False),
            Residual_Block(256, 256),

        )
        self.block5 = nn.Sequential(
            Residual_Block(256, 512, False),
            Residual_Block(512, 512),

            nn.AdaptiveAvgPool2d(output_size=(1,1))
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, st_maps):
        hr_per_clip = []
        for t in range(st_maps.size(1)):
            # st_maps[ batch_size, T, channel, slide_window_size, roi_nums]
            x = st_maps[:,t,:,:,:]
            x = self.block1(x)
            print(x.shape)
            x = self.block2(x)
            print(x.shape)
            x = self.block3(x)
            print(x.shape)
            x = self.block4(x)
            print(x.shape)
            x = self.block5(x)
            print(x.shape)
            # output dim Batch_Size×512
            x = x.view(x.size(0),-1)
            # output dim Batch_Size×1
            x = self.fc(x)
            x = x.squeeze(0)
            hr_per_clip.append(x)
        return torch.stack(hr_per_clip,dim=1).squeeze(0)

if __name__ == "__main__":
    net = ResNet18()
    input = torch.rand(1,18,3,300,25)
    output = net(input)
    print(output.size())

