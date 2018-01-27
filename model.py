import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision

class GenerativeModel(nn.Module):
    def __init__(self, dimen_in, dimen_out):
        super(GenerativeModel, self).__init__()
        self.dimen_in = dimen_in
        self.dimen_out = dimen_out

        self.conv1x16 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )

        self.conv16x32 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )

        self.conv32x64 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )

        self.conv64x64 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )

        self.conv64x32 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )

        self.conv32x16 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )

        self.conv16x1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=2, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv1x16(x)        # (batch, in_x, in_y, in_z)
        x = self.conv16x32(x)       # (batch, in_x / 2, in_y / 2, in_z)
        x = self.conv32x64(x)
        x = self.conv64x64(x)
        x = self.conv64x64(x)        
        x = self.conv64x32(x)
        x = self.conv32x16(x)
        x = self.conv16x1(x)
        x = x.view(x.size(0), -1)
        
        size = 1
        for i, n in enumerate(x.size()):
            if i > 0:
                size *= n

        x = nn.Linear(size, 10)
        return x

_g = GenerativeModel((64, 64, 3), (64, 64, 3))
print(_g)

