import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision

class G_1(nn.Module):
    def __init__(self, input_size=10, name="G_1"):
        super(G_1, self).__init__()

        self.input_size = input_size
        self.model_name = name

        self.fc_in = nn.Sequential(
            nn.Linear(self.input_size, 16),
            #nn.LeakyReLU(),
            nn.Linear(16, 7 * 7)
        )

        self.encoder = nn.Sequential(
            nn.ConvTranspose2d(1, 16, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=2),
            nn.LeakyReLU()
        )

        self.encoder_2 = nn.Sequential(
            nn.ConvTranspose2d(1, 16, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=2),
            nn.LeakyReLU()
        )

        self.transformer_pre = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=1, padding=1),
            nn.Tanh()
        )

        self.transformer_pre_2 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=1, padding=1),
            nn.Tanh()
        )

        self.transformer_pre_3 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=1, padding=1),
            nn.Tanh()
        )

        self.transformer = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=1, padding=1),
            nn.Tanh()
        )

        self.transformer_2 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=1, padding=1),
            nn.Tanh()
        )

        self.transformer_out = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=1, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x, noise):
        out = self.fc_in(x)
        out = out.view(out.size(0), 1, 7, 7)

        # Conv through label and noise
        out = self.encoder(out)
        _noise = self.transformer_pre(noise)
        _noise = self.transformer_pre_2(_noise)
        _noise = self.transformer_pre_3(_noise)        

        # Add noise to output 1
        out = out + _noise
        out = self.transformer(out)
        out_2 = self.transformer_2(out)

        # Add noise to output for the last time
        merge_1 = out_2 + _noise
        out = self.transformer_out(merge_1)
        
        return out

class G_2(nn.Module):
    def __init__(self, input_size=10, name="G_2"):
        super(G_2, self).__init__()

        self.input_size = input_size
        self.model_name = name

        self.fc_in = nn.Sequential(
            nn.Linear(self.input_size, 16),
            #nn.LeakyReLU(),
            nn.Linear(16, 7 * 7)
        )

        self.encoder = nn.Sequential(
            nn.ConvTranspose2d(1, 16, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=2),
            nn.LeakyReLU()
        )

        self.encoder_2 = nn.Sequential(
            nn.ConvTranspose2d(1, 16, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=2),
            nn.LeakyReLU()
        )

        self.transformer_pre = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=1, padding=1),
            nn.Tanh()
        )

        self.transformer_pre_2 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=1, padding=1),
            nn.Tanh()
        )

        self.transformer_pre_3 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=1, padding=1),
            nn.Tanh()
        )

        self.transformer = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=1, padding=1),
            nn.Tanh()
        )

        self.transformer_2 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=1, padding=1),
            nn.Tanh()
        )

        self.transformer_out = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=1, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x, noise):
        out = self.fc_in(x)
        out = out.view(out.size(0), 1, 7, 7)

        # Conv through label and noise
        out = self.encoder(out)
        _noise = self.transformer_pre(noise)
        _noise = self.transformer_pre_2(_noise)
        _noise = self.transformer_pre_3(_noise)        

        # Add noise to output 1
        out = out + _noise
        out = self.transformer(out)
        out_2 = self.transformer_2(out)

        # Add noise to output for the last time
        merge_1 = out_2 + _noise
        out = self.transformer_out(merge_1)
        
        return out

class G_3(nn.Module):
    def __init__(self, input_size=10, name="G_3"):
        super(G_3, self).__init__()

        self.input_size = input_size
        self.model_name = name

        self.fc_in = nn.Sequential(
            nn.Linear(self.input_size, 16),
            #nn.LeakyReLU(),
            nn.Linear(16, 7 * 7)
        )

        self.encoder = nn.Sequential(
            nn.ConvTranspose2d(1, 16, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=2),
            nn.LeakyReLU()
        )

        self.encoder_2 = nn.Sequential(
            nn.ConvTranspose2d(1, 16, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=2),
            nn.LeakyReLU()
        )

        self.transformer_pre = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.transformer_pre_2 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.transformer_pre_3 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.transformer_pre_4 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=1, padding=1),
            nn.LeakyReLU()
        )        

        self.transformer_pre_5 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.transformer = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.transformer_2 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.transformer_3 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.transformer_4 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.transformer_out = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=1, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x, noise):
        out = self.fc_in(x)
        out = out.view(out.size(0), 1, 7, 7)

        # Conv through label and noise
        out = self.encoder(out)
        _noise = self.transformer_pre(noise)
        _noise = self.transformer_pre_2(_noise)
        _noise = self.transformer_pre_3(_noise)        

        # Add noise to output 1
        out = out + _noise
        out = self.transformer(out)

        _noise = self.transformer_pre_4(_noise)
        _noise = self.transformer_pre_5(_noise)

        out = out + _noise
        out_2 = self.transformer_2(out)

        # Add noise to output for the last time
        merge_1 = out_2 + _noise
        out = self.transformer_3(merge_1)
        out = self.transformer_4(out)
        out = self.transformer_out(out)
        
        return out
        
class G_label_no_conv(nn.Module):
    def __init__(self, input_size=10, name="G_label_no_conv"):
        super(G_label_no_conv, self).__init__()

        self.input_size = input_size

        self.fc_in = nn.Sequential(
            nn.Linear(self.input_size, 16),
            #nn.LeakyReLU(),
            nn.Linear(16, 29 * 29)
        )

        self.encoder = nn.Sequential(
            nn.ConvTranspose2d(1, 16, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=2),
            nn.LeakyReLU()
        )

        self.encoder_2 = nn.Sequential(
            nn.ConvTranspose2d(1, 16, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=2),
            nn.LeakyReLU()
        )

        self.noise_autoencoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=5, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(1, 1, kernel_size=5, padding=1)
        )

        self.transformer_pre = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.transformer_pre_2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.transformer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.transformer_2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.transformer_out = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=1, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x, noise):
        out = self.fc_in(x)
        out = out.view(out.size(0), 1, 29, 29)

        # Conv through label and noise
        _noise = self.transformer_pre(noise)
        _noise = self.transformer_pre_2(_noise)
        
        # Add noise to output 1
        out = out + _noise
        out = self.transformer(out)
        out = out + _noise
        out_2 = self.transformer_2(out)

        # Add noise to output for the last time
        merge_1 = out_2 + _noise
        out = self.transformer_out(merge_1)
        
        return out

class G_label_no_conv_2(nn.Module):
    def __init__(self, input_size=10, name="G_label_no_conv_2"):
        super(G_label_no_conv_2, self).__init__()

        self.input_size = input_size

        self.fc_in = nn.Sequential(
            nn.Linear(self.input_size, 16),
            #nn.LeakyReLU(),
            nn.Linear(16, 29 * 29)
        )

        self.encoder = nn.Sequential(
            nn.ConvTranspose2d(1, 16, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=2),
            nn.LeakyReLU()
        )

        self.encoder_2 = nn.Sequential(
            nn.ConvTranspose2d(1, 16, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=2),
            nn.LeakyReLU()
        )

        self.noise_autoencoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=5, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(1, 1, kernel_size=5, padding=1)
        )

        self.transformer_pre = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.transformer_pre_2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.transformer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.transformer_2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.transformer_out = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=1, stride=1, padding=1),
            nn.Tanh()
        )

    """
    Noise: [transformer_pre]->[transformer_pre_2]->add(noise)->[transformer]->add(noise)->[transformer_2]-
                                                                                                          |--> add ->[transformer_out]
    x    : [fc_in]->reshape-------------------------------------------------------------------------------
    """
    def forward(self, x, noise):
        out = self.fc_in(x)
        out = out.view(out.size(0), 1, 29, 29)

        # Conv through label and noise
        _noise = self.transformer_pre(noise)
        _noise = self.transformer_pre_2(_noise)
        _noise = _noise + noise
        _noise = self.transformer(_noise)
        _noise = _noise + noise
        _noise = self.transformer_2(_noise)
        out = out + _noise

        out = self.transformer_out(out)
        
        return out