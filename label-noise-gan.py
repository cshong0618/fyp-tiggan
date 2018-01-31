import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.autograd import Variable

import pathlib
import os

import matplotlib.pyplot as plt

# Hyperparameters
epochs = 5
batch_size = 128
learning_rate = 1e-3

# MNIST dataset
train_dataset = datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(),
download=True)

test_dataset = datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor())

#Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Discrimination model
class D(nn.Module):
    def __init__(self, input_channels=1, output_size=10):
        super(D, self).__init__()

        self.input_channels = input_channels
        self.output_size = output_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_out = nn.Linear(7 * 7 * 32, self.output_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc_out(out)
        return out

# Generative model
class G(nn.Module):
    def __init__(self, input_size=10):
        super(G, self).__init__()

        self.input_size = input_size

        self.fc_in = nn.Linear(self.input_size, 7 * 7 )
        
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(1, 2, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(2, 1, kernel_size=5, stride=2, padding=0),
            nn.Tanh()
        )

    def forward(self, x, noise):
        out = self.fc_in(x)
        out = out.view(out.size(0), 1, 7, 7)
        out = self.conv1(out)
        return out


_d = D(1, 10)
_d.cuda()

_g = G(1)
_g.cuda()

# Loss and Optimizer
criterion_d = nn.CrossEntropyLoss()
optimizer_d = torch.optim.Adam(_d.parameters(), lr=learning_rate)

criterion_g = nn.CrossEntropyLoss()
optimizer_g = torch.optim.Adam(_g.parameters(), lr=learning_rate)


# Train the model
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())

        # D Forward + Backward + Optimize
        optimizer_d.zero_grad()
        outputs_d = _d(images)

        loss_d = criterion_d(outputs_d, labels)
        loss_d.backward()
        optimizer_d.step()

        if (i + 1) % 100 == 0:
            print("Epoch [%d/%d], Iter [%d/%d] D Loss: %.8f" % (epoch + 1, epochs,
                                                              i + 1, len(train_dataset) // batch_size, loss_d.data[0]))

        # G Forward + Backward + Optimize
        labels_g = labels.view(-1, 1).cuda()
        images_g = _g(labels_g.float(), [])

        optimizer_g.zero_grad()
        truth = _d(images_g)

        loss_g = criterion_g(truth, labels)
        loss_g.backward()
        optimizer_g.step()

        if (i + 1) % 100 == 0:
            print("Epoch [%d/%d], Iter [%d/%d] G Loss: %.8f" % (epoch + 1, epochs,
                                                              i + 1, len(train_dataset) // batch_size, loss_g.data[0]))

                                                        
# Test D
_d.eval()
correct_d = 0
total_d = 0

for images, labels in test_loader:
    images = Variable(images).cuda()
    outputs = _d(images)

    _, predicted = torch.max(outputs.data, 1)
    total_d += labels.size(0)
    correct_d += (predicted.cpu() == labels).sum()

print("Test accuracy of the model on the 10000 test images: %d %%" % (100 * correct_d/total_d))

# Test G
_g.eval()
correct_g = 0
total_g = 0

labels = Variable(torch.arange(0, 10).view(-1, 1)).cuda()
im_outputs = _g(labels, [])

figure_path = './sample'
pathlib.Path(figure_path).mkdir(parents=True, exist_ok=True)

for i, img in enumerate(im_outputs):
    if img.size(0) == 1:
        a = img[0].data.cpu().numpy()
    else:
        a = img.data.cpu().numpy()
    plt.imshow(a, cmap='gray')
    plt.savefig(os.path.join(figure_path, "%d.png" % i))

# Save the trained model
D_model_path = './model/d'
pathlib.Path(D_model_path).mkdir(parents=True, exist_ok=True)
torch.save(_d.state_dict(), os.path.join(D_model_path, '_d.pkl'))

G_model_path = './model/g'
pathlib.Path(D_model_path).mkdir(parents=True, exist_ok=True)
torch.save(_d.state_dict(), os.path.join(D_model_path, '_g.pkl'))