import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.autograd import Variable

import pathlib
import os
import sys 
import time

import matplotlib.pyplot as plt
import numpy as np

from helper import generate_images, generate_batch_images
import model


# Discrimination model
class D(nn.Module):
    def __init__(self, input_channels=1, output_size=10):
        super(D, self).__init__()

        self.input_channels = input_channels
        self.output_size = output_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_out = nn.Sequential(
            nn.Linear(7 * 7 * 64, self.output_size)
        )

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
            nn.Tanh()
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
            nn.Tanh()
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
            nn.Tanh()
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
            nn.Tanh()
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
        out = out.view(out.size(0), 1, 7, 7)

        # Conv through label and noise
        out = self.encoder(out)
        _noise = self.transformer_pre(noise)
        _noise = self.transformer_pre_2(_noise)
        
        # Add noise to output 1
        out = out + _noise
        out = self.transformer(out)
        out_2 = self.transformer_2(out)

        # Add noise to output for the last time
        merge_1 = out_2 + _noise
        out = self.transformer_out(merge_1)
        
        return out

def soft_label(np, low=0.3, high=0.7):
    if np == 0:
        return low
    elif np == 1:
        return high
    else: 
        return -1


if __name__ == "__main__":
    # Parameters
    start_time = "%d" % (int(time.time()))
    model_file_name = "%s-model" % (start_time)

    # Hyperparameters
    epochs = 100
    batch_size = 100
    learning_rate_d = 1e-3
    learning_rate_g = 1e-3

    # MNIST dataset
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(),
    download=True)

    test_dataset = datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor())

    #Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    print("Building discriminative model", end="\r")
    _d = D(1, 11)
    _d.cuda()
    sys.stdout.write("\r\033[K\n")
    print("Discriminative model done")

    print("Building generative model", end="\r")
    #_g = G(11)
    #_g = model.G_label_no_conv(11)
    _g = model.G_rnn(11)
    print(_g.parameters)
    _g.cuda()
    sys.stdout.write("\r\033[K\n")
    print("Generative model done")

    # Loss and Optimizer
    criterion_d = nn.CrossEntropyLoss().cuda()
    optimizer_d = torch.optim.Adam(_d.parameters(), lr=learning_rate_d)

    criterion_g = nn.CrossEntropyLoss().cuda()
    optimizer_g = torch.optim.Adam(_g.parameters(), lr=learning_rate_g)

    m = torch.distributions.Normal(torch.Tensor([-1.0]), torch.Tensor([1.0]))

    soft_label = np.vectorize(soft_label)

    # Logs
    d_loss_log = []
    g_loss_log = []
    log_name = "%s-log" % (start_time)

    with open(log_name, mode='a') as log:
        log.write("Model name: " + _g.__class__.__name__ + "\n")
        log.write("Start time: " + start_time + "\n")
        log.close()

    # Train the model
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            print("Training batch: %d" % (i), end="\r")
            sys.stdout.flush()

            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            labels_onehot = labels.data.cpu().numpy()
            labels_onehot = (np.arange(11) == labels_onehot[:,None]).astype(np.float)
            #labels_onehot = soft_label(labels_onehot, 0.1, 0.9)
            labels_onehot = torch.from_numpy(labels_onehot)
            labels_g = labels_onehot.cuda()
            labels_g = Variable(labels_g)
            #noise = Variable(m.sample_n(batch_size * 28 * 28).view(batch_size, 1, 28, 28).cuda())
            noise = Variable(torch.cuda.FloatTensor(batch_size, 1, 29, 29).normal_())
            # D Forward + Backward + Optimize
            optimizer_d.zero_grad()
            outputs_d = _d(images)

            # Create fake labels
            fake_labels = np.zeros(batch_size) + 10
            fake_labels_d = Variable(torch.from_numpy(fake_labels).long().cuda())

            # Generate fake images and classify it
            fake_images = _g(labels_g.float(), noise)
            fake_outputs = _d(fake_images)
            
            real_loss = criterion_d(outputs_d, labels)
            fake_loss = criterion_d(fake_outputs, fake_labels_d)

            loss_d = real_loss + fake_loss
            #loss_d = real_loss
            
            #if epoch < 2 or epoch % 3 == 0:
            loss_d.backward()
            optimizer_d.step()

            labels_fake = np.random.randint(0, 10, batch_size)
            labels_fake_onehot = (np.arange(11) == labels_fake[:,None]).astype(np.float)
            #labels_fake_onehot = soft_label(labels_fake_onehot, 0.1, 0.9)
            labels_fake_onehot = torch.from_numpy(labels_fake_onehot).cuda()
            labels_fake_onehot = Variable(labels_fake_onehot)

            labels_fake = torch.from_numpy(labels_fake)
            labels_fake = Variable(labels_fake.cuda())

            images_g = _g(labels_fake_onehot.float(), noise)
            optimizer_g.zero_grad()
            truth = _d(images_g)

            loss_g = criterion_g(truth, labels_fake)
            loss_g.backward()
            optimizer_g.step()

            d_loss_log.append(loss_d.data[0])
            g_loss_log.append(loss_g.data[0])

            if (i + 1) % 100 == 0:
                print("Epoch [%d/%d], Iter [%d/%d] D Loss:%.10f, G Loss: %.10f" % (epoch + 1, epochs,
                                                                i + 1, len(train_dataset) // batch_size, loss_d.data[0], loss_g.data[0]))
                with open(log_name, mode='a') as log:
                    log.write("d," + str(d_loss_log) + "\n")
                    log.write("g," + str(g_loss_log) + "\n")
                    log.close()
                    d_loss_log = []
                    g_loss_log = []

                
        print("Generating images: ", end="\r")
        generate_batch_images(_g, m, 5, start=0, end=9, prefix="training-epoch-%d" % (epoch + 1), figure_path=start_time+"-sample")
        sys.stdout.flush()
        print("Generated images for epoch %d" % (epoch + 1))
                                                            
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

    labels = Variable(torch.arange(0, 10).view(-1)).data.numpy()
    labels = (np.arange(11) == labels[:,None]).astype(np.float)
    labels = torch.from_numpy(labels)
    labels = Variable(labels.cuda())

    noise = Variable(torch.cuda.FloatTensor(10, 1, 29, 29).normal_())

    im_outputs = _g(labels.float(), noise)

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
    D_model_path = os.path.join(model_file_name, 'd')
    pathlib.Path(D_model_path).mkdir(parents=True, exist_ok=True)
    torch.save(_d.state_dict(), os.path.join(D_model_path, '_d.pkl'))

    G_model_path = os.path.join(model_file_name, 'g')
    pathlib.Path(G_model_path).mkdir(parents=True, exist_ok=True)
    torch.save(_g.state_dict(), os.path.join(G_model_path, '_g.pkl'))

    generate_batch_images(_g, m, 10, start=0, end=9, prefix="end_training", figure_path=start_time+"sample")
