import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.autograd import Variable

import pathlib
import os

import matplotlib.pyplot as plt
import numpy as np

def generate_images(_g, m, prefix="", suffix="", noise=None):
    labels = Variable(torch.arange(0, 10).view(-1)).data.numpy()
    labels = (np.arange(11) == labels[:,None]).astype(np.float)
    labels = torch.from_numpy(labels)
    labels = Variable(labels.cuda())

    if noise is None:
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
        plt.savefig(os.path.join(figure_path, "%s-%d-%s.png" % (prefix, i, suffix)))

def generate_batch_images(_g, m, batch_size, start=0, end=9, prefix="", suffix="", figure_path="./samples", noise="normal"):
    if start >= end:
        raise ArithmeticError("start is higher than end [%d > %d]" % (start, end))
    
    pathlib.Path(figure_path).mkdir(parents=True, exist_ok=True)    

    for n in range(start, end + 1):
        label = np.full((batch_size, 1), n)
        label_one_hot = (np.arange(11) == label[:,None]).astype(np.float)
        label_one_hot = torch.from_numpy(label_one_hot)
        label_one_hot = Variable(label_one_hot.cuda())

        if noise == "normal":
            noise = Variable(torch.cuda.FloatTensor(batch_size, 1, 29, 29).normal_())
        elif noise == "uniform":
            noise = Variable(torch.cuda.FloatTensor(batch_size, 1, 29, 29).uniform_())
        elif noise == "cauchy":
            noise = Variable(torch.cuda.FloatTensor(batch_size, 1, 29, 29).cauchy_())
        elif noise == "log_normal":
            noise = Variable(torch.cuda.FloatTensor(batch_size, 1, 29, 29).log_normal_())
        elif noise == "geometric":
            noise = Variable(torch.cuda.FloatTensor(batch_size, 1, 29, 29).geometric_())
        elif noise == "exponential":
            noise = Variable(torch.cuda.FloatTensor(batch_size, 1, 29, 29).exponential_())
        elif noise == "random":
            noise = Variable(torch.cuda.FloatTensor(batch_size, 1, 29, 29).random_())
            
        im_outputs = _g(label_one_hot.float(), noise)
        for i, img in enumerate(im_outputs):
            if img.size(0) == 1:
                a = img[0].data.cpu().numpy()
            else:
                a = img.data.cpu().numpy()

            plt.imshow(a, cmap='gray')
            plt.savefig(os.path.join(figure_path, "%s-%d-%d-%s.png" % (prefix, n, i, suffix)))
