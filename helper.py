import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.autograd import Variable

import pathlib
import os

import matplotlib.pyplot as plt
import numpy as np

def generate_images(_g, m, prefix="", suffix=""):
    labels = Variable(torch.arange(0, 10).view(-1)).data.numpy()
    labels = (np.arange(11) == labels[:,None]).astype(np.float)
    labels = torch.from_numpy(labels)
    labels = Variable(labels.cuda())

    noise = Variable(m.sample_n(10 * 29 * 29).view(10, 1, 29, 29).cuda())

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