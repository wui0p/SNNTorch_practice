import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import functional as SF
from snntorch import spikegen, utils, surrogate, backprop

import os
import sys
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from timeit import default_timer as Timer

import requests
import itertools

from pathlib import Path


# =====================================
#   DATALOADER
# =====================================
BATCH = 32
train_data = datasets.MNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor(),
    target_transform = None
)
test_data = datasets.MNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor()
)
train_dataloader = DataLoader(dataset = train_data,
                              batch_size = BATCH,
                              shuffle = True)
test_dataloader = DataLoader(dataset = test_data,
                             batch_size = BATCH,
                             shuffle = False)
num_classes = len(train_data.classes)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")


# =====================================
#   define network
# =====================================
spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5
num_steps = 50

net = nn.Sequential(
    nn.Conv2d(1, 12, 5),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.MaxPool2d(2),
    nn.Conv2d(12, 64, 5),
    nn.MaxPool2d(2),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Flatten(),
    nn.Linear(64*4*4, 10),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
).to(device)


# =====================================
#   forward pass
# =====================================
data, targets = next(iter(train_dataloader))
data = data.to(device)
targets = targets.to(device)

def forward_pass(net, num_steps, data):
    mem_rec = []
    spk_rec = []
    utils.reset(net) # reset all hidden stats in all LIF neurons in net module

    for step in range(num_steps):
        spk_out, mem_out = net(data)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)

    return torch.stack(spk_rec), torch.stack(mem_rec)
