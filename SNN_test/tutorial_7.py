import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import functional as SF
from snntorch import spikegen, utils, surrogate

import os
import sys
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F

import torchvision
import tonic
import tonic.transforms as transforms
from tonic import DiskCachedDataset, MemoryCachedDataset

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

dataset = tonic.datasets.NMNIST(save_to='./data', train=True)
events, target = dataset[0]
print(events)

sensor_size = tonic.datasets.NMNIST.sensor_size
frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                      transforms.ToFrame(sensor_size=sensor_size,
                                                         time_window=1000)
                                     ])
trainset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True)
testset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=False)

transform = tonic.transforms.Compose([torch.from_numpy,
                                      torchvision.transforms.RandomRotation([-10,10])])

cached_trainset = DiskCachedDataset(trainset, transform=transform, cache_path='/cache/nmnist/train')
cached_testset = DiskCachedDataset(testset, cache_path='/cache/nmnist/test')

trainloader = DataLoader(cached_trainset, batch_size=BATCH, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True)
testloader = DataLoader(cached_testset, batch_size=BATCH, collate_fn=tonic.collation.PadTensors(batch_first=False))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")


# =====================================
#   define network
# =====================================
spike_grad = surrogate.atan()
beta = 0.5

net = nn.Sequential(
    nn.Conv2d(2, 12, 5),
    nn.MaxPool2d(2),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Conv2d(12, 32, 5),
    nn.MaxPool2d(2),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Flatten(),
    nn.Linear(32*5*5, 10),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
).to(device)


def forward_pass(net, data):
    spk_rec = []
    utils.reset(net)

    for step in range(data.size(0)):
        spk_out, mem_out = net(data[step])
        spk_rec.append(spk_out)

    return torch.stack(spk_rec)

optimizer = torch.optim.Adam(net.parameters(), lr=2e-2, betas=(0.9, 0.999))
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)


# =====================================
#   training loop
# =====================================
epochs = 1
loss_hist = []
acc_hist = []

for epoch in range(epochs):
    for i, (data, targets) in enumerate(iter(trainloader)):
        data= data.to(device)
        targets = targets.to(device)

        net.train()
        spk_rec = forward_pass(net, data)
        loss_val = loss_fn(spk_rec, targets)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        loss_hist.append(loss_val.item())

        print(f"Epoch {epoch}, iteration {i} \n Train Loss: {loss_val.item():.2f}")
        acc = SF.accuracy_rate(spk_rec, targets)
        acc_hist.append(acc)
        print(f"Accuracy: {acc*100:.2f}%\n")

        if i == 50:
            break

# plot it out
fig = plt.figure(facecolor="w")
plt.plot(acc_hist)
plt.title("Train Set Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.show()
