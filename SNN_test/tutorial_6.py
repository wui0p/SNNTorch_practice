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

spk_rec, mem_rec = forward_pass(net, num_steps, data)


# =====================================
#   forward pass
# =====================================

# loss function
loss_fn = SF.ce_rate_loss()
loss_val = loss_fn(spk_rec, targets)
print(f"The loss from the untrained network is {loss_val.item():.2f}")

# accuracy
def acc_fn(dataloader, net, num_steps):
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()
        dataloader = iter(dataloader)

        for data, targets in dataloader:
            data = data.to(device)
            targets = targets.to(device)
            spk_rec, _ = forward_pass(net, num_steps, data)

            acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)

    return acc/total

test_acc = acc_fn(test_dataloader, net, num_steps)
print(f"The accuracy from the untrained network is {test_acc*100:.2f}% for the test datas")

# optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999))

# timer
def print_total_time(start: float,
                     end: float,
                     device: torch.device = None):
  total_time = end - start
  print(f"Train time on {device}: {total_time:.3f} seconds")
  print(f"Train time on {device}: {total_time/60:.1f} minutes")
  return total_time


# =====================================
#   training loop
# =====================================
epochs = 2
loss_hist = []
test_acc_hist = []

time_start = Timer()
for epoch in tqdm(range(epochs)):
    for data, targets in iter(train_dataloader):
        data = data.to(device)
        targets = targets.to(device)

        net.train()
        spk_rec, _ = forward_pass(net, num_steps, data)

        loss_val = loss_fn(spk_rec, targets)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        loss_hist.append(loss_val.item())

    test_acc = acc_fn(test_dataloader, net, num_steps)
    test_acc_hist.append(test_acc)
    print(f"Epoch {epoch}: test loss is {test_acc*100}% \n")
    

time_end = Timer()
total_time = print_total_time(start = time_start,
                              end = time_end,
                              device = device)
# test loss is 95.73%
# Train time on cpu: 1134.474 seconds
# Train time on cpu: 18.9 minutes