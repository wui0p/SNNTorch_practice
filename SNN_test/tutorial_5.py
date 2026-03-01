import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen, utils

import os
import sys
import torch
from torch import nn
import numpy as np

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader

import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from timeit import default_timer as Timer

import requests

from pathlib import Path
from helper_functions import accuracy_fn


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
num_inputs = 28*28
num_hidden = 1000
num_outputs = 10

num_steps = 25
beta = 0.95

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
    
net = Net().to(device)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))


# =====================================
#   training loop
# =====================================
def print_loss_accuracy(data, targets, train=False):
    with torch.no_grad():
        output, _ = net(data.view(BATCH, -1))
        _, idx = output.sum(dim=0).max(1)
        acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train Set Loss: {loss_hist[-1]:.2f} | Train Set Accuracy: {acc*100:.2f}%")
    else:
        print(f"Test Set Loss: {test_loss_hist[-1]:.2f} | Test Set Accuracy: {acc*100:.2f}%")

def train_printer():
    print(f"Epoch {epoch}")
    print_loss_accuracy(data, targets, train=True)
    print_loss_accuracy(test_data, test_targets, train=False)
    print("\n")

def print_total_time(start: float,
                     end: float,
                     device: torch.device = None):
  total_time = end - start
  print(f"Train time on {device}: {total_time:.3f} seconds")
  print(f"Train time on {device}: {total_time/60:.1f} minutes")
  return total_time


epochs = 5
loss_hist = []
test_loss_hist = []

data, targets = next(iter(train_dataloader))
data = data.to(device)
targets = targets.to(device)

time_start = Timer()
for epoch in tqdm(range(epochs)):
    train_batch = iter(train_dataloader)

    for data, targets in train_batch:
        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        net.train()
        spk_rec, mem_rec = net(data.view(BATCH, -1))

        # losses summed together
        loss_val = torch.zeros((1), dtype=torch.float, device = device)
        for step in range(num_steps):
            loss_val += loss(mem_rec[step], targets)

        # gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # store loss history
        loss_hist.append(loss_val.item())

        with torch.inference_mode():
            net.eval()
            test_data, test_targets = next(iter(test_dataloader))
            test_data = test_data.to(device)
            test_targets = test_targets.to(device)

            test_spk, test_mem = net(test_data.view(BATCH, -1))

            test_loss = torch.zeros((1), dtype=torch.float, device=device)
            for step in range(num_steps):
                test_loss += loss(test_mem[step], test_targets)
            test_loss_hist.append(test_loss.item())

    train_printer()

time_end = Timer()
total_time = print_total_time(start = time_start,
                              end = time_end,
                              device = device)
# Train Set Loss: 1.17 | Train Set Accuracy: 100.00%
# Test Set Loss: 1.05 | Test Set Accuracy: 100.00%
# Train time on cpu: 674.681 seconds
# Train time on cpu: 11.2 minutes
