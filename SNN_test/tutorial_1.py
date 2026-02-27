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
#   spiking and how it works
# =====================================
subset = 10
mnist_train = utils.data_subset(train_data, subset)

raw_vector = torch.ones(10) * 0.5
rate_coded_vector = torch.bernoulli(raw_vector)
print(rate_coded_vector)

raw_vector = torch.ones(100) * 0.5
rate_coded_vector = torch.bernoulli(raw_vector)
print(f"The output is spiking {(rate_coded_vector.sum()/len(rate_coded_vector))*100:.2f}% of the time")


# =====================================
#   generate spike
# =====================================
# iterate through batches
data = iter(train_dataloader)
data_it, targets_it = next(data)

# spiking data
num_steps = 100
spike_data = spikegen.rate(data_it, num_steps = num_steps, gain=1)
print(spike_data.size())

# raster plot
spike_data_sample = spike_data[:,0,0].reshape((num_steps, -1))
fig = plt.figure(facecolor='w', figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spike_data_sample, ax, s=1.5, c='black')

plt.title("Rate Encoding Input Layer")
plt.xlabel("Time Step")
plt.ylabel("Neuron Number")
plt.show()

# spiking data
spike_data = spikegen.rate(data_it, num_steps = num_steps, gain=0.25) # reduce the intensity by 4 times
print(spike_data.size())

# raster plot
spike_data_sample = spike_data[:,0,0].reshape((num_steps, -1))
fig = plt.figure(facecolor='w', figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spike_data_sample, ax, s=1.5, c='black')

plt.title("Rate Encoding Input Layer")
plt.xlabel("Time Step")
plt.ylabel("Neuron Number")
plt.show()


# =====================================
#   generate latency
# =====================================
spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01)

# raster plot
fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spike_data[:, 0].view(num_steps, -1), ax, s=25, c="black")

plt.title("Latency Encoding Input Layer")
plt.xlabel("Time Step")
plt.ylabel("Neuron Number")
plt.show()

# distribute the diring rate by using normalize & linear
# remove the "firing wall" at the end, due to background, useless informations
spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01, clip=True, normalize=True, linear=True)

# raster plot
fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spike_data[:, 0].view(num_steps, -1), ax, s=25, c="black")

plt.title("Latency Encoding Input Layer")
plt.xlabel("Time Step")
plt.ylabel("Neuron Number")
plt.show()

# =====================================
#   generate delta modulation
# =====================================
data = torch.Tensor([0, 1, 0, 2, 8, -20, 20, -5, 0, 1, 0])

spike_data = spikegen.delta(data, threshold=4)

# raster plot
fig = plt.figure(facecolor="w", figsize=(8, 1))
ax = fig.add_subplot(111)
splt.raster(spike_data, ax, c="black")

plt.title("Input Neuron")
plt.xlabel("Time step")
plt.yticks([])
plt.xlim(0, len(data))
plt.show()

# makesure that negative swing is considered
spike_data = spikegen.delta(data, threshold=4, off_spike=True)

# raster plot
fig = plt.figure(facecolor="w", figsize=(8, 1))
ax = fig.add_subplot(111)
splt.raster(spike_data, ax, c="black")

plt.title("Input Neuron")
plt.xlabel("Time step")
plt.yticks([])
plt.xlim(0, len(data))
plt.show()

print(spike_data)