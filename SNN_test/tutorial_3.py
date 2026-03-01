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
from plotting_functions import plot_cur_mem_spk, plot_snn_spikes


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
#   singular leaky neuron
# =====================================
num_steps = 200
lif1 = snn.Leaky(beta=0.8)

cur_in = torch.cat((torch.zeros(10), torch.ones(190)*0.21), 0)
mem = torch.zeros(1)
mem_rec = []
spk_rec = []

for step in range(num_steps):
    spk, mem = lif1(cur_in[step], mem)
    mem_rec.append(mem)
    spk_rec.append(spk)
mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)

plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1,
                 title="snn.Leaky Neuron Model")

# =====================================
#   layers of neuron
# =====================================
num_inputs = 784
num_hidden = 1000
num_outputs = 10
beta = 0.99

# initialize layers
fc1 = nn.Linear(num_inputs, num_hidden)
lif1 = snn.Leaky(beta = beta)
fc2 = nn.Linear(num_hidden, num_outputs)
lif2 = snn.Leaky(beta = beta)

# initialize the membrane U for both layers
mem1 = lif1.init_leaky()
mem2 = lif2.init_leaky()

mem2_rec = []
spk1_rec = []
spk2_rec = []

spk_in = spikegen.rate_conv(torch.rand((200, 784))).unsqueeze(1)

for step in range(num_steps):
    cur1 = fc1(spk_in[step]) # post-synaptic current <-- spk_in * weight
    spk1, mem1 = lif1(cur1, mem1)
    cur2 = fc2(spk1)
    spk2, mem2 = lif2(cur2, mem2)

    mem2_rec.append(mem2)
    spk1_rec.append(spk1)
    spk2_rec.append(spk2)
mem2_rec = torch.stack(mem2_rec)
spk1_rec = torch.stack(spk1_rec)
spk2_rec = torch.stack(spk2_rec)

plot_snn_spikes(spk_in, spk1_rec, spk2_rec, num_steps, "Fully Connected Spiking Neural Network")

splt.traces(mem2_rec.squeeze(1), spk=spk2_rec.squeeze(1))
fig = plt.gcf()
fig.set_size_inches(8, 6)
plt.show()

