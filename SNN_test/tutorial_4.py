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
from plotting_functions import plot_spk_cur_mem_spk, plot_spk_mem_spk


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
#   synaptic neron model
# =====================================
alpha = 0.9
beta = 0.8
num_steps = 200

lif1 = snn.Synaptic(alpha=alpha, beta=beta)

spk_period = torch.cat((torch.ones(1)*0.2, torch.zeros(9)), 0)
spk_in = spk_period.repeat(20)

syn, mem = lif1.init_synaptic()
spk_out = torch.zeros(1)
syn_rec = []
mem_rec = []
spk_rec = []

for step in range(num_steps):
    spk_out, syn, mem = lif1(spk_in[step], syn, mem)
    spk_rec.append(spk_out)
    syn_rec.append(syn)
    mem_rec.append(mem)
spk_rec = torch.stack(spk_rec)
syn_rec = torch.stack(syn_rec)
mem_rec = torch.stack(mem_rec)

plot_spk_cur_mem_spk(spk_in, syn_rec, mem_rec, spk_rec,
                     "Synaptic Conductance-based Neuron Model With Input Spikes")


# =====================================
#   alpha neuron model
# =====================================
alpha = 0.8
beta = 0.7

lif2 = snn.Alpha(alpha=alpha, beta=beta, threshold=0.5)

spk_in = (torch.cat((torch.zeros(10), torch.ones(1), torch.zeros(89),
        (torch.cat((torch.ones(1), torch.zeros(9)),0).repeat(10))), 0) * 0.85).unsqueeze(1)

syn_exc, syn_inh, mem = lif2.init_alpha()
mem_rec = []
spk_rec = []

for step in range(num_steps):
    spk_out, syn_exc, syn_inh, mem = lif2(spk_in[step], syn_exc, syn_inh, mem)
    mem_rec.append(mem.squeeze(0))
    spk_rec.append(spk_out.squeeze(0))
mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)

plot_spk_mem_spk(spk_in, mem_rec, spk_rec, "Alpha Neuron Model With Input Spikes")