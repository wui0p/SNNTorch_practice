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
from plotting_functions import plot_mem, plot_step_current_response, plot_current_pulse_response, plot_cur_mem_spk, plot_spk_mem_spk, plot_reset_comparison


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
#   LAPICQUE
# =====================================
time_step = 1e-3
R = 5
C = 1e-3
num_steps = 100

lif1 = snn.Lapicque(R=R, C=C, time_step=time_step)

mem = torch.ones(1) * 0.9 # initial U=0.9
cur_in = torch.zeros(num_steps, 1)
spk_out = torch.zeros(1)
mem_rec = [mem]

for step in range(num_steps):
    spk_out, mem = lif1(cur_in[step], mem)
    mem_rec.append(mem)
mem_rec = torch.stack(mem_rec) # convert the list of tensors into one tensor
plot_mem(mem_rec, "Lapicque's module without stimulus")

# now with step input
cur_in = torch.cat((torch.zeros(10, 1), torch.ones(190, 1)*0.1), 0)
mem = torch.zeros(1)
spk_out = torch.zeros(1)
mem_rec = [mem]

for step in range(num_steps):
    spk_out, mem = lif1(cur_in[step], mem)
    mem_rec.append(mem)
mem_rec = torch.stack(mem_rec)
plot_step_current_response(cur_in, mem_rec, 10)

# with pulse input
cur_in = torch.cat((torch.zeros(10, 1), torch.ones(20, 1)*(0.1), torch.zeros(170, 1)), 0)
mem = torch.zeros(1)
spk_out = torch.zeros(1)
mem_rec = [mem]

for step in range(num_steps):
    spk_out, mem = lif1(cur_in[step], mem)
    mem_rec.append(mem)
mem_rec = torch.stack(mem_rec)
plot_current_pulse_response(cur_in, mem_rec, "Lapicque's Neuron Model With Input Pulse",
                            vline1=10, vline2=30)


# using RC to recreate reseting in neuron
num_steps = 200
lif2 = snn.Lapicque(R=5.1, C=5e-3, time_step=1e-3)

cur_in = torch.cat((torch.zeros(10, 1), torch.ones(190, 1)*0.2), 0)
mem = torch.zeros(1)
spk_out = torch.zeros(1)
mem_rec = [mem]
spk_rec = [spk_out]

for step in range(num_steps):
  spk_out, mem = lif2(cur_in[step], mem)
  mem_rec.append(mem)
  spk_rec.append(spk_out)
mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)

plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1, vline=109, ylim_max2=1.3,
                 title="Lapicque Neuron Model With Step Input")
 
# setting threshold for it reset
lif3 = snn.Lapicque(R=5.1, C=5e-3, time_step=1e-3, threshold=0.5)

spk_in = spikegen.rate_conv(torch.ones((num_steps, 1)) * 0.40)
mem = torch.ones(1) * 0.5
spk_out = torch.zeros(1)
mem_rec = [mem]
spk_rec = [spk_out]

for step in range(num_steps):
  spk_out, mem = lif3(spk_in[step], mem)
  spk_rec.append(spk_out)
  mem_rec.append(mem)
mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)

plot_spk_mem_spk(spk_in, mem_rec, spk_rec, "Lapicque's Neuron Model With Input Spikes")

# reseting directly to zero and not subtracting from mem
lif4 = snn.Lapicque(R=5.1, C=5e-3, time_step=1e-3, threshold=0.5, reset_mechanism="zero") # reset by setting U to zero

spk_in = spikegen.rate_conv(torch.ones((num_steps, 1)) * 0.40)
mem = torch.ones(1) * 0.5
spk_out = torch.zeros(1)
mem_rec0 = [mem]
spk_rec0 = [spk_out]

for step in range(num_steps):
   spk_out, mem = lif4(spk_in[step], mem)
   spk_rec0.append(spk_out)
   mem_rec0.append(mem_rec0)
spk_rec0 = torch.stack(spk_rec0)
mem_rec0 = torch.stack(mem_rec0)

plot_reset_comparison(spk_in, mem_rec, spk_rec, mem_rec0, spk_rec0)


