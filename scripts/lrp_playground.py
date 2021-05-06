#!/bin/env python
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fcn_linear import fully_connected_linear_network
from models.perf_eval import find_nearest, get_perf_stats

path_to_rep_dir = "/home/ho/coding/encoder-analysis/"
path_to_run_dir = path_to_rep_dir + "run_dir/"
sys.path.append(path_to_run_dir + "linear_image_classifier/args_NL_1_bsize_2000_lr_0.01_SB_1.0_N_1")

import extargs as args

# initialise logfile

logfile = open(args.logfile, "a")
print("logfile continued by lrp_playground.py", file=logfile, flush=True)

# set up results directory

expt_tag = args.expt
expt_dir = path_to_rep_dir + "experiments/" + expt_tag + "/"

input_size = int(40 * 40)

# import the trained model
fcn = fully_connected_linear_network(input_size, args.output_size, args.learning_rate)
fcn.load_state_dict(torch.load(expt_dir + "state_dict.pt"))
fcn.eval()

print("model loaded.", file=logfile, flush=True)

# import the test images with classifications and true labels

out_dat = np.load(expt_dir + "out_dat.npy")  # guesses of fcn
tedat = np.load(expt_dir + "tedat.npy")  # test input
telab = np.load(expt_dir + "telab.npy")  # test true labels

# for single test image classification compute relevance of input features

epsilon = 0


def rho(layer):
    return torch.minimum(layer, torch.zeros_like(layer))  # only use positive contribution


def relprop(a, layer, R):
    """a : activation in lower layer
       b : weights to higher layer
       R : relevance of higher layer
       returns: relevance of lower layer"""

    z = epsilon + fcn.forward(a)
    s = R / (z + 1e-9)
    (z * s.data).sum().backward()
    c = a.grad
    #print(a, c)
    R = a * c

    return R

def relprop_2(a, layer, R):
    #for k in range(len(a)): # for every output neuron -> here only once

    z = epsilon + torch.dot(a, rho(layer).squeeze())
    s = R / z
    c = layer * s
    R = a * c
    return R

sigmoid = nn.Sigmoid()

for i in range(10):
    x = tedat[i, :]
    x = torch.Tensor(x)#.view(-1, 1)
    #print(fcn(x), sigmoid(fcn(x)))
    #print(sigmoid(fcn(x)).unsqueeze(1), fcn.layer.weight)
    #print(x,fcn.layer.weight)
    R = relprop_2(x, fcn.layer.weight, sigmoid(fcn(x)))
    n = np.array([ r.detach().numpy() for r in R]).reshape((40, 40))
    img = np.reshape(x, (40, 40))
    fig, ax = plt.subplots(1,2)
    ax1, ax2 = ax
    n1 = ax1.imshow(np.abs(img), cmap='Reds')
    n2 = ax2.imshow(n, cmap='Blues')
    plt.colorbar(n1, ax=ax1, shrink=0.4)# #location='left', anchor=(0, 0.3), shrink=0.7)
    plt.colorbar(n2, ax=ax2, shrink=0.4)# #location='right', anchor=(0, 0.3), shrink=0.7)
    fig.tight_layout()
    fig.show()
# display input relevance as overlay on pictures
