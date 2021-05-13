#!/bin/env python
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
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


def rho(weight, pos):
    if pos:
        return torch.maximum(weight, torch.zeros_like(weight))  # only use positive contribution
    else:
        return torch.minimum(weight, torch.zeros_like(weight))  # only use negative contribution


def relprop(a, layer, R):
    """a : activation in lower layer
       b : weights to higher layer
       R : relevance of higher layer
       returns: relevance of lower layer"""

    z = epsilon + fcn.forward(a)
    s = R / (z + 1e-9)
    (z * s.data).sum().backward()
    c = a.grad
    # print(a, c)
    R = a * c

    return R


def relprop_2(a, layer, R, pos=True):
    # for k in range(len(a)): # for every output neuron -> here only once

    z = epsilon + torch.dot(a, rho(layer.weight, pos).squeeze()) + rho(layer.bias, pos).squeeze()

    s = R / z
    c = rho(layer.weight, pos)  # * s

    R = a * c
    return R


sigmoid = nn.Sigmoid()

tmul = torch.mul(torch.Tensor(tedat), fcn.layer.weight)
r_border = max(abs(torch.max(tmul).detach()), abs(torch.min(tmul).detach()))  # np.max([r, -r])
v_border = np.max(np.concatenate((tedat, -tedat))) + 0.02

indices_b = np.where(telab == 0)[0]
indices_s = np.where(telab == 1)[0]
print(telab.shape, out_dat.shape)
wrong_indeces = np.where(telab != np.round(out_dat.squeeze()))[0]
print(wrong_indeces.shape)

for ind_range, name in ((indices_b[:5], "qcd"), (indices_s[:5], "top"), (wrong_indeces[:5], "wrong")):
    for i in ind_range:
        i = int(i)
        x = tedat[i, :]
        x = torch.Tensor(x)  # .view(-1, 1)
        # print(fcn(x), sigmoid(fcn(x)))
        # print(sigmoid(fcn(x)).unsqueeze(1), fcn.layer.weight)
        # print(x,fcn.layer.weight)

        Rp = relprop_2(x, fcn.layer, fcn(x), pos=True)

        Rn = relprop_2(x, fcn.layer, fcn(x), pos=False)

        n_p = np.array([r.detach().numpy() for r in Rp]).reshape((40, 40))
        n_n = np.array([r.detach().numpy() for r in Rn]).reshape((40, 40))

        r = torch.mul(x, fcn.layer.weight).detach()[0].reshape((40, 40))

        # display input relevance as overlay on pictures

        colormap = cm.get_cmap("bwr", 100)
        oneside = np.linspace(0, 1, 256) ** 0.5

        csamplespace = np.concatenate((-oneside[::-1], oneside)) / max(oneside) / 2 + 0.5
        # plt.plot(csamplespace)
        # plt.show()

        colors = colormap(csamplespace)
        colormap = ListedColormap(colors)

        greymap = cm.get_cmap("Greys", 100)
        oneside = np.linspace(0, 1, 256) ** 0.5  # np.logspace(0, 1, 256, base=5) - 1

        # csamplespace = np.concatenate((-oneside[::-1], oneside)) / 2 + 0.5

        greys = greymap(oneside)
        greymap = ListedColormap(greys)

        img = np.reshape(x, (40, 40))
        fig, ax = plt.subplots(1, 2, figsize=(7, 4))
        ax1, ax2 = ax

        n1 = ax1.imshow(np.abs(img), vmin=0, vmax=v_border, cmap=greymap)
        n2 = ax2.imshow(r, cmap=colormap, vmin=-r_border, vmax=r_border)
        ax1.title.set_text("original jet image")
        ax2.title.set_text("'relevance'")
        plt.suptitle("true label: " + str(telab[i]) + "      $\sigma(z)=$ " + str(round(out_dat[i, 0], 3)))
        plt.colorbar(n1, ax=ax1, shrink=0.7)  # #location='left', anchor=(0, 0.3), shrink=0.7)
        plt.colorbar(n2, ax=ax2, shrink=0.7)  # #location='right', anchor=(0, 0.3), shrink=0.7)

        plt.figtext(0.3, 0.04, "$z = \sum_j R_j^+ + \sum_j R_j^- + w_0 = $", ha="center", fontsize=10)
        plt.figtext(0.7, 0.04,
                    rf"${n_p.sum():.2f} + {n_n.sum():.2f} + {float(fcn.layer.bias[0].detach().numpy()):.2f}= {np.sum(n_p + n_n) + float(fcn.layer.bias[0].detach().numpy()):.2f} \quad \rightarrow \sigma(z)={sigmoid(np.sum(n_p + n_n) + fcn.layer.bias[0].detach()):.2f}$",
                    ha="center", fontsize=10)
        fig.tight_layout()
        fig.savefig(expt_dir + f"relevance_{name}_{i}.pdf")
        fig.show()
print(float(fcn.layer.bias[0].detach().numpy()))
print("relevance plots saved.", file=logfile, flush=True)

w = fcn.layer.weight.detach().numpy().reshape((40, 40))

imgs = tedat.reshape(6000, 40, 40)
b = imgs[telab == 0].mean(axis=0)
s = imgs[telab == 1].mean(axis=0)

fig, ax = plt.subplots(1, 3, figsize=(10, 3.5))
ax1, ax2, ax3 = ax

n1 = ax1.imshow(b, cmap=greymap)
n2 = ax2.imshow(s, cmap=greymap)
n3 = ax3.imshow(w, cmap=colormap, vmin=-np.max([np.abs(w)]), vmax=np.max([np.abs(w)]))
# plt.suptitle("true: " + str(telab[i]) + "      fcn: " + str(round(out_dat[i, 0], 3)))
ax1.title.set_text("background")
ax2.title.set_text("signal")
ax3.title.set_text("weights")
plt.figtext(0.8, 0.01, "performance of weights \nauc: 0.927, inverse false positive rate: 22.7", ha="center",
            fontsize=10)  # , bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
plt.colorbar(n1, ax=ax1, shrink=0.6)  # #location='left', anchor=(0, 0.3), shrink=0.7)
plt.colorbar(n2, ax=ax2, shrink=0.6)  # #location='right', anchor=(0, 0.3), shrink=0.7)
plt.colorbar(n3, ax=ax3, shrink=0.6)  # #location='right', anchor=(0, 0.3), shrink=0.7)
fig.tight_layout()
fig.savefig(expt_dir + f"weights.pdf")
fig.show()

print("input/model plot saved.", file=logfile, flush=True)
