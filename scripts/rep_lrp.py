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
sys.path.append(path_to_run_dir + "clr_run15/args_NL_200_NC_20_temp_0.1_bsize_1000_lr_0.001_SB_0.05_N_1")

import extargs as args

# initialise logfile

logfile = open(args.logfile, "a")
print("logfile continued by lrp_playground.py", file=logfile, flush=True)

# set up results directory

expt_tag = args.expt
expt_dir = path_to_rep_dir + "experiments/" + expt_tag + "/"

# import the trained model
fcn = fully_connected_linear_network(200, 1, args.learning_rate)
fcn.load_state_dict(torch.load(expt_dir + "state_dict.pt"))
fcn.eval()

print("model loaded.", file=logfile, flush=True)

# import the test images with classifications and true labels

out_dat = np.load(expt_dir + "out_dat.npy")  # guesses of fcn
reps_te = np.load(expt_dir + "reps_te.npy")  # test input
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

    c = rho(layer.weight, pos) * s

    R = a * c
    return R


def relprop_3(a, layer, R, pos=True):
    # for k in range(len(a)): # for every output neuron -> here only once

    z = epsilon + torch.dot(a, (layer.weight).squeeze()) + (layer.bias).squeeze()

    s = R / z

    c = (layer.weight) * s

    R = a * c
    return R


sigmoid = nn.Sigmoid()

for j in range(1):
    i = 100 + j
    x = reps_te[i, :]
    x = torch.Tensor(x)  # .view(-1, 1)
    # print(fcn(x), sigmoid(fcn(x)))
    # print(sigmoid(fcn(x)).unsqueeze(1), fcn.layer.weight)
    # print(x,fcn.layer.weight)

    # Rp = relprop_2(x, fcn.layer, fcn(x), pos=True)
    # Rn = relprop_2(x, fcn.layer, fcn(x), pos=False)

    R = torch.mul(x, fcn.layer.weight)

    d = np.arange(200)
    #x = x[175:]
    # r = R[175:]
    #r = R[175:]
    r = np.array([s.detach().numpy() for s in R]).squeeze()

    # , vmin=-torch.max(torch.Tensor([r,-r])), vmax=torch.max(torch.Tensor([r,-r]))
    plt.scatter(d, x, c=r, vmin=-np.max([r, -r]), vmax=np.max([r, -r]), cmap="bwr")
    cbar = plt.colorbar(shrink=0.9)
    cbar.set_label("relevance wrt. top classification")
    plt.gca().xaxis.grid(True)
    plt.xlabel("representation dimension")
    plt.ylabel("value in representation")
    plt.suptitle(f"feature relevance of jet representation for classification")
    plt.title(f"true:{telab[i]} fcn:{out_dat[i, 0]:.2f}")
    plt.tight_layout()
    plt.savefig(expt_dir + f"relevance_{j}.pdf")
    plt.show()

    # img = np.reshape(x, (40, 40))
    # fig, ax = plt.subplots(1, 2, figsize=(7, 4))
    # ax1, ax2 = ax
#
# n1 = ax1.imshow(np.abs(img), cmap='Reds')
# n2 = ax2.imshow(n_p-n_n, cmap='bwr', vmin=-np.max([np.abs(n_p), np.abs(n_n)]), vmax=np.max([np.abs(n_p), np.abs(n_n)]))
# ax1.title.set_text("original jet image")
# ax2.title.set_text("'relevance'")
# plt.suptitle("true label: " + str(telab[i]) + "      fcn label: " + str(round(out_dat[i, 0], 3)))
# plt.colorbar(n1, ax=ax1, shrink=0.7)  # #location='left', anchor=(0, 0.3), shrink=0.7)
# plt.colorbar(n2, ax=ax2, shrink=0.7)  # #location='right', anchor=(0, 0.3), shrink=0.7)

# sp = round(n_p.sum(), 2)
# print(sp)
# plt.figtext(0.7, 0.01, f"relevance balance {n_p.sum():.2f} - {n_n.sum():.2f} = {np.sum(n_p-n_n):.2f}", ha="center", fontsize=10)
# fig.tight_layout()
# fig.savefig(expt_dir + f"relevance_{j}.pdf")
# fig.show()

print("relevance plots saved.", file=logfile, flush=True)

w = fcn.layer.weight.detach().numpy().squeeze()

b = reps_te[telab == 0].mean(axis=0)
s = reps_te[telab == 1].mean(axis=0)

fig, ax = plt.subplots(1, 3, figsize=(10, 3.5))
plt.subplots_adjust(bottom=0.2)
ax1, ax2, ax3 = ax

#plt.scatter(d, x, c=r, vmin=-np.max([r, -r]), vmax=np.max([r, -r]), cmap="bwr")
n1 = ax1.scatter(np.arange(200), b, c="g")
n2 = ax2.scatter(np.arange(200), s, c="g")
n3 = ax3.scatter(np.arange(200), w, c="g")

# plt.suptitle("true: " + str(telab[i]) + "      fcn: " + str(round(out_dat[i, 0], 3)))
ax1.title.set_text("qcd")
ax2.title.set_text("top")
ax3.title.set_text("weights")
plt.figtext(0.8, 0.01, "performance of weights \nauc: 0.922 imtafe: 18.2", ha="center",
            fontsize=10)  # , bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
# plt.colorbar(n1, ax=ax1, shrink=0.6)  # #location='left', anchor=(0, 0.3), shrink=0.7)
# plt.colorbar(n2, ax=ax2, shrink=0.6)  # #location='right', anchor=(0, 0.3), shrink=0.7)
# plt.colorbar(n3, ax=ax3, shrink=0.6)  # #location='right', anchor=(0, 0.3), shrink=0.7)
#fig.tight_layout()
fig.savefig(expt_dir + f"weights.pdf")
fig.show()

relevance_b = w*b
relevance_s = w*s

plt.figure(figsize=(14, 4))
plt.scatter(np.arange(200), b, s=15, c=relevance_b, marker="s", label="mean value of qcd", vmin=-np.max([relevance_b, -relevance_b]), vmax=np.max([relevance_b, -relevance_b]), cmap="bwr")
plt.scatter(np.arange(200), s, s=15, c=relevance_s, marker="^", label="mean value of top",     vmin=-np.max([relevance_s, -relevance_s]), vmax=np.max([relevance_s, -relevance_s]), cmap="bwr")
plt.gca().xaxis.grid(True)
cbar = plt.colorbar(shrink=0.9)
cbar.set_label("relevance in linear classification")
leg = plt.legend()
for marker in leg.legendHandles:
    marker.set_color('grey')
plt.tight_layout()
plt.show()


print("input/model plot saved.", file=logfile, flush=True)

