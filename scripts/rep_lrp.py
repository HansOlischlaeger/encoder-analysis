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

colormap = cm.get_cmap("bwr", 100)
oneside = np.linspace(0, 1, 256) ** 0.8

csamplespace = np.concatenate((-oneside[::-1], oneside)) / max(oneside) / 2 + 0.5
# plt.plot(csamplespace)
# plt.show()

colors = colormap(csamplespace)
colormap = ListedColormap(colors)


sigmoid = nn.Sigmoid()

indices_b = np.where(telab == 0)[0][:5]
indices_s = np.where(telab == 1)[0][:5]

tmul = torch.mul(torch.Tensor(reps_te), fcn.layer.weight)
extraord_i_relhigh = torch.where(tmul > 1.4)[0]
extraord_i_rellow = torch.where(tmul < -1.59)[0]
print("reps_te", reps_te.shape)
extraord_i_rephigh = np.where(reps_te > 0.37)[0]
#print("extraord_i_rephigh", extraord_i_rephigh[1], extraord_i_rephigh[0])
extraord_i_replow = np.where(reps_te < -0.35)[0]
extraord_i_fcnhigh = torch.where(fcn(torch.Tensor(reps_te)) > 0.9)[0]

r_border = max(abs(torch.max(tmul).detach()), abs(torch.min(tmul).detach()))  # np.max([r, -r])
v_border = np.max(np.concatenate((reps_te, -reps_te))) + 0.02
print("vb",v_border)
plot_individual_jets = True
if plot_individual_jets:
    for ind_range, name in (
            (indices_b, "qcd"), (indices_s, "top"), (extraord_i_relhigh, "rel_high"), (extraord_i_rellow, "rel_low"),
            (extraord_i_rephigh, "rep_high"), (extraord_i_replow, "rep_low"), (extraord_i_fcnhigh, "fcn_high")):
        for i in ind_range:
            i = int(i)
            x = reps_te[i, :]
            x = torch.Tensor(x)  # .view(-1, 1)

            R = torch.mul(x, fcn.layer.weight)

            d = np.arange(200)
            r = np.array([s.detach().numpy() for s in R]).squeeze()

            plt.scatter(d, x, c=r, vmin=-r_border, vmax=r_border, cmap=colormap)#, cmap="Spectral_r")
            plt.ylim(-v_border, v_border)
            cbar = plt.colorbar(shrink=0.9)
            cbar.set_label("relevance wrt. top classification")
            plt.gca().xaxis.grid(True)
            plt.xlabel("representation dimension")
            plt.ylabel("value in representation")
            plt.suptitle(f"feature relevance of jet representation for classification")
            plt.title(f"{name}    z={float(fcn(x)[0]):.2f}    $\sigma$(z)={float(sigmoid(fcn(x))[0]):.2f}    true:{telab[i]}")
            plt.tight_layout()
            plt.savefig(expt_dir + f"relevance_{name}_{i}.pdf")
            plt.show()

print("relevance plots saved.", file=logfile, flush=True)

w = fcn.layer.weight.detach().numpy().squeeze()

b = np.median(reps_te[telab == 0], axis=0)
s = np.median(reps_te[telab == 1], axis=0)
fb = (b>s) * -b + (s>b) * b
fs = (b>s) * -s + (s>b) * s
# b_std = reps_te[telab == 0].std(axis=0)
# s_std = reps_te[telab == 1].std(axis=0)
#
# print("qcd",reps_te[telab == 0].shape)
#
# print(((b>s) * -reps_te[telab == 0] + (s>b) * reps_te[telab == 0]).shape)
# f_reps_b = (b>s) * -reps_te[telab == 0] + (s>b) * reps_te[telab == 0]
# f_reps_s = (b>s) * -reps_te[telab == 1] + (s>b) * reps_te[telab == 1]
# b75, b25 = np.percentile(f_reps_b, [75 ,50], axis=0)
# s75, s25 = np.percentile(f_reps_s, [50 ,25], axis=0)
# b_iqr = b75 - b25
# s_iqr = s75 - s25


fig, ax = plt.subplots(1, 3, figsize=(10, 3.5))
plt.subplots_adjust(bottom=0.2)
ax1, ax2, ax3 = ax

# plt.scatter(d, x, c=r, vmin=-np.max([r, -r]), vmax=np.max([r, -r]), cmap="bwr")
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
# fig.tight_layout()
fig.savefig(expt_dir + f"weights.pdf")
#fig.show()





relevance_b = w * b
relevance_s = w * s



plt.figure(figsize=(9, 3.5))
r_border = np.max([relevance_b, -relevance_b, relevance_s, -relevance_s])

plt.scatter(np.arange(200), b, s=20, c=relevance_b, marker="s", label="mean value qcd", vmin=-r_border,
            vmax=r_border, cmap=colormap)
plt.scatter(np.arange(200), s, s=20, c=relevance_s, marker="^", label="mean value top", vmin=-r_border,
            vmax=r_border, cmap=colormap)
plt.gca().xaxis.grid(True)
plt.xlabel("dimension $ \quad$")
cbar = plt.colorbar(shrink=0.9)
cbar.set_label("relevance in linear classification")
leg = plt.legend()
for marker in leg.legendHandles:
    marker.set_color('grey')
plt.tight_layout()
plt.savefig(expt_dir + "mean_reps_relevance.pdf")
plt.show()

print("input/model plot saved.", file=logfile, flush=True)


# # reordered mean reps, sorted by distance
#
# relevance_b = w * b
# relevance_s = w * s
#
# dists = s-b
# sorted_indices = np.argsort(dists)
#
# plt.figure(figsize=(9, 3.5))
# r_border = np.max([relevance_b, -relevance_b, relevance_s, -relevance_s])
#
# plt.scatter(np.arange(200), b[sorted_indices], s=20, c=relevance_b[sorted_indices], marker="s", label="mean value qcd", vmin=-r_border,
#             vmax=r_border, cmap=colormap)
# plt.scatter(np.arange(200), s[sorted_indices], s=20, c=relevance_s[sorted_indices], marker="^", label="mean value top", vmin=-r_border,
#             vmax=r_border, cmap=colormap)
# plt.gca().xaxis.grid(True)
# print(sorted_indices.shape, sorted_indices[::25].shape)
# plt.gca().set_xticks(np.concatenate((np.arange(200)[::25], [199])))
# plt.gca().set_xticklabels(np.concatenate((sorted_indices[::25], [sorted_indices[-1]])))
# plt.xlabel("dimension sorted for distance of top to qcd mean | $sort(top  - qcd)$")
# cbar = plt.colorbar(shrink=0.9)
# cbar.set_label("relevance in linear classification")
# leg = plt.legend()
# for marker in leg.legendHandles:
#     marker.set_color('grey')
# plt.tight_layout()
# plt.xticks(None)
# plt.savefig(expt_dir + "mean_reps_relevance_distsorted.pdf")
# plt.show()
#
# print("input/model plot saved.", file=logfile, flush=True)



# reordered mean reps, sorted by absolute distance

relevance_b = w * b
relevance_s = w * s

dists = abs(s-b)
sorted_indices = np.argsort(dists)


plt.figure(figsize=(9, 3.5))
r_border = np.max([relevance_b, -relevance_b, relevance_s, -relevance_s])

plt.scatter(np.arange(200), fb[sorted_indices], s=20, c=relevance_b[sorted_indices], marker="s", label="mean value qcd", vmin=-r_border,
            vmax=r_border, cmap=colormap)
plt.scatter(np.arange(200), fs[sorted_indices], s=20, c=relevance_s[sorted_indices], marker="^", label="mean value top", vmin=-r_border,
            vmax=r_border, cmap=colormap)
plt.gca().xaxis.grid(True)
print(sorted_indices.shape, sorted_indices[::25].shape)
plt.gca().set_xticks(np.concatenate((np.arange(200)[::25], [199])))
plt.gca().set_xticklabels(np.concatenate((sorted_indices[::25], [sorted_indices[-1]])))
plt.xlabel("dimension sorted for distance of top to qcd mean | $sort(|top  - qcd|)$")
cbar = plt.colorbar(shrink=0.9)
cbar.set_label("relevance in linear classification")
leg = plt.legend()
for marker in leg.legendHandles:
    marker.set_color('grey')
plt.tight_layout()
plt.xticks(None)
plt.savefig(expt_dir + "mean_reps_relevance_absdistsorted.pdf")
plt.show()

print("input/model plot saved.", file=logfile, flush=True)


# reordered mean reps, sorted by max mean

relevance_b = w * b
relevance_s = w * s

maxmean = np.amax(np.abs(np.array([b, s])), axis=0)
print(maxmean.shape)
sorted_indices = np.argsort(maxmean)

plt.figure(figsize=(9, 3.5))
r_border = np.max([relevance_b, -relevance_b, relevance_s, -relevance_s])

# plt.errorbar(np.arange(200)[::-5], fs[sorted_indices][::-5], yerr=s_iqr[sorted_indices][::-5], fmt="", linestyle='None', capsize=4, label="IQR", ecolor="red")
# plt.errorbar(np.arange(200)[::-5], fb[sorted_indices][::-5], yerr=b_iqr[sorted_indices][::-5], fmt="", linestyle='None', capsize=4, label="IQR", ecolor="blue")
plt.scatter(np.arange(200), fb[sorted_indices], s=20, c=relevance_b[sorted_indices], marker="s", label="mean value qcd", vmin=-r_border,
            vmax=r_border, cmap=colormap)
plt.scatter(np.arange(200), fs[sorted_indices], s=20, c=relevance_s[sorted_indices], marker="^", label="mean value top", vmin=-r_border,
            vmax=r_border, cmap=colormap)

plt.gca().xaxis.grid(True)
print(sorted_indices.shape, sorted_indices[::25].shape)
plt.gca().set_xticks(np.concatenate((np.arange(200)[::25], [199])))
plt.gca().set_xticklabels(np.concatenate((sorted_indices[::25], [sorted_indices[-1]])))
plt.xlabel("dimension sorted for highest deviation from 0 | $sort(max(qcd,top))$")
cbar = plt.colorbar(shrink=0.9)
cbar.set_label("relevance in linear classification")
leg = plt.legend(loc=2)
for marker in leg.legendHandles:
    marker.set_color('grey')
plt.tight_layout()
plt.xticks(None)
plt.savefig(expt_dir + "mean_reps_relevance_maxmeansorted.pdf")
plt.show()

print("input/model plot saved.", file=logfile, flush=True)



