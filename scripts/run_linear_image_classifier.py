#!/bin/env python3.7

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

logfile = open(args.logfile, "w") ############ change this to "a" before deployment!!
print("logfile initialised", file=logfile, flush=True)

# set up results directory

expt_tag = args.expt
expt_dir = path_to_rep_dir + "experiments/" + expt_tag + "/"

if os.path.isdir(expt_dir):
    pass
    #sys.exit("ERROR: experiment already exists, don't want to overwrite it by mistake")

else:
    os.makedirs(expt_dir)

print("experiment: " + str(args.expt), file=logfile, flush=True)

print("loading data", flush=True, file=logfile)

# load jet image data

df_pt_sig = pd.read_hdf(args.sig_path, key='table', start=0, stop=20000)  # 100000
df_pt_bg = pd.read_hdf(args.bkg_path, key='table', start=0, stop=20000)  # 100000

sdat = np.zeros((df_pt_sig.shape[0], 1600))
for i in range(df_pt_sig.to_numpy().shape[0]):
    sdat[i] = df_pt_sig.to_numpy()[i][0:1600]
bdat = np.zeros((df_pt_bg.shape[0], 1600))
for i in range(df_pt_bg.to_numpy().shape[0]):
    bdat[i] = df_pt_bg.to_numpy()[i][0:1600]

input_size = int(40 * 40)

print("creating labelled dataset and shuffling", flush=True, file=logfile)

# creating the dataset

nsig = int(args.sbratio * len(bdat))

labels = [0 for i in bdat] + [1 for i in sdat[0:nsig]]
dat = list(bdat) + list(sdat)

ldz = list(zip(dat, labels))
random.shuffle(ldz)
dat, labels = zip(*ldz)

dat = np.array(dat)
labels = np.array(labels)

print("splitting into train/test", flush=True, file=logfile)

# train/test split

njets = len(dat)
ntrain = int(0.9 * njets)
ntest = int(0.1 * njets)

trdat = dat[0:ntrain]
tedat = dat[-ntest:]
trlab = labels[0:ntrain]
telab = labels[-ntest:]

# print data dimensions
print("training data shape: " + str(trdat.shape), flush=True, file=logfile)
print("testing data shape: " + str(tedat.shape), flush=True, file=logfile)
print("training labels shape: " + str(trlab.shape), flush=True, file=logfile)
print("testing labels shape: " + str(telab.shape), flush=True, file=logfile)

print("--- network architecture ---", flush=True, file=logfile)
print("input size: " + str(input_size), flush=True, file=logfile)
print("number of hidden layers: 0", flush=True, file=logfile)
print("output size: " + str(args.output_size), flush=True, file=logfile)
print("--- network architecture ---", flush=True, file=logfile)

print("learning rate: " + str(args.learning_rate), flush=True, file=logfile)

print("initialising the network", flush=True, file=logfile)

# initialise the network

fcn = fully_connected_linear_network(input_size, args.output_size, args.learning_rate)

# doing the training

indices_list = torch.split(torch.randperm(trdat.shape[0]), args.batch_size)

print("starting training loop, running for " + str(args.n_epochs) + " epochs", flush=True, file=logfile)

bce_loss = nn.BCELoss()
sigmoid = nn.Sigmoid()


for epoch in range(args.n_epochs):
    losses = []

    # the inner loop goes through the dataset batch by batch
    for i, indices in enumerate(indices_list):
        fcn.optimizer.zero_grad()

        # pass all jets (including augmented) through the network
        x = trdat[indices, :]
        l = trlab[indices]
        x = torch.Tensor(x).view(-1, input_size)
        l = torch.Tensor(l).view(-1, 1)
        z = sigmoid(fcn(x))
        # print( x.size() , flush=True, file=logfile )
        # print( l.size(), flush=True, file=logfile )
        # print( z.size(), flush=True, file=logfile )

        # compute the loss based on predictions of the net and the correct answers
        loss = bce_loss(z, l)
        loss.backward()
        fcn.optimizer.step()

    losses.append(loss.item())
    print("epoch: " + str(epoch), flush=True, file=logfile)
    print("loss: " + str(loss.detach().numpy()), flush=True, file=logfile)


print("training DONE", flush=True, file=logfile)

print("evaluating trained network on testing data", flush=True, file=logfile)

# evaluate the network on the testing data

out_dat = sigmoid(fcn(torch.Tensor(tedat).view(-1, input_size))).detach().numpy()
out_lbs = telab
auc, imtafe = get_perf_stats(out_lbs, out_dat)

print("auc: " + str(auc), flush=True, file=logfile)
print("imtafe: " + str(imtafe), flush=True, file=logfile)

print("saving output", flush=True, file=logfile)

# save out results

np.save(expt_dir + "losses.npy", losses)        # loss development during training
np.save(expt_dir + "out_dat.npy", out_dat)      # guesses of fcn
np.save(expt_dir + "tedat.npy", tedat)          # test input
np.save(expt_dir + "telab.npy", telab)          # test true labels

# save state dict of the model for later lrp
torch.save(fcn.state_dict(), expt_dir + "state_dict.pt")

print("END of run", flush=True, file=logfile)
