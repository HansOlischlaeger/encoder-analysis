#!/bin/env python3.7

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.jet_augs import rotate_jet, normalise_pt
from models.fcn import fully_connected_network
from models.fcn_linear import fully_connected_linear_network
from models.losses import contrastive_loss
from models.perf_eval import find_nearest, get_perf_stats, linear_classifier_test

path_to_rep_dir = "/home/ho/coding/encoder-analysis/"
path_to_run_dir = path_to_rep_dir + "run_dir/"
sys.path.append(path_to_run_dir+"clr_run2/args_NL_100_temp_0.1_bsize_1000_lr_0.01_SB_1.0_N_1")

import extargs as args

t0 = time.time()

# initialise logfile

logfile = open(args.logfile, "a")
print("logfile initialised", file=logfile, flush=True)

# set gpu device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device: " + str(device), flush=True, file=logfile)

# set up results directory

expt_tag = args.expt
expt_dir = path_to_rep_dir + "experiments/" + expt_tag + "/"

if os.path.isdir(expt_dir):
    sys.exit("ERROR: experiment already exists, don't want to overwrite it by mistake")
else:
    os.makedirs(expt_dir)

print("experiment: " + str(args.expt), file=logfile, flush=True)

print("loading data", flush=True, file=logfile)

# load data

sig_dat = np.load(args.sig_path, allow_pickle=True)
bkg_dat = np.load(args.bkg_path, allow_pickle=True)

# jets are pt ordered, now cropping to the leading 20

sdat = []
for i in sig_dat:
    sdat.append(np.array([i[0][0:20], i[1][0:20], i[2][0:20]]))
sdat = np.array(sdat)

bdat = []
for i in bkg_dat:
    bdat.append(np.array([i[0][0:20], i[1][0:20], i[2][0:20]]))
bdat = np.array(bdat)

input_size = int(20 * 3)

print("shuffling data and doing the S/B split", flush=True, file=logfile)

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

print("creating augmented datasets", flush=True, file=logfile)

# create the augmented datasets

# rotations

trdat_rot = np.zeros_like(trdat)

for i in range(trdat.shape[0]):
    rot_angle = np.random.uniform(low=0, high=2 * np.pi)
    trdat_rot[i, :, :] = rotate_jet(trdat[i, :, :], rot_angle=rot_angle)

tedat_rot = np.zeros_like(tedat)

for i in range(tedat.shape[0]):
    rot_angle = np.random.uniform(low=0, high=2 * np.pi)
    tedat_rot[i, :, :] = rotate_jet(tedat[i, :, :], rot_angle=rot_angle)

# normalise the pTs

trdat = normalise_pt(trdat)
trdat_rot = normalise_pt(trdat_rot)

tedat = normalise_pt(tedat)
tedat_rot = normalise_pt(tedat_rot)

# print data dimensions
print("training data shape: " + str(trdat.shape), flush=True, file=logfile)
print("testing data shape: " + str(tedat.shape), flush=True, file=logfile)
print("training labels shape: " + str(trlab.shape), flush=True, file=logfile)
print("testing labels shape: " + str(telab.shape), flush=True, file=logfile)

t1 = time.time()

print("time taken to load and preprocess data: " + str(np.round(t1 - t0, 2)) + " seconds", flush=True, file=logfile)

print("--- network architecture ---", flush=True, file=logfile)
print("input size: " + str(input_size), flush=True, file=logfile)
print("number of hidden layers: " + str(args.n_hidden), flush=True, file=logfile)
print("size of hidden layers: " + str(args.hidden_size), flush=True, file=logfile)
print("output size: " + str(args.output_size), flush=True, file=logfile)
print("--- network architecture ---", flush=True, file=logfile)

print("learning rate: " + str(args.learning_rate), flush=True, file=logfile)
print("temperature: " + str(args.temperature), flush=True, file=logfile)

print("initialising the network", flush=True, file=logfile)

# initialise the network

fcn = fully_connected_network(input_size, args.output_size, args.hidden_size, args.n_hidden, args.learning_rate)

# send network to device

fcn.to(device)

# set-up parameters for training the linear classifier

linear_input_size = args.output_size
linear_n_epochs = 200
linear_learning_rate = 0.01
linear_batch_size = args.batch_size

# doing the training

indices_list = torch.split(torch.randperm(trdat.shape[0]), args.batch_size)

print("starting training loop, running for " + str(args.n_epochs) + " epochs", flush=True, file=logfile)

auc_epochs = []
imtafe_epochs = []
losses = []

for epoch in range(args.n_epochs):
    te0 = time.time()

    # the inner loop goes through the dataset batch by batch
    for i, indices in enumerate(indices_list):
        losses_e = []
        fcn.optimizer.zero_grad()

        # pass all jets (including augmented) through the network
        x_i = trdat[indices, :, :]
        x_j = trdat_rot[indices, :, :]
        x_i = torch.Tensor(x_i).view(-1, input_size).to(device)
        x_j = torch.Tensor(x_j).view(-1, input_size).to(device)
        z_i = fcn(x_i)
        z_j = fcn(x_j)

        # compute the loss based on predictions of the net and the correct answers
        loss = contrastive_loss(z_i, z_j, args.temperature).to(device)
        # loss = loss.to( device )
        loss.backward()
        fcn.optimizer.step()
        losses_e.append(loss.detach().cpu().numpy())

    losses.append(np.mean(np.array(losses_e)))
    te1 = time.time()
    print("epoch: " + str(epoch) + ", loss: " + str(losses[-1]), flush=True, file=logfile)
    print("time taken: " + str(np.round(te1 - te0, 2)), flush=True, file=logfile)

    if epoch % 10 == 0:
        # run the linear classifier test
        te0_test = time.time()
        reps_tr = F.normalize(fcn(torch.tensor(trdat).view(-1, input_size).to(device)).detach().cpu()).numpy()
        reps_te = F.normalize(fcn(torch.tensor(tedat).view(-1, input_size).to(device)).detach().cpu()).numpy()
        out_dat_fe, out_lbs_fe, losses_fe = linear_classifier_test(linear_input_size, linear_batch_size,
                                                                   linear_n_epochs, linear_learning_rate, reps_tr,
                                                                   trlab, reps_te, telab)
        auc, imtafe = get_perf_stats(out_lbs_fe, out_dat_fe)
        te1_test = time.time()
        print("----------", flush=True, file=logfile)
        print("linear classifier test - time taken: " + str(np.round(te1_test - te0_test, 2)), flush=True, file=logfile)
        print("auc: " + str(np.round(auc, 4)) + ", imtafe: " + str(imtafe), flush=True, file=logfile)
        print("----------", flush=True, file=logfile)
        auc_epochs.append(auc)
        imtafe_epochs.append(imtafe)

t2 = time.time()
print("training DONE, training time: " + str(np.round(t2 - t1, 2)), flush=True, file=logfile)

print("evaluating trained network on testing data", flush=True, file=logfile)

# evaluate the network on the testing data

reps = F.normalize(fcn(torch.tensor(tedat).view(-1, input_size).to(device)).detach().cpu()).numpy()
reps_aug = F.normalize(fcn(torch.tensor(tedat_rot).view(-1, input_size).to(device)).detach().cpu()).numpy()

print("saving out data/results", flush=True, file=logfile)

# save out results

np.save(expt_dir + "clr_losses.npy", losses)
np.save(expt_dir + "test_reps.npy", reps)
np.save(expt_dir + "test_reps_aug.npy", reps_aug)
np.save(expt_dir + "auc_epochs.npy", np.array(auc_epochs))
np.save(expt_dir + "imtafe_epochs.npy", np.array(imtafe_epochs))

print("END of rep-learning run", flush=True, file=logfile)

print("----------------------------", flush=True, file=logfile)

print("STARTING the final linear classifier run", flush=True, file=logfile)
print("same learning rate: " + str(args.learning_rate), flush=True, file=logfile)
print("initialising the network", flush=True, file=logfile)

t3 = time.time()

reps_tr = F.normalize(fcn(torch.tensor(trdat).view(-1, input_size).to(device)).detach().cpu()).numpy()
reps_te = F.normalize(fcn(torch.tensor(tedat).view(-1, input_size).to(device)).detach().cpu()).numpy()

out_dat_f, out_lbs_f, losses_f = linear_classifier_test(linear_input_size, linear_batch_size, linear_n_epochs,
                                                        linear_learning_rate, reps_tr, trlab, reps_te, telab)
auc, imtafe = get_perf_stats(out_lbs_f, out_dat_f)
ep = 0
for lss in losses_f:
    print("epoch: " + str(ep) + ", loss: " + str(lss), flush=True, file=logfile)
    ep += 1
print("auc: " + str(auc), flush=True, file=logfile)
print("imtafe: " + str(imtafe), flush=True, file=logfile)
print(len(losses_f), flush=True, file=logfile)

t4 = time.time()

print("linear classifier test  DONE, training time: " + str(np.round(t4 - t3, 2)), flush=True, file=logfile)

print("saving output", flush=True, file=logfile)

# save out results

np.save(expt_dir + "linear_losses.npy", losses_f)
np.save(expt_dir + "test_linear_cl.npy", out_dat_f)
np.save(expt_dir + "test_lbs.npy", out_lbs_f)

t5 = time.time()
print("END of run, total time taken: " + str(np.round(t5 - t0, 2)), flush=True, file=logfile)
