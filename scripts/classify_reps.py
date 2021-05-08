
# load reps and labels
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

sys.path.append(path_to_run_dir + "clr_run15/args_NL_200_NC_20_temp_0.1_bsize_1000_lr_0.001_SB_0.05_N_1")
import extargs as args

run_dir = path_to_run_dir + "clr_run15/args_NL_200_NC_20_temp_0.1_bsize_1000_lr_0.001_SB_0.05_N_1/"
expt_dir = path_to_rep_dir + "experiments/clr_run15/NL_200_NC_20_temp_0.1_sb_0.05_bsize_1000_lr_0.001_1/"
logfile = open(run_dir+"logfile.txt", "w") ############ change this to "a" before deployment!!
print("logfile initialised", file=logfile, flush=True)


# load reps and labels
telab = np.load(expt_dir + "test_lbs.npy")
reps_te = np.load(expt_dir + "test_reps.npy")



# train classifier with test data
linear_input_size = args.output_size
linear_n_epochs = 200
linear_learning_rate = 0.01
linear_batch_size = args.batch_size
indices_list = torch.split(torch.randperm(reps_te.shape[0]), linear_batch_size)
fcn = fully_connected_linear_network(linear_input_size, 1, linear_learning_rate)
print("starting training loop, running for " + str(linear_n_epochs) + " epochs", flush=True, file=logfile)

bce_loss = nn.BCELoss()
sigmoid = nn.Sigmoid()

for epoch in range(linear_n_epochs):
    losses = []

    # the inner loop goes through the dataset batch by batch
    for i, indices in enumerate(indices_list):
        fcn.optimizer.zero_grad()

        # pass all jets (including augmented) through the network
        x = reps_te[indices, :]

        l = telab[indices]
        x = torch.Tensor(x).view(-1, linear_input_size)
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



# evaluate the network on the testing data
print("evaluating trained network on testing data", flush=True, file=logfile)

out_dat = sigmoid(fcn(torch.Tensor(reps_te).view(-1, linear_input_size))).detach().numpy()
out_lbs = telab
auc, imtafe = get_perf_stats(out_lbs, out_dat)

print("auc: " + str(auc), flush=True, file=logfile)
print("imtafe: " + str(imtafe), flush=True, file=logfile)

print("saving output", flush=True, file=logfile)

# save out results

np.save(expt_dir + "losses.npy", losses)        # loss development during training
np.save(expt_dir + "out_dat.npy", out_dat)      # guesses of fcn
np.save(expt_dir + "reps_te.npy", reps_te)      # test input (representations learned with contrastive loss)
np.save(expt_dir + "telab.npy", telab)          # test true labels

# save state dict of the model for later lrp
torch.save(fcn.state_dict(), expt_dir + "state_dict.pt")

print("END of run", flush=True, file=logfile)
