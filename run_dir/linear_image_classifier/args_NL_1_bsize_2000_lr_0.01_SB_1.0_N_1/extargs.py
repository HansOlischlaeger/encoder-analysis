#!/bin/env python3

path_to_rep_dir = "/home/ho/coding/encoder-analysis/"
path_to_data_dir = "/home/ho/coding/encoder-analysis/data-parsed/"
logfile = path_to_rep_dir + "run_dir/linear_image_classifier/args_NL_1_bsize_2000_lr_0.01_SB_1.0_N_1/logfile.txt"
sig_path = path_to_data_dir + "toptagging/train.img40X40.crop.rot.flip.norm.sig.h5"
bkg_path = path_to_data_dir + "toptagging/train.img40X40.crop.rot.flip.norm.bkg.h5"
output_size = 1
# n_hidden =
# hidden_size =
sbratio = 1.0
n_epochs = 100
learning_rate = 0.06
batch_size = 2000# 2000
# temperature =
expt = "linear_image_classifier/NL_1_sb_1.0_bsize_2000_lr_0.01_1"
