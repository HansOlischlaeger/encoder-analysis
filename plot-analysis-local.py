#!/bin/env python3.7

# python script for producing plots from trained representations to measure performance

import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import os



REPS_DIR = "run_results/test_reps.npy"
REPS_AUG_DIR = "run_results/test_reps_aug.npy"
LIN_DIR = "run_results/test_linear_cl.npy"

reps = np.load(REPS_DIR)
reps_aug = np.load(REPS_AUG_DIR)
lins = np.load(LIN_DIR)

print("Loaded all files successfully.")

for a in [reps, reps_aug, lins]:
    print(a.shape)

print("end")
