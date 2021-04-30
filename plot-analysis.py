#!/bin/env python3.7

# python script for producing plots from trained representations to measure performance

import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import os
import extargs as args

logfile = open( args.logfile, "a" )
print( "================================================", file=logfile, flush=True )
print( "= plot-analysis.py =============================", file=logfile, flush=True )
print(, file=logfile, flush=True )

expt_tag=args.expt
expt_dir="/remote/gpu04/olischlaeger/projects/rep_learning/experiments/"+expt_tag+"/"



REPS_DIR = expt_dir+"test_reps.npy"
REPS_AUG_DIR = expt_dir+"test_reps_aug.npy"
LIN_DIR = expt_dir+"test_linear_cl.npy"

reps = np.load(REPS_DIR)
reps_aug = np.load(REPS_AUG_DIR)
lins = np.load(LIN_DIR)

print("Loaded all files successfully.", file=logfile, flush=True )


for a in [reps, reps_aug, lins]:
    print(a.shape)

print("end", file=logfile, flush=True )
