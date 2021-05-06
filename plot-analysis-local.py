#!/bin/env python3.7

# python script for producing plots from trained representations to measure performance

import numpy as np
from sklearn.manifold import TSNE
import matplotlib

matplotlib.use("pdf")
from matplotlib import pyplot as plt

import os

logfile = open("logfile.txt", "w")  # "a")
# logfile = open( args.logfile, "a" )
print("================================================", file=logfile, flush=True)
print("= plot-analysis.py =============================", file=logfile, flush=True)
print(file=logfile, flush=True)

expt_tag = "run_results"
expt_dir = "" + expt_tag + "/"

REPS_DIR = expt_dir + "test_reps.npy"
REPS_AUG_DIR = expt_dir + "test_reps_aug.npy"
LIN_DIR = expt_dir + "test_linear_cl.npy"
LAB_DIR = expt_dir + "test_lbs.npy"

reps = np.load(REPS_DIR)
reps_aug = np.load(REPS_AUG_DIR)
telin = np.load(LIN_DIR)
telab = np.load(LAB_DIR)
print("telab check:", np.count_nonzero(telab), np.count_nonzero(telab == 0), np.count_nonzero(telab)/np.count_nonzero(telab == 0))

print("Loaded all files successfully.", file=logfile, flush=True)

data = reps

# different parameters for t-SNE and setup
perplexities = np.arange(0, 7) * 15 + 10  # std is 30
learning_rates = np.array([200])  # std is 200

forceOverride = False

if not os.path.isdir(expt_dir + "analysis"):
    os.mkdir(expt_dir + "analysis")
if not os.path.isdir(expt_dir + "analysis/tSNE_npy"):
    os.mkdir(expt_dir + "analysis/tSNE_npy")

# train a number of t-SNEs and save them in analysis/tSNE_npy/
for p in perplexities:
    for lr in learning_rates:
        filename = f'tSNE_p={p}_lr={lr}.npy'
        if forceOverride or filename not in os.listdir(expt_dir + "analysis/tSNE_npy"):
            npypath = f'{expt_dir}analysis/tSNE_npy/{filename}'
            np.save(npypath,
                    TSNE(n_components=2, perplexity=p, learning_rate=lr).fit_transform(data))
            print("Saved tSNE components in", npypath, file=logfile, flush=True)
        else:
            print("Skipped", p, lr, "| ", filename, "already exists!", file=logfile, flush=True)

# generate plots for all t-SNEs saved in analysis/tSNE_npy/
for filename in os.listdir(expt_dir + "analysis/tSNE_npy/"):

    if filename[:5] == "tSNE_" and filename[-4:] == ".npy":
        s = filename.split(".")[0].split("_")[1:]
        data_embedded = np.load(f'{expt_dir}analysis/tSNE_npy/{filename}')
        plt.suptitle("Visualization with t-SNE")
        plt.title(' '.join(s))  # make a title with the parameters from file name information

        plt.scatter(data_embedded[:, 0][telab == 0], data_embedded[:, 1][telab == 0], alpha=0.5, color='C0',
                    label='qcd')
        plt.scatter(data_embedded[:, 0][telab == 1], data_embedded[:, 1][telab == 1], alpha=0.2, color='C3',
                    label='top')
        plt.legend()
        plt.tight_layout()
        pdfpath = f'{expt_dir}analysis/{filename.split(".")[0]}.pdf'
        plt.savefig(pdfpath)
        plt.close()
        print("Saved tSNE pdf in", pdfpath, file=logfile, flush=True)

    else:
        print("unexpected file in tSNE_npy folder:", filename, file=logfile, flush=True)

dists = [telin[telab == 0].squeeze(), telin[telab == 1].squeeze()]
print(dists[0].shape, dists[1].shape)
iqr = max(np.percentile(telin[telab == 1], 75) - np.percentile(telin[telab == 1], 25),
          np.percentile(telin[telab == 0], 75) - np.percentile(telin[telab == 0], 25))
binwidth = iqr * 2 * min(np.count_nonzero(telab), np.count_nonzero(telab == 0)) ** -0.333
plt.hist(dists, bins=np.arange(min(telin), max(telin) + binwidth, binwidth), color=['C0', 'C3'], label=["qcd", "top"],
         histtype="step")
plt.legend()
pdfpath = expt_dir + 'analysis/lin_hist.pdf'
plt.tight_layout()
plt.savefig(pdfpath)
plt.close()
print("Saved lin hist pdf in", pdfpath, file=logfile, flush=True)

print(file=logfile, flush=True)
print("= end of plot-analysis.py ======================", file=logfile, flush=True)
print("================================================", file=logfile, flush=True)
print(file=logfile, flush=True)
logfile.close()

with open("logfile.txt", "r") as log:
    print(log.read())
