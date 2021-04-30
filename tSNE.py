# train a t-SNE mapping to get a 2D visualisation of the jets

import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import os
import re


REPS_DIR = "run_results/test_reps.npy"
REPS_AUG_DIR = "run_results/test_reps_aug.npy"
LIN_DIR = "run_results/test_linear_cl.npy"

reps = np.load(REPS_DIR)
reps_aug = np.load(REPS_AUG_DIR)
lins = np.load(LIN_DIR)

data = reps
# data = np.concatenate([reps, reps_aug])

# visualize latent representations with t-SNE with different parameters
perplexities = []  # np.arange(0, 1) * 15 + 10    # std is 30
learning_rates = np.array([200])  # std is 200
size = 10000
export_dir = "tSNE-visualization"
forceOverride = True

if not os.path.isdir(export_dir + "/pdf"):
    os.mkdir(export_dir + "/pdf")
if not os.path.isdir(export_dir + "/npy"):
    os.mkdir(export_dir + "/npy")

for p in perplexities:
    for lr in learning_rates:
        filename = f'tSNE_p={p}_lr={lr}.npy'
        if forceOverride or filename not in os.listdir(export_dir + "/npy"):
            np.save(f'{export_dir}/npy/{filename}',
                    TSNE(n_components=2, perplexity=p, learning_rate=lr).fit_transform(data))
            print("Saved tSNE components in", filename)
        else:
            print("Skipped", p, lr, "| ", filename, "already exists!")

for filename in os.listdir(export_dir + "/npy"):

    if filename[:5] == "tSNE_" and filename[-4:] == ".npy":
        s = filename.split(".")[0].split("_")[1:]
        data_embedded = np.load(f'{export_dir}/npy/{filename}')
        plt.suptitle("Visualization with t-SNE")
        plt.title(' '.join(s))

        plt.scatter(data_embedded[:, 0][:size], data_embedded[:, 1][:size], alpha=0.5, color='C0', label='c1')
        plt.scatter(data_embedded[:, 0][size:], data_embedded[:, 1][size:], alpha=0.3, color='C3', label='c2')
        plt.legend()
        pdfpath = f'{export_dir}/pdf/{filename.split(".")[0]}.pdf'
        plt.savefig(pdfpath)
        print("Saved tSNE pdf in", pdfpath)
        plt.show()
    else:
        print("unexpected file:", filename)
