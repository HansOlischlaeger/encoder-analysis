# train a t-SNE mapping to get a 2D visualisation of the jets

import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import os
import re


# load randomly generated test data
def makeNormalCluster(center, std, size):
    noise = np.random.normal(0, std, (size, center.shape[0]))
    return noise + center


dim = 50
size = 1000
center = np.zeros(dim)
center[0] = 1

c1 = makeNormalCluster(center, 1, size)
c2 = makeNormalCluster(-center, 1, size)

data = np.concatenate([c1, c2])

# visualize latent representations with t-SNE with different parameters
perplexities = np.arange(0, 5) * 15 + 10    # std is 30
learning_rates = np.array([200])            # std is 200

export_dir = "tSNE-visualization"

if not os.path.isdir(export_dir+"/pdf"):
    os.mkdir(export_dir+"/pdf")
if not os.path.isdir(export_dir+"/npy"):
    os.mkdir(export_dir+"/npy")

for p in perplexities:
    for lr in learning_rates:
        filename = f'tSNE_p={p}_lr={lr}.npy'
        if filename not in os.listdir(export_dir+"/npy"):
            np.save(f'{export_dir}/npy/{filename}', TSNE(n_components=2, perplexity=p, learning_rate=lr).fit_transform(data))
            print("Saved tSNE components in", filename)
        else:
            print("Skipped",p,lr,"| ",filename, "already exists!")

for filename in os.listdir(export_dir+"/npy"):

    if filename[:5] == "tSNE_" and filename[-4:] == ".npy":
        s = filename.split(".")[0].split("_")[1:]
        data_embedded = np.load(f'{export_dir}/npy/{filename}')
        plt.suptitle("Visualization with t-SNE")
        plt.title(' '.join(s))

        plt.scatter(data_embedded[:, 0][:size], data_embedded[:, 1][:size], alpha=0.5, color='C0', label='qcd')
        plt.scatter(data_embedded[:, 0][size:], data_embedded[:, 1][size:], alpha=0.3, color='C3', label='top')
        plt.show()
        pdfpath = f'{export_dir}/pdf/{filename.split(".")[0]}.pdf'
        plt.savefig(pdfpath)
        print("Saved tSNE pdf in", pdfpath)
    else:
        print("unexpected file:", filename)
