import sys
import numpy as np
import pandas as pd
import scanpy as sc

from matplotlib import pyplot as plt
from matplotlib.colors import rgb2hex
import seaborn as sns

from sklearn.neighbors import NearestNeighbors

cells = pd.read_csv(str(sys.argv[1]), header=0, delimiter = "\t")

sample = str(sys.argv[1])
filename = sample.split('.', 1)[0]

def plotter(df, label_cols, s=0.2):
    n_cols = len(label_cols)
    
    # might need adjusting the figure size
    fig = plt.figure(figsize=(2*n_cols,4), dpi=180)
    
    for k,c in enumerate(label_cols):
        ax = fig.add_subplot(1,n_cols,k+1)
        ax.set_aspect('equal')
        ax.axis('off')
        cmap = {
            l:rgb2hex(c) for l,c in zip(np.unique(df[c]), sns.color_palette('tab10', len(df[c].unique())))
        }
        print(cmap)
        colors = [cmap[l] for l in df[c]]
        ax.scatter(df['coords_1'], df['coords_2'], s=s, lw=0, c=colors)
        ax.set_title(c)

#     ax = fig.add_subplot(1,2,2)
#     ax.set_aspect('equal')
#     ax.axis('off')
#     cmap = {
#         l:rgb2hex(c) for l,c in zip(np.unique(df.is_immune), sns.color_palette('tab10', len(df.is_immune.unique())))
#     }
#     print(cmap)
#     colors = [cmap[l] for l in df.is_immune]
#     ax.scatter(df['coords_1'], df['coords_2'], s=s, lw=0, c=colors)

NN = NearestNeighbors(n_neighbors=50, radius=820, n_jobs=16) ### change radius
NN.fit(cells[['coords_1', 'coords_2']])

X = cells[['coords_1', 'coords_2']].values
nbrs = NN.radius_neighbors(X=X, radius=820, return_distance=False, sort_results=False) ### change radius

import tqdm.auto as tqdm

cells['neighbor_immune_rate'] = 0.
cells['is_immune_numeric'] = (cells['is_immune']=='immune').astype(np.uint8)
L = cells['is_immune_numeric'].values
Y = np.zeros(cells.shape[0])

for c,n in tqdm.tqdm(enumerate(nbrs), total=cells.shape[0]):
    Y[c] = np.mean(L[n])
    
cells['neighbor_immune_rate'] = Y

_ = plt.hist(cells.neighbor_immune_rate, bins=100)
plt.title(str(filename))

extension2 = '.hist.png'
out2 = filename+extension2
plt.savefig(str(out2))
plt.close()

# pick some cutoff
cells['immune_neighborhood'] = 'non_immune'
cells.loc[cells['neighbor_immune_rate']>0.4, 'immune_neighborhood'] = 'immune'

plotter(cells, label_cols=['celltype', 'is_immune', 'immune_neighborhood'])

out4 = filename + '.is_immune.png'
plt.savefig(str(out4))
plt.close()

# summarizing cell types 
cols = ['perc_40', 'perc_50', 'perc_60', 'perc_70', 'perc_80', 'total_count']
perc_threshold = [0.4,0.5,0.6,0.7,0.8]
celltype = ['neopla', 'connec', 'necros', 'inflam', 'no-neo']
df2 = pd.DataFrame(columns=cols, index=celltype)

for i in celltype:
    df2.loc[i].perc_40 = len(cells.loc[(cells['neighbor_immune_rate']>=0.4) & (cells['celltype'] == i)])
    df2.loc[i].perc_50 = len(cells.loc[(cells['neighbor_immune_rate']>=0.5) & (cells['celltype'] == i)])
    df2.loc[i].perc_60 = len(cells.loc[(cells['neighbor_immune_rate']>=0.6) & (cells['celltype'] == i)])
    df2.loc[i].perc_70 = len(cells.loc[(cells['neighbor_immune_rate']>=0.7) & (cells['celltype'] == i)])
    df2.loc[i].perc_80 = len(cells.loc[(cells['neighbor_immune_rate']>=0.8) & (cells['celltype'] == i)])
    df2.loc[i].total_count = len(cells.loc[(cells['celltype']==i)])


df2['sample'] = filename
output = filename+'celltype_threshold.csv'
df2.to_csv(output, sep='\t', index=True)



