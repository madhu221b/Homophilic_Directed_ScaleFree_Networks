import os
import pandas as pd
import numpy as np
import seaborn as sns
import pickle as pkl
import argparse

from palettable.cartocolors.diverging import Geyser_7
from palettable.colorbrewer.diverging import RdBu_3

fm = 0.3
N = 1000
YM, Ym = 2.5, 2.5
d = 0.03
topk = 10 # to extract top k 
hMM_list, hmm_list = np.arange(0,1.1,0.1), np.arange(0,1.1,0.1)


def get_grid(files):
    grid = np.zeros((len(hmm_list),len(hMM_list)))
    for file_name in files:
        hMM, hmm = file_name.split("hMM")[-1].split("-")[0], file_name.split("hmm")[-1].split("-")[0]
        hMM_idx, hmm_idx = int(float(hMM)*10), int(float(hmm)*10)
        print("hMM: {}, hmm: {}".format(hMM, hmm))
        
        # Sanity checks
        N_extracted = int(file_name.split("N")[-1].split("-")[0])
        assert N == N_extracted # ensure that these files are of the N we want
        fm_extracted = float(file_name.split("fm")[-1].split("-")[0])
        assert fm == fm_extracted # ensure that these files are of the fm we want
        Ym_extracted = float(file_name.split("plom")[-1].split("-")[0])
        assert Ym == Ym_extracted # ym is same
        YM_extracted = float(file_name.split("ploM")[-1].split("-")[0])
        assert YM == YM_extracted # YM is same
        d_extracted = float(file_name.split("d")[-1].split("-")[0])
        assert d == d_extracted # d is same
        
        
        df = pd.read_csv(file_name)
        
        # compute no of entries in topk
        total = len(df)
        k = round(topk/100.,2)    # k%
        t = int(round(k*total))   # No. of unique ranks in top-k

        df.sort_values(by='pagerank', ascending=False, inplace=True)
        topnodes = df[0:t] # first top-k ranks (list)
        fm_hat = topnodes.minority.sum()/topnodes.shape[0]
        
        grid[hmm_idx][hMM_idx] = fm_hat
    return grid

def generate_heatmap(file_path, reco_type):
    print("file path", file_path)
    all_files = os.listdir(file_path)
    csv_files = [os.path.join(file_path,file_name) for file_name in all_files if "netmeta" not in file_name and ".csv" in file_name]
    grid = get_grid(csv_files)
    if reco_type == "before":
        heatmap = grid.T  - fm

    hmm_ticks = [np.round(hmm,2) for hmm in hmm_list]
    hMM_ticks = [np.round(hMM,2) for hMM in hMM_list]
    ax = sns.heatmap(heatmap, cmap=Geyser_7.mpl_colormap,xticklabels=hmm_ticks,yticklabels=hMM_ticks)
    ax.invert_yaxis()
    ax.set_xlabel("Homophily for Minority Class")
    ax.set_ylabel("Homophily for Majority Class")
    fig = ax.get_figure()
    fig.savefig("out_{}.png".format(reco_type))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path/generate heat map on which model", type=str, default='.')
    parser.add_argument("--reco", help="before/after recommendation", type=str, default='')
    args = parser.parse_args()
    generate_heatmap(args.path, args.reco)
    args = parser.parse_args()
    