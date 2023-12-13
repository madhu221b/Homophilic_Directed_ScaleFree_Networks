import os
import pandas as pd
import numpy as np
import seaborn as sns
import pickle as pkl
import argparse
import matplotlib.pyplot as plt

fm = 0.3
N = 1000
YM, Ym = 2.5, 2.5
d = 0.03
topk = 10 # to extract top k 
hMM_list, hmm_list = np.arange(0,1.1,0.1), np.arange(0,1.1,0.1)

plot_directory = "../plots/heatmap/visibility"
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)

def get_grid(files):
    grid = np.zeros((len(hmm_list),len(hMM_list)))
    for file_name in files:
      
        hMM, hmm = file_name.split("hMM")[-1].split("-")[0], file_name.split("hmm")[-1].split("-")[0]
        hMM, hmm = hMM.replace(".csv",""), hmm.replace(".csv","")
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
       #  grid[hmm_idx][hMM_idx] = 1
    return grid

def generate_heatmap(file_path,model, reco_type):
    all_files = os.listdir(file_path)
    csv_files = [os.path.join(file_path,file_name) for file_name in all_files if "netmeta" not in file_name and ".csv" in file_name]
    grid = get_grid(csv_files)
    print(len(csv_files))
    if reco_type == "before":
        with open('dpah_before.npy', 'wb') as f:
                 np.save(f, grid)
        heatmap = grid.T  - fm
    elif reco_type == "after":
        print("Going here")
        with open('dpah_before.npy', 'rb') as f:
             before_fm_hat = np.load(f)

        heatmap = grid.T - before_fm_hat.T
    hmm_ticks = [np.round(hmm,2) for hmm in hmm_list]
    hMM_ticks = [np.round(hMM,2) for hMM in hMM_list]
    if reco_type == "before": vmax,vmin = 1, -1
    else: vmax,vmin = 0.32, -0.32
    ax = sns.heatmap(heatmap, cmap=plt.cm.coolwarm,xticklabels=hmm_ticks,yticklabels=hMM_ticks,vmax=vmax,vmin=vmin)
    # ax = sns.heatmap(heatmap, cmap=plt.cm.coolwarm,xticklabels=hmm_ticks,yticklabels=hMM_ticks)
    ax.invert_yaxis()
    cbar = ax.collections[0].colorbar
    print(vmax,vmin)
    cbar.set_ticks([vmin, float((vmax+vmin)/2), vmax])
    if reco_type == "before":
      cbar.set_ticklabels(["{} : Minorities \n Underrepresented".format(vmin),0,"{} : Minorities \n  Overrepresented".format(vmax)],update_ticks=True)
    else:
        cbar.set_ticklabels(["{} : Negative \n Variation".format(vmin),0,"{} : Positive \n Variation".format(vmax)],update_ticks=True)

    ax.set_xlabel("Homophily for Minority Class")
    ax.set_ylabel("Homophily for Majority Class")

    fig = ax.get_figure()
    fig.savefig(plot_directory+"/out_{}_{}.png".format(model,reco_type),bbox_inches='tight')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model", type=str, default='.')
    parser.add_argument("--reco", help="before/after recommendation", type=str, default='')
    args = parser.parse_args()
    path = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/{}".format(args.model)
    generate_heatmap(path,args.model, args.reco)
    args = parser.parse_args()
    