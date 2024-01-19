import os
import pandas as pd
import numpy as np
import seaborn as sns
import pickle as pkl
import argparse
import networkx as nx

import matplotlib.pyplot as plt

fm = 0.3
N = 1000
YM, Ym = 2.5, 2.5
d = 0.03
topk = 10 # to extract top k 
hMM_list, hmm_list = np.arange(0,1.1,0.1), np.arange(0,1.1,0.1)

plot_directory = "../plots/heatmap/centrality"
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)

def get_grid(files, model,centrality):
    grid = np.zeros((len(hmm_list),len(hMM_list)))
    for file_name in files:
        fm_ext = float(file_name.split("fm")[-1].split("-")[0])
        if fm_ext != fm: continue
        hMM, hmm = file_name.split("hMM")[-1].split("-")[0], file_name.split("hmm")[-1].split("-")[0]
        hMM, hmm = hMM.replace(".gpickle",""), hmm.replace(".gpickle","")
        hMM_idx, hmm_idx = int(float(hMM)*10), int(float(hmm)*10)
      
        print("hMM: {}, hmm: {}".format(hMM, hmm))
     

    
        
        g = nx.read_gpickle(file_name)
        dict_folder = "./centrality/{}/{}".format(centrality,model)
        if not os.path.exists(dict_folder): os.makedirs(dict_folder)
        dict_file_name = dict_folder+"/_hMM{}_hmm{}.pkl".format(hMM,hmm)
        if not os.path.exists(dict_file_name):
            if centrality == "betweenness":
                centrality_dict = nx.betweenness_centrality(g, normalized=True)
            elif centrality == "closeness":
                centrality_dict = nx.closeness_centrality(g)
            else:
                print("Invalid Centrality measure")
                return
            with open(dict_file_name, 'wb') as f:                
                pkl.dump(centrality_dict,f)
        else:
            with open(dict_file_name, 'rb') as f:                
                centrality_dict = pkl.load(f)
        
        
        minority_centrality = [val for node, val in centrality_dict.items() if g.nodes[node]["m"] == 1]
        avg_val = np.mean(minority_centrality)

        grid[hmm_idx][hMM_idx] = avg_val
    return grid

def generate_heatmap(file_path, model, reco_type, centrality):
    all_files = os.listdir(file_path)
    graph_files = [os.path.join(file_path,file_name) for file_name in all_files if "netmeta" not in file_name and ".gpickle" in file_name]
    grid = get_grid(graph_files, model, centrality)
    print(len(graph_files))
    if reco_type == "before":
        with open('dpah_before_{}.npy'.format(centrality), 'wb') as f:
                 print("Saving the centrality metrics before recommendation.")
                 np.save(f, grid)
                 heatmap = grid.T
    elif reco_type == "after":
        # with open('dpah_before_{}.npy'.format(centrality), 'rb') as f:
        #      before_fm_hat = np.load(f)
   
        # heatmap = grid.T - before_fm_hat.T
        print(np.max(grid),np.min(grid))
        heatmap = grid.T

    hmm_ticks = [np.round(hmm,2) for hmm in hmm_list]
    hMM_ticks = [np.round(hMM,2) for hMM in hMM_list]
    if centrality == "betweenness":
        vmin, vmax = 0, 0.004
    else:
        vmin, vmax = 0.2, 0.5
        print(vmin,vmax)
    ax = sns.heatmap(heatmap, cmap=plt.cm.coolwarm,xticklabels=hmm_ticks,yticklabels=hMM_ticks,vmin=vmin,vmax=vmax)
    ax.invert_yaxis()
    ax.set_xlabel("Homophily for Minority Class")
    ax.set_ylabel("Homophily for Majority Class")
    fig = ax.get_figure()
    fig.savefig(plot_directory+"/out_{}_{}_{}.png".format(reco_type,model,centrality))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model", type=str, default='.')
    parser.add_argument("--reco", help="before/after recommendation", type=str, default='')
    parser.add_argument("--centrality", help="closeness/betweenness", type=str, default='')
   
    args = parser.parse_args()
    path = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/{}".format(args.model)
    generate_heatmap(path, args.model, args.reco, args.centrality)
    args = parser.parse_args()
    