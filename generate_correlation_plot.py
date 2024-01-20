import argparse
import pandas as pd
import time
import os
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns

plot_directory = "../plots/correlation/"
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)

def load_pickle(file_path):

    with open(file_path, "rb") as f:
        data = pkl.load(f)
        return data

def get_combined_dataframe(model):
    # get pageranks
    path = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/{}".format(model)
    all_files = os.listdir(path)
    csv_files = [os.path.join(path,file_name) for file_name in all_files if "netmeta" not in file_name and ".csv" in file_name]
    for file_name in csv_files:
        df = pd.read_csv(file_name)

        df = df[["node", "pagerank", "minority"]]

        hMM, hmm = file_name.split("hMM")[-1].split("-")[0], file_name.split("hmm")[-1].split("-")[0]
        hMM, hmm = hMM.replace(".csv",""), hmm.replace(".csv","")
        print("hMM:{}, hmm:{}".format(hMM,hmm))
        betn_centrality_path = "//home/mpawar/Homophilic_Directed_ScaleFree_Networks/centrality/betweenness/{}/_hMM{}_hmm{}.pkl".format(model,hMM,hmm)
        closness_centrality_path = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/centrality/closeness/{}/_hMM{}_hmm{}.pkl".format(model,hMM,hmm)
        
        betn_dict, close_dict = load_pickle(betn_centrality_path), load_pickle(closness_centrality_path)
        df_betn = pd.DataFrame([{"node": key, "betweenness": value} for key, value in betn_dict.items()])
        df_closeness = pd.DataFrame([{"node": key, "closeness": value} for key, value in close_dict.items()])
        df_join = pd.concat([df.set_index('node'),df_betn.set_index('node'),df_closeness.set_index('node')], axis=1, join='inner')
        df_join = df_join.loc[df_join['minority'] == 1]
     
        df_final = df_join[["pagerank", "betweenness", "closeness"]]
        corr = df_final.corr()
        ax = sns.heatmap(corr)
        fig = ax.get_figure()
        if not os.path.exists(plot_directory+model): os.makedirs(plot_directory+model)
        fig.savefig(plot_directory+"{}/corr_{}_{}.png".format(model,hMM,hmm))
        fig.clear()

def run(model):
    get_combined_dataframe(model)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model", type=str, default="")
    
    args = parser.parse_args()
    
    start_time = time.time()
    run(args.model)
    