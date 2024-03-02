import os
import pickle as pkl
from collections import Counter
import numpy as np

import networkx as nx
from sklearn.manifold import TSNE
from gensim import models

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from org.gesis.lib.n2v_utils import get_walks
from visualize_plots import get_label, hMM_list, hmm_list

DIM = 64
WALK_LEN = 10
NUM_WALKS = 200


def get_walks_(g, model,walker_file_name):
    if "fw" in model:
        p = q = 1.0
        walks = get_walks(g,model="fw")
    else:
        extra_params = {"beta":2.0}
        walks = get_walks(g,model="indegree",extra_params=extra_params)
   
    
    print("Saving walker at file name:", walker_file_name)
    with open(walker_file_name, 'wb') as f:
        pkl.dump(walks, f)
    return walks

def visualize_walk_prop(walks,g):
    walk_dict = dict() # {"m":{"m":[],"M":[]}, "M":{"m":[],"M":[]}}
    node_attr = nx.get_node_attributes(g, "group")

    for walk in walks:
        walk = [int(_) for _ in walk]
        source_node = get_label(node_attr[walk[0]])

        if len(walk) == 1: continue
        if source_node not in walk_dict: walk_dict[source_node] = dict()

        walk_ids = [get_label(node_attr[_]) for _ in walk[1:]]
        count_ids = Counter(walk_ids)
        sum_ = sum(count_ids.values())
        count_ids = {k:np.round((v/sum_)*100.0,2) for k,v in count_ids.items()}
        for k, v in count_ids.items():
            if k not in walk_dict[source_node]: walk_dict[source_node][k] = list()
            walk_dict[source_node][k].append(v)
    

    print("Percent of m visited when rws start from m: ", sum(walk_dict["m"]["m"])/len(walk_dict["m"]["m"]))
    print("Percent of M visited when rws start from m: ", sum(walk_dict["m"]["M"])/len(walk_dict["m"]["M"]))
    print("~~~~~~~~~~~~~~~")
    print("Percent of m visited when rws start from M: ", sum(walk_dict["M"]["m"])/len(walk_dict["M"]["m"]))
    print("Percent of M visited when rws start from M: ", sum(walk_dict["M"]["M"])/len(walk_dict["M"]["M"]))
    return walk_dict

def plot_rw(filename,hMM,hmm):
    g = nx.read_gpickle(filename)
    node2group = {node:g.nodes[node]["m"] for node in g.nodes()}
    nx.set_node_attributes(g, node2group, 'group')

    if "fw" in filename: model = "fw"
    else: model = "indegree"
    
    walker_folder = "./walker/{}".format(model)
    if not os.path.exists(walker_folder): os.makedirs(walker_folder)
    walker_file_name = walker_folder + "/_hMM{}_hmm{}.pkl".format(hMM,hmm)
    print("Walker file name: ", walker_file_name)
    if not os.path.exists(walker_file_name):
        walks = get_walks_(g, model, walker_file_name)
    else:
        with open(walker_file_name, 'rb') as f:
                walks = pkl.load(f)
    
    walk_dict = visualize_walk_prop(walks,g)
    return walk_dict

def get_fig(hmm):
    linestyles = ["solid", "dashed"]
    colors = ["o","b"]
    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax2 = ax1.twinx()
    ax2.set_axis_off()

    main_directory = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/"
    
    models = ["indegree_beta_2.0"]
    
    for i, model in enumerate(models):
        res_dict = {"startm_avgm":{},
                   "startm_avgM":{},
                   "startM_avgm":{},
                   "startM_avgM":{}}
        for hMM in hMM_list:
            hMM = np.round(hMM,2)
            file_path = main_directory+"{}_fm_0.3/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_{}.gpickle".format(model,model,hMM,hmm,29)
            walk_dict = plot_rw(file_path,hMM,hmm)

            startm_avgm = sum(walk_dict["m"]["m"])/len(walk_dict["m"]["m"])
            startm_avgM = sum(walk_dict["m"]["M"])/len(walk_dict["m"]["M"])
            startM_avgm = sum(walk_dict["M"]["m"])/len(walk_dict["M"]["m"])
            startM_avgM = sum(walk_dict["M"]["M"])/len(walk_dict["M"]["M"])

            res_dict["startm_avgm"][hMM] = startm_avgm
            res_dict["startm_avgM"][hMM] = startm_avgM
            res_dict["startM_avgm"][hMM] = startM_avgm
            res_dict["startM_avgM"][hMM] = startM_avgM
    
 
        
        for j, (k,sub_dict) in enumerate(res_dict.items()):
            if "avgm" in k: 
                color = "orange"
                label = "avg percent of m in rws"
            else: 
                color = "b"
                label = "avg percent of M in rws"
            if k.startswith("startm"): linestyle = "-"
            else: linestyle = "--"
            ax1.plot(sub_dict.keys(), sub_dict.values(), marker="o",label=label,color=color,linestyle=linestyle)
     

    dummy_lines = []
    for linestyle in ["-","--"]:
        dummy_lines.append(ax1.plot([],[], c="black", ls = linestyle)[0])
    
    lines = ax1.get_lines()
    ax1.legend([line for line in lines if line.get_linestyle()=="-"],[line.get_label() for line in lines if line.get_linestyle()=="-"], bbox_to_anchor=(0.3, 1.0))
    ax2.legend([dummy_line for dummy_line in dummy_lines],["Start Node m", "Start Node M"], bbox_to_anchor=(0.7, 1.0))
    ax1.set_xlabel("Varying hMM for Fixed hmm: {}".format(hmm))
    ax1.set_ylabel("perc of nodes in random walks")
    ax1.set_ylim(0,105)

    fig.savefig('walk_perc_hmm_{}.png'.format(hmm),bbox_inches='tight')   # save the figure to file
    plt.close(fig)  



def get_fig_v2(hMM):
    linestyles = ["solid", "dashed"]
    colors = ["o","b"]
    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax2 = ax1.twinx()
    ax2.set_axis_off()

    main_directory = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/"
    
    models = ["indegree_beta_2.0"]
    
    for i, model in enumerate(models):
        res_dict = {"startm_avgm":{},
                   "startm_avgM":{},
                   "startM_avgm":{},
                   "startM_avgM":{}}
        for hmm in hmm_list:
            hmm = np.round(hmm,2)
            file_path = main_directory+"{}_fm_0.3/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_{}.gpickle".format(model,model,hMM,hmm,29)
            walk_dict = plot_rw(file_path,hMM,hmm)

            startm_avgm = sum(walk_dict["m"]["m"])/len(walk_dict["m"]["m"])
            startm_avgM = sum(walk_dict["m"]["M"])/len(walk_dict["m"]["M"])
            startM_avgm = sum(walk_dict["M"]["m"])/len(walk_dict["M"]["m"])
            startM_avgM = sum(walk_dict["M"]["M"])/len(walk_dict["M"]["M"])

            res_dict["startm_avgm"][hmm] = startm_avgm
            res_dict["startm_avgM"][hmm] = startm_avgM
            res_dict["startM_avgm"][hmm] = startM_avgm
            res_dict["startM_avgM"][hmm] = startM_avgM
    
 
        
        for j, (k,sub_dict) in enumerate(res_dict.items()):
            if "avgm" in k: 
                color = "orange"
                label = "avg percent of m in rws"
            else: 
                color = "b"
                label = "avg percent of M in rws"
            if k.startswith("startm"): linestyle = "-"
            else: linestyle = "--"
            ax1.plot(sub_dict.keys(), sub_dict.values(), marker="o",label=label,color=color,linestyle=linestyle)
     

    dummy_lines = []
    for linestyle in ["-","--"]:
        dummy_lines.append(ax1.plot([],[], c="black", ls = linestyle)[0])
    
    lines = ax1.get_lines()
    ax1.legend([line for line in lines if line.get_linestyle()=="-"],[line.get_label() for line in lines if line.get_linestyle()=="-"], bbox_to_anchor=(0.3, 1.0))
    ax2.legend([dummy_line for dummy_line in dummy_lines],["Start Node m", "Start Node M"], bbox_to_anchor=(0.7, 1.0))
    ax1.set_xlabel("Varying hmm for Fixed hMM: {}".format(hMM))
    ax1.set_ylabel("perc of nodes in random walks")
    ax1.set_ylim(0,105)

    fig.savefig('walk_perc_hMM_{}.png'.format(hMM),bbox_inches='tight')   # save the figure to file
    plt.close(fig)  



if __name__ == "__main__":
    main_directory = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/"

    hMM = 1.0
    get_fig_v2(hMM)

    # for hmm in hmm_list:
    #     hMM = 0.0
    #     hmm = np.round(hmm,2)
    #     # file_name = main_directory+"fw_p_1.0_q_1.0_fm_0.3/fw_p_1.0_q_1.0-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_28.gpickle".format(hMM,hmm)
    #     file_name = main_directory+"indegree_beta_2.0_fm_0.3/indegree_beta_2.0-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_28.gpickle".format(hMM,hmm)
    #     plot_rw(file_name,hMM,hmm)