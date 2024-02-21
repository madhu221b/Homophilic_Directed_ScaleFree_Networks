import random
import os
import numpy as np
import pandas as pd
import pickle as pkl
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


from fairwalk.fairwalk  import FairWalk
from degreewalk.customwalk  import CustomWalk
from node2vec import Node2Vec
from node2vec_code.node2vec.node2vec import Node2Vec as custom_N2V
from common_ngh_aware.node2vec.node2vec import Node2Vec as common_N2V

from walkers.degreewalker import DegreeWalker
from walkers.indegreewalker import InDegreeWalker
from walkers.commonnghaware import CommonNeighborWalker
from walkers.levywalker import LevyWalker
from walkers.fairindegreewalker import FairInDegreeWalker
# Hyperparameter for node2vec/fairwalk
DIM = 64
WALK_LEN = 10
NUM_WALKS = 200

walker_dict = {
  "degree" : DegreeWalker,
  "indegree": InDegreeWalker,
  "fairindegree": FairInDegreeWalker, 
  "commonngh": CommonNeighborWalker,
  "levy": LevyWalker

}
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def rewiring_list(G, node, number_of_rewiring):
        nodes_to_be_unfollowed = []
        node_neighbors = np.array(list(G.successors(node)))
        nodes_to_be_unfollowed = np.random.permutation(node_neighbors)[:number_of_rewiring]
        return list(map(lambda x: tuple([node, x]), nodes_to_be_unfollowed))

def get_walks(G,model="n2v",extra_params=dict(),num_cores=8):
    if model == "n2v":
         node2vec = Node2Vec(G, dimensions=DIM, walk_length= 10, num_walks=4, workers=num_cores)
         return node2vec.walks
    elif model == "fw":
        fw_model = FairWalk(G, dimensions=DIM, walk_length=10, num_walks=4, workers=num_cores)
        return fw_model.walks

    WalkerObj = walker_dict[model] # degree_beta_1.0 for instance
    walkobj = WalkerObj(G, dimensions=DIM, walk_len=10, num_walks=4, workers=num_cores,**extra_params)
    return walkobj.walks

def recommender_model_walker(G,t=0,path="",model="n2v",extra_params=dict(),num_cores=8, is_walk_viz=False):
    WalkerObj = walker_dict[model.split("_")[0]] # degree_beta_1.0 for instance
    walkobj = WalkerObj(G, dimensions=DIM, walk_len=WALK_LEN, num_walks=NUM_WALKS, workers=num_cores,**extra_params)
    if is_walk_viz:
        dict_path = path.replace(".gpickle","") + "_frac.pkl"
        print(dict_path)
        get_walk_plots(walkobj.walks, G,t,dict_path)
       
    model = walkobj.fit() 
    emb_df = (pd.DataFrame([model.wv.get_vector(str(n)) for n in G.nodes()], index = G.nodes))
    return model, emb_df


def recommender_model(G,t=0,path="",model="n2v",p=1,q=1,num_cores=8, is_walk_viz=False):
    if model == "n2v":
       print("[N2V] Using p value: {}, Using q value : {}".format(p,q))
       node2vec = Node2Vec(G, dimensions=DIM, walk_length=WALK_LEN, num_walks=NUM_WALKS, workers=num_cores,p=p,q=q)
       if is_walk_viz:
        dict_path = path.replace(".gpickle","") + "_frac.pkl"
        print(dict_path)
        get_walk_plots(node2vec.walks, G,t,dict_path)
       
       model = node2vec.fit() 
       emb_df = (pd.DataFrame([model.wv.get_vector(str(n)) for n in G.nodes()], index = G.nodes))
    elif model == "custom_n2v":
       print("Going in Custom N2V")
       node2vec = custom_N2V(G, dimensions=DIM, walk_length=WALK_LEN, num_walks=NUM_WALKS, workers=num_cores)
       model = node2vec.fit() 
       emb_df = (pd.DataFrame([model.wv.get_vector(str(n)) for n in G.nodes()], index = G.nodes))
    elif model == "fw":
        print("[FW] Using p value: {}, Using q value : {}".format(p,q))
        fw_model = FairWalk(G, dimensions=DIM, walk_length=WALK_LEN, num_walks=NUM_WALKS, workers=num_cores,p=p,q=q)
        if is_walk_viz:
            dict_path = path.replace(".gpickle","") + "_frac.pkl"
            print(dict_path)
            get_walk_plots(fw_model.walks, G,t,dict_path)
        model = fw_model.fit() 
        emb_df = (pd.DataFrame([model.wv.get_vector(str(n)) for n in G.nodes()], index = G.nodes))
    elif model == "cw":
        print("[CW] Using p value: {}, Using q value : {}".format(p,q))
        fw_model = CustomWalk(G, dimensions=DIM, walk_length=WALK_LEN, num_walks=NUM_WALKS, workers=num_cores,p=p,q=q)
        if is_walk_viz:
            dict_path = path.replace(".gpickle","") + "_frac.pkl"
            print(dict_path)
            get_walk_plots(fw_model.walks, G,t,dict_path)
        model = fw_model.fit() 
        emb_df = (pd.DataFrame([model.wv.get_vector(str(n)) for n in G.nodes()], index = G.nodes))
    elif model == "cnw":
        print("[CNW] Using p value: {}, Using q value : {}".format(p,q))
        cnw_model = common_N2V(G, dimensions=DIM, walk_length=WALK_LEN, num_walks=NUM_WALKS, workers=num_cores,p=p,q=q)
        if is_walk_viz:
            dict_path = path.replace(".gpickle","") + "_frac.pkl"
            print(dict_path)
            get_walk_plots(cnw_model.walks, G,t,dict_path)
        model = cnw_model.fit() 
        emb_df = (pd.DataFrame([model.wv.get_vector(str(n)) for n in G.nodes()], index = G.nodes))
    return model, emb_df


def get_top_recos(g, embeddings, u, N=1):
    all_nodes = g.nodes()
    df = embeddings
    results = []
    for src_node in u:
        source_emb = df[df.index == src_node]
        other_nodes = [n for n in all_nodes if n not in list(g.adj[src_node]) + [src_node]]
        other_embs = df[df.index.isin(other_nodes)]

        sim = cosine_similarity(source_emb, other_embs)[0].tolist()
        idx = other_embs.index.tolist()

        idx_sim = dict(zip(idx, sim))
        idx_sim = sorted(idx_sim.items(), key=lambda x: x[1], reverse=True)
        
        similar_nodes = idx_sim[:N]
        v = [tgt[0] for tgt in similar_nodes][0]
        results.append((src_node,v))
       
    return results 


def get_walk_plots(walks, g, t,dict_path):
    try:
        print("get trace for t = ", t)
        node_to_walk_dict = {}
        len_majority_nodes  = len([node for node in g.nodes() if g.nodes[node]["m"] == 0])
        len_minority_nodes = len(g.nodes()) - len_majority_nodes
        
        for sub_walk in walks:
         
            sub_walk = [int(node) for node in sub_walk]
            src_node = sub_walk[0]
            if src_node not in node_to_walk_dict:
                node_to_walk_dict[src_node] = list(set(sub_walk))
            else:
                new_list = list(set(node_to_walk_dict[src_node]+sub_walk))
                node_to_walk_dict[src_node] = new_list
        
        maj_frac, min_frac = 0, 0
       
    
        for _, walk in node_to_walk_dict.items():
           
            majority_nodes_visited = len([_ for node in walk if g.nodes[node]["m"] == 0])
            minority_nodes_visited = len([_ for node in walk if g.nodes[node]["m"] == 1])
            maj_frac += round(majority_nodes_visited/len_majority_nodes,2)
            min_frac += round(minority_nodes_visited/len_minority_nodes,2)
            
        avg_maj_frac = round(maj_frac/len(g.nodes()),2)
        avg_min_frac = round(min_frac/len(g.nodes()),2)
        print("~~~~~~~~~~~ maj frac: {}, min frac:{}".format(avg_maj_frac, avg_min_frac))
        if os.path.exists(dict_path):
             with open(dict_path, 'rb') as f:
                dict_ = pkl.load(f)
        else:
            dict_ = {}

        # also compute avg betweenness centrality
        centrality_dict = nx.betweenness_centrality(g, normalized=True)
        minority_centrality = [val for node, val in centrality_dict.items() if g.nodes[node]["m"] == 1]
        avg_val = np.mean(minority_centrality)    
        
        dict_[t] = {"maj":avg_maj_frac, "min": avg_min_frac,"bet":avg_val}
        print("!! comes here to write")
        os.makedirs(os.path.dirname(dict_path), exist_ok=True)
        with open(dict_path, 'wb') as f:
           pkl.dump(dict_, f)
       
    except Exception as error:
        print("Error in get walk plots: ", error)