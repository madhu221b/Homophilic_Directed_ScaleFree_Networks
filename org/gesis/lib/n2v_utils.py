import random
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from fairwalk.fairwalk  import FairWalk
from node2vec import Node2Vec
from node2vec_code.node2vec.node2vec import Node2Vec as custom_N2V
# Hyperparameter for node2vec/fairwalk
DIM = 64
WALK_LEN = 10
NUM_WALKS = 200

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def rewiring_list(G, node, number_of_rewiring):
        nodes_to_be_unfollowed = []
        node_neighbors = np.array(list(G.successors(node)))
        nodes_to_be_unfollowed = np.random.permutation(node_neighbors)[:number_of_rewiring]
        return list(map(lambda x: tuple([node, x]), nodes_to_be_unfollowed))


def recommender_model(G,model="n2v",q=1,num_cores=8):
    if model == "n2v":
       print("Using q value: ",q)
       node2vec = Node2Vec(G, dimensions=DIM, walk_length=WALK_LEN, num_walks=NUM_WALKS, workers=num_cores,q=q)
       model = node2vec.fit() 
       emb_df = (pd.DataFrame([model.wv.get_vector(str(n)) for n in G.nodes()], index = G.nodes))
    elif model == "custom_n2v":
       print("Going in Custom N2V")
       node2vec = custom_N2V(G, dimensions=DIM, walk_length=WALK_LEN, num_walks=NUM_WALKS, workers=num_cores)
       model = node2vec.fit() 
       emb_df = (pd.DataFrame([model.wv.get_vector(str(n)) for n in G.nodes()], index = G.nodes))
    elif model == "fw":
        fw_model = FairWalk(G, dimensions=DIM, walk_length=WALK_LEN, num_walks=NUM_WALKS, workers=num_cores)
        model = fw_model.fit() 
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