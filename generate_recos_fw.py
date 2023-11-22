import os
import numpy as np
import networkx as nx
from fairwalk import FairWalk
from tqdm import tqdm
import random
import time
import argparse
from fast_pagerank import pagerank_power

from org.gesis.lib import io
from org.gesis.lib.io import create_subfolders
from org.gesis.lib.graph import get_node_metadata_as_dataframe
from org.gesis.lib.io import save_csv
from org.gesis.lib.graph import get_circle_of_trust_per_node

from joblib import delayed
from joblib import Parallel
from collections import Counter



DPAH_path = "../Homophilic_Directed_ScaleFree_Networks/DPAH"
# get all files (each files corresponds to one configuration of hmm & hMM)
all_files = os.listdir(DPAH_path)
graph_files = [file_name for file_name in all_files if ".gpickle" in file_name]

MAIN_SEED = 42
EPOCHS = 30
MODEL = "fw"

fm = 0.3
N = 1000
YM, Ym = 2.5, 2.5
d = 0.03

# Hyperparameter for node2vec
DIM = 64
WALK_LEN = 10
NUM_WALKS = 200

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def recommender_model(G, num_cores=8):
    fw_model = FairWalk(G, dimensions=DIM, walk_length=WALK_LEN, num_walks=NUM_WALKS, workers=num_cores)
    model = fw_model.fit() 
    return model

def get_top_recos(G, model):
    results = []
    for ns in G.nodes():
        nt = int(model.wv.most_similar(str(ns))[0][0]) # generate top 1 recommendation
        results.append((ns,nt))
    return results
   

def rewiring_list(G, node, number_of_rewiring):
        nodes_to_be_unfollowed = []
        node_neighbors = np.array(list(G.successors(node)))
        nodes_to_be_unfollowed = np.random.permutation(node_neighbors)[:number_of_rewiring]
        return list(map(lambda x: tuple([node, x]), nodes_to_be_unfollowed))

def make_one_timestep(G, seed):
        '''Defines each timestep of the simulation:
            0. each node makes experiments
            1. loops in the permutation of nodes choosing the INFLUENCED node u (u->v means u follows v, v can influence u)
            2. loops s (number of interactions times)
            3. choose existing links with 1-a prob, else recommends
                4. if recommendes: invokes recommend_nodes() to choose the influencers nodes that are not already linked u->v

        '''
                              
        # set seed
        set_seed(seed)

        n2v_model = recommender_model(G)
        recos = get_top_recos(G, n2v_model) 

        for i,(u,v) in enumerate(recos):
            seed += i
            set_seed(seed)
            edges_to_be_removed = rewiring_list(G, u, 1)
            G.remove_edges_from(edges_to_be_removed) # deleting previously existing links
            G.add_edge(u,v)
            seed += 1
        return G


def run(hMM, hmm):
        
        # Setting seed
        np.random.seed(MAIN_SEED)
        random.seed(MAIN_SEED)
        folder_path = "../Homophilic_Directed_ScaleFree_Networks/{}".format(MODEL)
        new_filename = get_filename(MODEL, N, fm, d, YM, Ym, hMM, hmm) +".gpickle"
        new_path = os.path.join(folder_path, new_filename) 
        if os.path.exists(new_path):
           print("File exists for configuration hMM:{}, hmm:{}".format(hMM,hmm))
           return 
        print("hMM: {}, hmm: {}".format(hMM, hmm))

        # read the base graph from DPAH folder
        old_filename = "DPAH-N" + new_filename.replace(".gpickle","").split("N")[-1] + "-ID0.gpickle"
        g = nx.read_gpickle(os.path.join(DPAH_path,old_filename))

        node2group = {node:g.nodes[node]["m"] for node in g.nodes()}
        nx.set_node_attributes(g, node2group, 'group')
   
        iterable = tqdm(range(EPOCHS), desc='Timesteps', leave=True) 
        time = 0
        for time in iterable:
            seed = MAIN_SEED+time+1 
            g_updated = make_one_timestep(g.copy(), seed)
            g = g_updated
        
            if time == EPOCHS-1:
                save_metadata(g, hMM, hmm, MODEL)

def get_filename(model,N,fm,d,YM,Ym,hMM,hmm):
    return "{}-N{}-fm{}{}{}{}{}{}".format(model, N, 
                                             round(fm,1), 
                                             '-d{}'.format(round(d,5)), 
                                             '-ploM{}'.format(round(YM,1)), 
                                             '-plom{}'.format(round(Ym,1)), 
                                             '-hMM{}'.format(hMM),
                                             '-hmm{}'.format(hmm))

def save_metadata(g, hMM, hmm, model):
    folder_path = "../Homophilic_Directed_ScaleFree_Networks/{}".format(model)
    create_subfolders(folder_path)
    filename = get_filename(model, N, fm, d, YM, Ym, hMM, hmm)
    
    fn = os.path.join(folder_path,'{}.gpickle'.format(filename))
    io.save_gpickle(g, fn)

    ## [Personal] Specifying jobs
    njobs = 24
    df = get_node_metadata_as_dataframe(g, njobs=njobs)
    csv_fn = os.path.join(folder_path,'{}.csv'.format(filename))
    io.save_csv(df, csv_fn)
    
    print("Saving graph and csv file at, ", filename)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--hMM", help="homophily between Majorities", type=float, default=0.5)
    # parser.add_argument("--hmm", help="homophily between minorities", type=float, default=0.5)
    parser.add_argument("--start", help="homophily between Majorities", type=float, default=0.1)
    parser.add_argument("--end", help="homophily between minorities", type=float, default=0.5)
    args = parser.parse_args()
    
    start_time = time.time()
    start_idx, end_idx = args.start, args.end
    # run(args.hMM, args.hmm)
    print("STARTING IDX", start_idx, ", END IDX", end_idx)
    num_cores = 36
    [Parallel(n_jobs=num_cores)(delayed(run)(np.round(hMM,2), np.round(hmm,2)) for hMM in np.arange(start_idx, end_idx, 0.1) for hmm in np.arange(0.0,1.1,0.1))]
    # import numpy as np
    # for hMM in np.arange(0.0, 1.1, 0.1):
    #     for hmm in np.arange(0.0,1.1,0.1):
    #         hMM, hmm = np.round(hMM, 2), np.round(hmm, 2)
    #         run(hMM, hmm)

    print("--- %s seconds ---" % (time.time() - start_time))
        