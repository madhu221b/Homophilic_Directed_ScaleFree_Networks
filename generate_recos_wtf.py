import os
import numpy as np
import seaborn as sns
import networkx as nx
import powerlaw
import random
import time
import argparse
from palettable.cartocolors.diverging import Geyser_7
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

EPOCHS = 30
MODEL = "wtf"

fm = 0.3
N = 1000
YM, Ym = 2.5, 2.5
d = 0.03

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


def get_top_recos(A, p=0.85, top=10, num_cores=36):
    cot_per_node = get_circle_of_trust_per_node(A, p, top, num_cores)
    results = Parallel(n_jobs=num_cores)(delayed(_salsa_top1)(node_index, cot, A, top) for node_index, cot in enumerate(cot_per_node))
    return results
    
def _salsa_top1(node_index, cot, A, top=10):
    try:
        BG = nx.Graph()
        BG.add_nodes_from(['h{}'.format(vi) for vi in cot], bipartite=0)  # hubs
        edges = [('h{}'.format(vi), int(vj)) for vi in cot for vj in np.argwhere(A[vi,:] != 0 )[:,1]]
        BG.add_nodes_from(set([e[1] for e in edges]), bipartite=1)  # authorities
        BG.add_edges_from(edges)
        centrality = Counter({n: c for n, c in nx.eigenvector_centrality_numpy(BG).items() if type(n) == int
                                                                                        and n not in cot
                                                                                        and n != node_index
                                                                                        and n not in np.argwhere(A[node_index,:] != 0 )[:,1] })
        del(BG)
        #time.sleep(0.01)
        return [n for n, pev in centrality.most_common(1)][0]
    except Exception as e:
        return []

def run(hMM, hmm):
    folder_path = "../Homophilic_Directed_ScaleFree_Networks/{}".format(MODEL)
    


    new_filename = get_filename(MODEL, N, fm, d, YM, Ym, hMM, hmm) +".gpickle"
    new_path = os.path.join(folder_path, new_filename) 
    print(new_path)
    if os.path.exists(new_path):
        print("File exists for configuration hMM:{}, hmm:{}".format(hMM,hmm))
        return 
    print("hMM: {}, hmm: {}".format(hMM, hmm))

    # read the base graph from DPAH folder
    old_filename = "DPAH-N" + new_filename.replace(".gpickle","").split("N")[-1] + "-ID0.gpickle"
    g = nx.read_gpickle(os.path.join(DPAH_path,old_filename))
    

    for t in range(EPOCHS):
        print("Generating recommendations for epoch step:", t)
        nodes = g.nodes()
        A = nx.to_scipy_sparse_matrix(g,nodes)
        recos = get_top_recos(A) 
        for src_node, target_node in enumerate(recos):
            
            # if no edge already exists
            if not g.has_edge(src_node, target_node):
                # chose a random edge to be removed of src node
                chosen_edge = random.choice(list(g.out_edges(src_node)))
                g.remove_edge(*chosen_edge)
                # add the new out edge
                g.add_edge(src_node, target_node)
        # save all metadata on last epoch
        if t == EPOCHS-1:
            save_metadata(g, hMM, hmm, MODEL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hMM", help="homophily between Majorities", type=float, default=0.5)
    parser.add_argument("--hmm", help="homophily between minorities", type=float, default=0.5)
    args = parser.parse_args()
    
    start_time = time.time()
    # run(args.hMM, args.hmm)
    import numpy as np
    for hMM in np.arange(0.0, 1.1, 0.1):
        for hmm in np.arange(0.0,1.1,0.1):
            hMM, hmm = np.round(hMM, 2), np.round(hmm, 2)
            run(hMM, hmm)

    print("--- %s seconds ---" % (time.time() - start_time))
        