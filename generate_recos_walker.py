import os
import numpy as np
import networkx as nx
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
from org.gesis.lib.n2v_utils import set_seed, rewiring_list, recommender_model_walker, get_top_recos
from joblib import delayed
from joblib import Parallel
from collections import Counter



DPAH_path = "../Homophilic_Directed_ScaleFree_Networks/DPAH"
# get all files (each files corresponds to one configuration of hmm & hMM)
all_files = os.listdir(DPAH_path)
graph_files = [file_name for file_name in all_files if ".gpickle" in file_name]

MAIN_SEED = 42
EPOCHS = 30


fm = 0.3
N = 1000
YM, Ym = 2.5, 2.5
d = 0.03

   
def make_one_timestep(g, seed,t=0,path="",model="",extra_params=dict()):
        '''Defines each timestep of the simulation:
            0. each node makes experiments
            1. loops in the permutation of nodes choosing the INFLUENCED node u (u->v means u follows v, v can influence u)
            2. loops s (number of interactions times)
            3. choose existing links with 1-a prob, else recommends
                4. if recommendes: invokes recommend_nodes() to choose the influencers nodes that are not already linked u->v

        '''
                              
        # set seed
        set_seed(seed)

        print("Generating Node Embeddings")
        _, embeds = recommender_model_walker(g,t,path,model=model,extra_params=extra_params)
        print("Getting Link Recommendations from {} Model".format(model))
        u = g.nodes()
        recos = get_top_recos(g,embeds, u) 
        new_edges = 0
        for i,(u,v) in enumerate(recos):
            seed += i
            set_seed(seed)
            if not g.has_edge(u,v):
               edges_to_be_removed = rewiring_list(g, u, 1)
               g.remove_edges_from(edges_to_be_removed) # deleting previously existing links
               new_edges += 1
               g.add_edge(u,v)
            seed += 1
        print("No of new edges added: ", new_edges)
        return g


def run(hMM, hmm,model,extra_params):
    try:  
        # Setting seed
        np.random.seed(MAIN_SEED)
        random.seed(MAIN_SEED)
        folder_path = "../Homophilic_Directed_ScaleFree_Networks/{}".format(model)
        new_filename = get_filename(model, N, fm, d, YM, Ym, hMM, hmm) +".gpickle"
        new_path = os.path.join(folder_path, new_filename) 
        if os.path.exists(new_path) and False: # disabling this condition
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
            is_file, g_obj =  is_file_exists(hMM,hmm,model,time)
            if not is_file:
                print("File does not exist for time {}, creating now".format(time))
                seed = MAIN_SEED+time+1 
                g_updated = make_one_timestep(g.copy(),seed,time,new_path,model,extra_params)
                g = g_updated
                save_metadata(g, hMM, hmm, model,t=time)
            else:
                print("File exists for time {}, loading it... ".format(time))
                g = g_obj

                # if time == EPOCHS-1:
            
    except Exception as e:
         print("Error in run : ", e)


def is_file_exists(hMM, hmm, model,t):
    folder_path = "../Homophilic_Directed_ScaleFree_Networks/{}".format(model)
    filename = get_filename(model, N, fm, d, YM, Ym, hMM, hmm)
    fn = os.path.join(folder_path,'{}_t_{}.gpickle'.format(filename,t))
    if os.path.exists(fn):
        return True, nx.read_gpickle(fn)
    else:
        return False, None
    

def get_filename(model,N,fm,d,YM,Ym,hMM,hmm):
    return "{}-N{}-fm{}{}{}{}{}{}".format(model, N, 
                                             round(fm,1), 
                                             '-d{}'.format(round(d,5)), 
                                             '-ploM{}'.format(round(YM,1)), 
                                             '-plom{}'.format(round(Ym,1)), 
                                             '-hMM{}'.format(hMM),
                                             '-hmm{}'.format(hmm))

def save_metadata(g, hMM, hmm, model,t=0):
    folder_path = "../Homophilic_Directed_ScaleFree_Networks/{}".format(model)
    create_subfolders(folder_path)
    filename = get_filename(model, N, fm, d, YM, Ym, hMM, hmm)
    
    
    fn = os.path.join(folder_path,'{}_t_{}.gpickle'.format(filename,t))
    io.save_gpickle(g, fn)

    ## [Personal] Specifying jobs
    njobs = 24
    if t == EPOCHS - 1:
        df = get_node_metadata_as_dataframe(g, njobs=njobs)
        csv_fn = os.path.join(folder_path,'{}_t_{}.csv'.format(filename,t))
        io.save_csv(df, csv_fn)
    
    print("Saving graph and csv file at, ", fn.replace(".gpickle",""))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hMM", help="homophily between Majorities", type=float, default=0.5)
    parser.add_argument("--hmm", help="homophily between minorities", type=float, default=0.5)
    parser.add_argument("--model", help="Different Walker Models", type=str)
    parser.add_argument("--beta", help="Beta paramater", type=float, default=1.0)
    parser.add_argument("--alpha", help="Alpha paramater (Levy)", type=float, default=1.0)

    parser.add_argument("--start", help="Start idx", type=float, default=0.1)
    parser.add_argument("--end", help="End idx", type=float, default=0.5)
    args = parser.parse_args()
    
    start_time = time.time()
    extra_params = dict()
    if args.model == "commonngh":
        model = args.model
    elif args.model == "levy":
       model =  "{}_alpha_{}".format(args.model,args.alpha)
       extra_params = {"alpha":args.alpha}
    else:
       model =  "{}_beta_{}".format(args.model,args.beta)
       extra_params = {"beta":args.beta}
    # run(args.hMM, args.hmm, model=model, extra_params=extra_params)

    start_idx, end_idx = args.start, args.end
    print("STARTING IDX", start_idx, ", END IDX", end_idx)
    num_cores = 36
    [Parallel(n_jobs=num_cores)(delayed(run)(np.round(hMM,2), np.round(hmm,2), model=model, extra_params=extra_params) for hMM in np.arange(start_idx, end_idx, 0.1) for hmm in np.arange(0.0,1.1,0.1))]


    print("--- %s seconds ---" % (time.time() - start_time))
        