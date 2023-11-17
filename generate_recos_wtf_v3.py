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

from joblib import delayed
from joblib import Parallel
from collections import Counter



DPAH_path = "../Homophilic_Directed_ScaleFree_Networks/DPAH"
# get all files (each files corresponds to one configuration of hmm & hMM)
all_files = os.listdir(DPAH_path)
graph_files = [file_name for file_name in all_files if ".gpickle" in file_name]

MAIN_SEED = 42
EPOCHS = 30
MODEL = "wtf"

fm = 0.3
N = 1000
YM, Ym = 2.5, 2.5
d = 0.03

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def twitter_wtf(network, sparse_adj, node_id, k_for_circle_of_trust=20, tol=1e-8,
                damping_factor=.85, k_for_recommendation=-1):
    """This method aims to realize a link prediction algorithm used by Twitter to perform
        the WTF recommendation on the platform.
        The algorithm can be seen at 'https://web.stanford.edu/~rezab/papers/wtf_overview.pdf'.

        The algorithm consists of two phases:
            1) Compute the circle of trust for the user you want to recommend(top-k nodes in PPR)
            2) Compute the top-k nodes using score propagation
    """
    k_for_circle_of_trust = int(network.number_of_nodes()*.1)
    #1st phase: Compute circle of trust of user according to Personalized PageRank
    personalize = np.zeros(shape=network.number_of_nodes())
    personalize[node_id] = 1
    values_of_personalized_pr = pagerank_power(sparse_adj, p=damping_factor, personalize=personalize, tol=1e-6)
    circle_of_trust = values_of_personalized_pr.argsort()[-k_for_circle_of_trust:][::-1]

    #2nd phase: init bipartite graph
    bipartite_graph = nx.DiGraph()
    #add nodes belonging to the circle of trust as hubs(H)
    for node in circle_of_trust:
        #these nodes are "hubs"(H) in the bipartite graph
        bipartite_graph.add_node(str(node)+"H")
    #add out neighbors of nodes belonging to the circle of trust as authorities(A)
    for node in circle_of_trust:
        for out_neighbor in network.neighbors(node):
            #direction is inverted for a matter of simplicity in the sequent phases
            bipartite_graph.add_edge(str(out_neighbor)+"A", str(node)+"H")

    #retrieve adjacency matrix of bipartite graph
    A = nx.to_numpy_array(bipartite_graph)

    #retrieve list of all nodes splitted by authority or hub
    all_nodes = list(bipartite_graph.nodes())
    hub_nodes = [int(x[:-1]) for x in all_nodes if 'H' in x]
    authority_nodes = [int(x[:-1]) for x in all_nodes if 'A' in x]

    #3rd phase: start building ingredients of our SALSA algorithm
    #these are the transition matrices determined by the bipartite graph
    S_prime = A[len(hub_nodes):, :][:, :len(hub_nodes)].copy()
    R_prime = S_prime.T.copy()
    #normalize both matrices
    denominator_S_prime = S_prime.sum(axis=0)
    denominator_S_prime[denominator_S_prime == 0] = 1
    S_prime = S_prime / denominator_S_prime
    denominator_R_prime = R_prime.sum(axis=0)
    denominator_R_prime[denominator_R_prime == 0] = 1
    R_prime = R_prime / denominator_R_prime
    #these are the vectors which contain the score of similarity
    #and relevance
    s = np.zeros(shape=(len(hub_nodes), 1), dtype=np.float)
    r = np.zeros(shape=(len(authority_nodes), 1), dtype=np.float)

    #at the beginning of the procedure we put the similarity
    #of the user we want to give the recommendation equal to 1
    index_of_node_to_recommend = np.where(circle_of_trust == node_id)[0][0]
    s[index_of_node_to_recommend] = 1.

    #init damping vector
    alpha = 1 - damping_factor
    alpha_vector = np.zeros(shape=(len(hub_nodes), 1), dtype=np.float)
    alpha_vector[index_of_node_to_recommend] = alpha

    #4th phase: run the algorithm
    convergence = False
    while not convergence:
        s_ = s.copy()
        r_ = r.copy()
        r_ = S_prime.dot(s)
        s_ = alpha_vector + (1 - alpha)*(R_prime.dot(r))
        #compute difference and check if convergence has been reached
        diff = abs(s_ - s)
        if np.linalg.norm(diff) < tol:
            convergence=True
        #update real vectors
        s = s_
        r = r_

    #5th phase: order by score and delete neighbors of node to be recommended
    #of course we don't want to recommend people that the user already follow
    neighbors_to_not_recommend = nx.neighbors(network, node_id)
    relevance_scores = r.flatten()
    if k_for_recommendation == -1:
        k_for_recommendation = 0 #Take all the nodes!

    neighbors_to_not_recommend = set(neighbors_to_not_recommend)
    results = []
    for node in relevance_scores.argsort()[::-1]:
        if node not in neighbors_to_not_recommend and node != node_id:
            results.append(((node_id, node, relevance_scores[node])))
            if len(results) == k_for_recommendation:
                break
    return results

def recommender_sys(u, G, num_cores=36):
    A = nx.to_scipy_sparse_matrix(G)
    # top_k_edges = Parallel(n_jobs=num_cores)(delayed(twitter_wtf)(network=G, sparse_adj=A,
    #                                   node_id=u, damping_factor=0.85, k_for_circle_of_trust=10,k_for_recommendation=1))
    top_k_edges = twitter_wtf(network=G, sparse_adj=A,
                                      node_id=u, damping_factor=0.85, k_for_circle_of_trust=10,k_for_recommendation=1)
    return top_k_edges

def recommend_nodes(u, G):
        '''Recommends influencers (list of 1) from non-existing successors:
            uniform or invokes recommender systems
        '''
 
        top_k_edges = recommender_sys(u,G)
        return top_k_edges

  
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
        A = nx.to_scipy_sparse_matrix(G)
        recos = get_top_recos(A) 
        recos = [ele for ele in recos if len(ele)!=0]
        if len(recos) == 0: return G # no recos generated for 1.0 1.0
        
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
        return (node_index,[n for n, pev in centrality.most_common(1)][0])
    except Exception as e:
        return []

# def run(hMM, hmm):
#     folder_path = "../Homophilic_Directed_ScaleFree_Networks/{}".format(MODEL)
    


#     new_filename = get_filename(MODEL, N, fm, d, YM, Ym, hMM, hmm) +".gpickle"
#     new_path = os.path.join(folder_path, new_filename) 
#     print(new_path)
#     if os.path.exists(new_path):
#         print("File exists for configuration hMM:{}, hmm:{}".format(hMM,hmm))
#         return 
#     print("hMM: {}, hmm: {}".format(hMM, hmm))

#     # read the base graph from DPAH folder
#     old_filename = "DPAH-N" + new_filename.replace(".gpickle","").split("N")[-1] + "-ID0.gpickle"
#     g = nx.read_gpickle(os.path.join(DPAH_path,old_filename))
    

#     for t in range(EPOCHS):
#         print("Generating recommendations for epoch step:", t)
#         nodes = g.nodes()
#         A = nx.to_scipy_sparse_matrix(g,nodes)
#         recos = get_top_recos(A) 
#         for src_node, target_node in enumerate(recos):
            
#             # if no edge already exists
#             if not g.has_edge(src_node, target_node):
#                 # chose a random edge to be removed of src node
#                 chosen_edge = random.choice(list(g.out_edges(src_node)))
#                 g.remove_edge(*chosen_edge)
#                 # add the new out edge
#                 g.add_edge(src_node, target_node)
#         # save all metadata on last epoch
#         if t == EPOCHS-1:
#             save_metadata(g, hMM, hmm, MODEL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hMM", help="homophily between Majorities", type=float, default=0.5)
    parser.add_argument("--hmm", help="homophily between minorities", type=float, default=0.5)
    args = parser.parse_args()
    
    start_time = time.time()
    run(args.hMM, args.hmm)
    # num_cores = 36
    # [Parallel(n_jobs=num_cores)(delayed(run)(np.round(hMM,2), np.round(hmm,2)) for hMM in np.arange(0.0, 1.1, 0.1) for hmm in np.arange(0.0,1.1,0.1))]
    # # import numpy as np
    # # for hMM in np.arange(0.0, 1.1, 0.1):
    # #     for hmm in np.arange(0.0,1.1,0.1):
    # #         hMM, hmm = np.round(hMM, 2), np.round(hmm, 2)
    # #         run(hMM, hmm)

    print("--- %s seconds ---" % (time.time() - start_time))
        