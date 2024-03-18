import os
import pickle as pkl
import networkx as nx
from sklearn.manifold import TSNE
from gensim import models

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from org.gesis.lib.n2v_utils import recommender_model_walker, recommender_model

DIM = 64
WALK_LEN = 10
NUM_WALKS = 200

def get_embeddings(g, model,embedding_file_name):
    if "fw" in model:
        p = q = 1.0
        model, embeds = recommender_model(g,model="fw",p=p,q=q)
    elif "fairindegree" in model:
        model, embeds = recommender_model_walker(g,model="fairindegreev2")
    elif "nonlocalindegree" in model:
        extra_params = {"beta":2.0,"alpha":1.0}
        model, embeds = recommender_model_walker(g,model="nonlocalindegree",extra_params=extra_params)
    else:
        extra_params = {"beta":2.0}
        model, embeds = recommender_model_walker(g,model="indegree",extra_params=extra_params)
   
    
    print("Saving embeddings at file name:", embedding_file_name)
    with open(embedding_file_name, 'wb') as f:
        pkl.dump(embeds, f)
    return embeds

def plot_embeddings(filename,hMM,hmm,t):
    g = nx.read_gpickle(filename)
    node2group = {node:g.nodes[node]["m"] for node in g.nodes()}
    nx.set_node_attributes(g, node2group, 'group')

    if "fw" in filename: model = "fw"
    elif "fairindegreev2" in filename: model = "fairindegreev2"
    elif "nonlocalindegree" in filename: model = "nonlocalindegree"
    else: model = "indegree"
    
    embedding_folder = "./embedding/{}".format(model)
    if not os.path.exists(embedding_folder): os.makedirs(embedding_folder)
    embedding_file_name = embedding_folder + "/_hMM{}_hmm{}_t_{}.pkl".format(hMM,hmm,t)
    print("Embedding file name: ", embedding_file_name)
    if not os.path.exists(embedding_file_name):
        embeds = get_embeddings(g, model, embedding_file_name)
    else:
        with open(embedding_file_name, 'rb') as f:
                embeds = pkl.load(f)

    print("embedding shape: ", embeds.shape)
    # tsne = TSNE(n_components=2, random_state=42)
    # embeddings2d = tsne.fit_transform(embeds)
    # x, y = embeddings2d[:,0], embeddings2d[:,1]

    # print("Plotting tsne embeddings")
    # fig, ax = plt.subplots(nrows=1, ncols=1) 
    # ax.scatter(x, y, alpha=.1)

    # fig.savefig('trial_emb.png',bbox_inches='tight')   # save the figure to file
    # plt.close(fig)  
    




if __name__ == "__main__":
    main_directory = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/"

    hMM, hmm = 0.8, 0.2
    for t in [0,15,28]:
       # file_name = main_directory+"fw_p_1.0_q_1.0_fm_0.3/fw_p_1.0_q_1.0-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_{}.gpickle".format(hMM,hmm,t)
       # file_name = main_directory+"indegree_beta_2.0_fm_0.3/indegree_beta_2.0-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_{}.gpickle".format(hMM,hmm,t)
       file_name = main_directory+"nonlocalindegree_alpha_1.0_beta_2.0_fm_0.3/nonlocalindegree_alpha_1.0_beta_2.0-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_{}.gpickle".format(hMM,hmm,t)
       plot_embeddings(file_name,hMM,hmm,t)