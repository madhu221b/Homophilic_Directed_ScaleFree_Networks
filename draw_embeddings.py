import os
import pickle as pkl
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from gensim import models

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from org.gesis.lib.n2v_utils import read_graph

DIM = 64
WALK_LEN = 10
NUM_WALKS = 200


def get_metadata_df(g, cent_dict):
    df_cent = pd.DataFrame.from_dict(cent_dict, orient='index', columns=['betn'])

    node_attr = nx.get_node_attributes(g, "group")
    df_identity = pd.DataFrame.from_dict(node_attr, orient='index', columns=['group'])

    in_degree = dict(g.in_degree())
    df_degree = pd.DataFrame.from_dict(in_degree, orient='index', columns=['indegree'])
    df_all = pd.concat([df_cent, df_identity, df_degree], axis=1)
    return df_all 
    

def plot_embeddings(model,filename,ds,t):
    print(filename)
    g = nx.read_gpickle(filename)
    
    scale_degree_size = 10000
    fig, ax = plt.subplots(nrows=1, ncols=1) 
    
     
    embedding_folder = "./embedding/{}".format(model)
    if not os.path.exists(embedding_folder): os.makedirs(embedding_folder)
    embedding_file_name = embedding_folder + "/_{}_t_{}.pkl".format(ds, t)

    print("!! Embedding file name: ", embedding_file_name)
    with open(embedding_file_name, 'rb') as f:
                embeds = pkl.load(f) # node idx & 64 embeds
 
    dict_path = '/home/mpawar/Homophilic_Directed_ScaleFree_Networks/model_{}_name_{}/seed_42/_{}-name_{}_t_{}.pkl'.format(model,ds,model,ds,t)
    print("!! dict path: ",dict_path)
    if os.path.exists(dict_path):
        with open(dict_path, "rb") as f:
            cent_dict = pkl.load(f)
    else:
        cent_dict = nx.betweenness_centrality(g,normalized=True)
    embeds_data = get_metadata_df(g, cent_dict)
    print(embeds_data.shape)

    tsne = TSNE(n_components=2, random_state=42)
    embeddings2d = tsne.fit_transform(embeds)

    df_embeds = pd.DataFrame({'x': embeddings2d[:,0], 'y': embeddings2d[:, 1]}, index=embeds.index)
    df_all = pd.concat([df_embeds,embeds_data],axis=1)

    ## Coloring the points by betn
    cmap =  "OrRd"
    norm = plt.Normalize(0, 0.004)
    print("Plotting tsne embeddings")
    if ds == "facebook_locale":
         df_1, lbl_1 = df_all[df_all['group'] == 126], 126
         df_0, lbl_0 = df_all[df_all['group'] == 127], 127
    else:
        df_1, lbl_1 = df_all[df_all['group'] == 1], 1
        df_0, lbl_0 = df_all[df_all['group'] == 0], 0
        df_2, lbl_2 = df_all[df_all['group'] == 2], 2
        

    df_0_size, df_1_size = df_0["betn"] * scale_degree_size + 15,  df_1["betn"] * scale_degree_size + 15
    

    ax.scatter(df_0["x"], df_0["y"],  s=df_0_size, marker="^",alpha=0.6,label=lbl_0,color="orange")
    ax.scatter(df_1["x"], df_1["y"], s=df_1_size, alpha=0.6, marker="o",label=lbl_1,color="blue")
    
    if ds == "facebook_syn_3":
       df_2_size = df_2["betn"]*scale_degree_size + 15
       ax.scatter(df_2["x"], df_2["y"], s=df_2_size, alpha=0.6, marker="x",label=lbl_2,color="green")

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    # ax.set_xlim(-60,60)
    # ax.set_ylim(-60,60)
    
    if ds == "rice": llim, uplim = -35,35
    elif ds == "facebook_syn_3": llim, uplim = -90,90
    elif ds == "facebook_locale": llim, uplim = -80,80
    else: llim, uplim = -100,100
    ax.set_xlim(llim,uplim)
    ax.set_ylim(llim,uplim)
    ax.legend(loc="upper right")
    fig.savefig('tsne_emb_model_{}_ds{}_t_{}.png'.format(model,ds,t),bbox_inches='tight')   # save the figure to file
    plt.close(fig)  
    

def plot_embeddings_bet(model,filename,ds,t):
    print("filename of graph: ", filename)
    g = nx.read_gpickle(filename)
    print("g.nodes: ", len(g.nodes()))
    
    scale_degree_size = 10000
    fig, ax = plt.subplots(nrows=1, ncols=1) 
    
     
    embedding_folder = "./embedding/{}".format(model)
    if not os.path.exists(embedding_folder): os.makedirs(embedding_folder)
    embedding_file_name = embedding_folder + "/_{}_t_{}.pkl".format(ds, t)

    print("Embedding file name: ", embedding_file_name)
    with open(embedding_file_name, 'rb') as f:
                embeds = pkl.load(f) # node idx & 64 embeds

    cent_dict = nx.betweenness_centrality(g, normalized=True)
    embeds_data = get_metadata_df(g, cent_dict)
    print(embeds_data.shape)
    

    tsne = TSNE(n_components=2, random_state=42)
    embeddings2d = tsne.fit_transform(embeds)

    df_embeds = pd.DataFrame({'x': embeddings2d[:,0], 'y': embeddings2d[:, 1]}, index=embeds.index)
    df_all = pd.concat([df_embeds,embeds_data],axis=1)

    ## Coloring the points by betn
    cmap =  "OrRd"
    norm = plt.Normalize(0, 0.004)
    print("Plotting tsne embeddings")

    df_min = df_all[df_all['group'] == 1]
    df_maj = df_all[df_all['group'] == 0]
    
    print(df_all)
    idx = df_min.groupby("group")['betn'].idxmax().iloc[0]
    node = idx

    pred_idx = list(g.successors(node))
    pred_df_all = df_all.loc[pred_idx,:]
    
    pred_df_min = pred_df_all [pred_df_all ['group'] == 1]
    pred_df_maj = pred_df_all [pred_df_all ['group'] == 0]
    pred_min_size, pred_maj_size = pred_df_min["betn"] * scale_degree_size + 15,  pred_df_maj["betn"] * scale_degree_size + 15
    
    ax.scatter(pred_df_min["x"], pred_df_min["y"],  s=pred_min_size, marker="^",alpha=0.6,label=1,color="orange")
    ax.scatter(pred_df_maj["x"], pred_df_maj["y"], s=pred_maj_size, alpha=0.6, marker="o",label=0,color="blue")
    ax.scatter(df_all.loc[node,"x"],df_all.loc[node,"y"],s=df_all.loc[node,"betn"]*scale_degree_size+15,color="green",marker="^")
    # df_mid_size = df_mid["betn"]*scale_degree_size + 15

    # ax.scatter(df_min["x"], df_min["y"],  s=df_min_size, marker="^",alpha=0.6,label=1,color="orange")
    # ax.scatter(df_maj["x"], df_maj["y"], s=df_maj_size, alpha=0.6, marker="o",label=0,color="blue")
    
    # ax.scatter(df_mid["x"], df_mid["y"], s=df_mid_size, alpha=0.6, marker="x",label=2,color="green")

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    # ax.set_xlim(-60,60)
    # ax.set_ylim(-60,60)

    ax.set_xlim(-40,40)
    ax.set_ylim(-40,40)
    ax.legend(loc="upper right")
    fig.savefig('tsne_emb_model_{}_ds{}_t_{}.png'.format(model,ds,t),bbox_inches='tight')   # save the figure to file
    plt.close(fig)  

def plot_embeddings_syn(args_model,model,filename,hMM,hmm,t):
    fig, ax = plt.subplots(nrows=1, ncols=1) 
    
    g = read_graph(filename)
    scale_degree_size = 10000

    embedding_folder = "./embedding/{}".format(model)
    if not os.path.exists(embedding_folder): os.makedirs(embedding_folder)
    embedding_file_name = embedding_folder + "/_hMM{}_hmm{}_t_{}.pkl".format(hMM,hmm,t)


    centrality_file_name = "./centrality/betweenness/{}_fm_0.3/_hMM{}_hmm{}.pkl".format(model,hMM,hmm)
    print("Embedding file name: ", embedding_file_name)

    with open(embedding_file_name, 'rb') as f:
                embeds = pkl.load(f) # node idx & 64 embeds
    if t == 28 and  os.path.exists(centrality_file_name):
        with open(centrality_file_name, 'rb') as f:
                    cent_dict = pkl.load(f) # node idx & 64 embeds
    else:
        cent_dict = nx.betweenness_centrality(g, normalized=True)
    
    embeds_data = get_metadata_df(g, cent_dict)
    
    tsne = TSNE(n_components=2, random_state=42)
    embeddings2d = tsne.fit_transform(embeds)

    df_embeds = pd.DataFrame({'x': embeddings2d[:,0], 'y': embeddings2d[:, 1]}, index=embeds.index)
    df_all = pd.concat([df_embeds,embeds_data],axis=1)

    ## Coloring the points by betn
    cmap =  "OrRd"
    norm = plt.Normalize(0, 0.004)
    print("Plotting tsne embeddings")

    df_min = df_all[df_all['group'] == 1]
    df_maj = df_all[df_all['group'] == 0]
    print(df_min.shape, df_maj.shape)
    df_min_size, df_maj_size = df_min["betn"] * scale_degree_size + 15,  df_maj["betn"] * scale_degree_size + 15
    ax.scatter(df_min["x"], df_min["y"],  s=df_min_size, marker="^",alpha=0.6,label="Minority",color="orange")
    ax.scatter(df_maj["x"], df_maj["y"], s=df_maj_size, alpha=0.6, marker="o",label="Majority",color="blue")
    
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_xlim(-50,50)
    ax.set_ylim(-50,50)

    ax.legend(loc="upper right")
    fig.savefig('tsne_emb_model_{}_hMM{}_hmm{}_t_{}.png'.format(model,hMM,hmm,t),bbox_inches='tight')   # save the figure to file
    plt.close(fig)  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hMM", help="homophily between Majorities", type=float, default=0.5)
    parser.add_argument("--hmm", help="homophily between minorities", type=float, default=0.5)
    parser.add_argument("--model", help="Different Walker Models", type=str)
    parser.add_argument("--name", help="Real Datasets (rice)", type=str)
    parser.add_argument("--p", help="Return parameter", type=float, default=1.0)
    parser.add_argument("--q", help="In-out parameter", type=float, default=1.0)
    parser.add_argument("--beta", help="Beta paramater", type=float, default=2.0)
    parser.add_argument("--alpha", help="Alpha paramater (Levy)", type=float, default=1.0)
    parser.add_argument("--seed", help="Seed", type=int, default=42)
   
    args = parser.parse_args()
    
    extra_params = dict()
    if args.model == "commonngh":
        model = args.model
    elif args.model in ["levy", "highlowindegree"]:
       model =  "{}_alpha_{}".format(args.model,args.alpha)
       extra_params = {"alpha":args.alpha}
    elif args.model in ["levy", "highlowindegree"]:
         extra_params = {"alpha":args.alpha}
    elif args.model in ["fw","n2v"]:
        model = args.model + "_p_{}_q_{}".format(args.p,args.q)
        extra_params = {"p":args.p,"q":args.q}
    elif args.model in  ["nonlocalindegree","nonlocaltrialindegree","nonlocalindegreelocalrandom","nllindegreelocalrandom","nlindlocalind"]:
        model = "{}_alpha_{}_beta_{}".format(args.model,args.alpha,args.beta)
        extra_params = {"alpha":args.alpha,"beta":args.beta}
    elif args.model == "fairindegreev2":
        model = args.model
    else:
       model =  "{}_beta_{}".format(args.model,args.beta)
       extra_params = {"beta":args.beta}

    main_directory = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/"
    t = 28
    
    """Real ds"""
    ds = args.name
    filename = main_directory+"model_{}_name_{}/seed_42/_{}-name_{}_t_{}.gpickle".format(model,ds,model,ds,t)
    plot_embeddings(model,filename,ds,t)
    
    
    # hMM, hmm = args.hMM, args.hmm
    # filename = main_directory+"{}_fm_0.3/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_{}.gpickle".format(model,model,hMM,hmm,t)
    # plot_embeddings_syn(args.model,model,filename,hMM, hmm, t)
    # plot_embeddings_bet(model,filename,ds,t)


    # ds = "twitter"
    # for t in [0,15,28]:
    #     file_name = main_directory+"model_nlindlocalind_alpha_0.3_beta_2.0_name_{}/seed_42/_nlindlocalind_alpha_0.3_beta_2.0-name_{}_t_{}.gpickle".format(ds,ds,t)
    #     plot_embeddings(file_name,ds,t)