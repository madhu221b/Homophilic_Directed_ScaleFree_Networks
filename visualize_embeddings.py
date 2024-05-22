import os
import pickle as pkl
import networkx as nx
from sklearn.manifold import TSNE
from gensim import models
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from org.gesis.lib.n2v_utils import recommender_model_walker, recommender_model, read_graph

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
    elif "nonlocaltrialindegree" in model:
        extra_params = {"beta":2.0,"alpha":1.0}
        model, embeds = recommender_model_walker(g,model="nonlocaltrialindegree",extra_params=extra_params)
    elif "nlindlocalind" in model:
        extra_params = {"beta":2.0,"alpha":0.3}
        model, embeds = recommender_model_walker(g,model="nlindlocalind",extra_params=extra_params)
    elif "nonlocaladaptivealpha" in model:
        extra_params = {"beta":2.0}
        model, embeds = recommender_model_walker(g,model="nonlocaladaptivealpha",extra_params=extra_params)
    else:
        extra_params = {"beta":2.0}
        model, embeds = recommender_model_walker(g,model="indegree",extra_params=extra_params)
   
    
    print("Saving embeddings at file name:", embedding_file_name)
    with open(embedding_file_name, 'wb') as f:
        pkl.dump(embeds, f)
    return embeds

def plot_embeddings(filename,model,extra_params,MM,hmm,t):
    g = read_graph(filename)

    if "fw" not in model:
       model_, embeds = recommender_model_walker(g,model=model.split("_")[0],extra_params=extra_params)
    else:
        p , q = extra_params["p"], extra_params["q"]
        model_, embeds = recommender_model(g,model=model.split("_")[0],p=p,q=q)
    
    embedding_folder = "./embedding/{}".format(model)
    if not os.path.exists(embedding_folder): os.makedirs(embedding_folder)
    embedding_file_name = embedding_folder + "/_hMM{}_hmm{}_t_{}.pkl".format(hMM,hmm,t)

    print("Saving embeddings at file name:", embedding_file_name)
    with open(embedding_file_name, 'wb') as f:
          pkl.dump(embeds, f)

    print("embedding shape: ", embeds.shape)


def plot_embeddings_ds(model,extra_params,filename,ds,t):
    g = read_graph(filename)
    if "fw" not in model:
       model_, embeds = recommender_model_walker(g,model=model.split("_")[0],extra_params=extra_params)
    else:
        p , q = extra_params["p"], extra_params["q"]
        model_, embeds = recommender_model(g,model=model.split("_")[0],p=p,q=q)

    
    embedding_folder = "./embedding/{}".format(model)
    if not os.path.exists(embedding_folder): os.makedirs(embedding_folder)
    embedding_file_name = embedding_folder + "/_{}_t_{}.pkl".format(ds,t)
   
    print("Saving embeddings at file name:", embedding_file_name)
    with open(embedding_file_name, 'wb') as f:
        pkl.dump(embeds, f)

    print("embedding shape: ", embeds.shape)
 
    


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
    t = 0
    ds = args.name
    filename = main_directory+"model_{}_name_{}/seed_42/_{}-name_{}_t_{}.gpickle".format(model,ds,model,ds,t)
    plot_embeddings_ds(model,extra_params,filename,ds,t)
    
    # hMM, hmm = args.hMM, args.hmm
    # filename = main_directory+"{}_fm_0.3/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_{}.gpickle".format(model,model,hMM,hmm,t)
    # plot_embeddings(filename,model,extra_params,hMM,hmm,t)
