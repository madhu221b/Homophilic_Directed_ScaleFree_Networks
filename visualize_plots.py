import os
import pickle as pkl
import numpy as np
import seaborn as sns
import networkx as nx
import operator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



from org.gesis.lib.n2v_utils import get_walks


T = 30
hMM_list, hmm_list = np.arange(0,1.1,0.1), np.arange(0,1.1,0.1)


def get_line_plot_t_vs_frac_bet(path,homophily_list, model):
    fig, ax = plt.subplots( nrows=1, ncols=1 ) 

    dict_files = [file_name for file_name in os.listdir(path) if "frac" in file_name]
    for dict_path in dict_files:
        dict_path = os.path.join(path,dict_path)
        with open(dict_path,"rb") as f:
            dict_ = pkl.load(f)
        
        hMM, hmm = dict_path.split("hMM")[-1].split("-")[0], dict_path.split("hmm")[-1].split("_")[0]
        label = "hMM:{} , hmm:{}".format(hMM,hmm)
        if "{},{}".format(hMM,hmm) in homophily_list:
            bet = {t:round(val["bet"] ,5) for t,val in dict_.items()}
            dict_ = {t:round((val["min"]-val["maj"]) ,2) for t,val in dict_.items()}
            dict_,bet = dict(sorted(dict_.items())), dict(sorted(bet.items()))
            # print("hMM:{}, hmm:{}, bet : {} ".format(hMM,hmm,bet))
            print("hMM:{}, hmm:{}, T=30 : {} ".format(hMM,hmm,bet[29]*100))
            # ax.plot(dict_.keys(), dict_.values(),marker="o",label=label)
            ax.plot(bet.keys(), bet.values(),marker="o",label=label)
    
    # idxs = [_ for _ in range(len(dict_))]
    # ax.set_xticks(idxs)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Avg betweenness of minority nodes")
    ax.legend(loc = "lower right",bbox_to_anchor=(0.8,0.3))
    fig.savefig('plots/time_vs_frac_{}.png'.format(model),bbox_inches='tight')   # save the figure to file
    plt.close(fig)    # close the figure window

def get_line_plot_t_vs_frac(path):
    fig, ax = plt.subplots( nrows=1, ncols=1 ) 

    dict_files = [file_name for file_name in os.listdir(path) if "frac" in file_name]
    for dict_path in dict_files:
        dict_path = os.path.join(path,dict_path)
        with open(dict_path,"rb") as f:
            dict_ = pkl.load(f)
        
        hMM, hmm = dict_path.split("hMM")[-1].split("-")[0], dict_path.split("hmm")[-1].split("_")[0]
        label = "hMM:{} , hmm:{}".format(hMM,hmm)
        if hmm == "0.5" and hMM == "0.5": continue
        dict_ = {t:round((val["min"]-val["maj"]) ,2) for t,val in dict_.items()}
        dict_ = dict(sorted(dict_.items()))
        ax.plot(dict_.keys(), dict_.values(),marker="o",label=label)
        # ax.plot(dict_.keys(), dict_.values(),marker="-",label=label)
    
    # idxs = [_ for _ in range(len(dict_))]
    # ax.set_xticks(idxs)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Difference (frac minority - frac majority)")
    ax.legend(loc = "lower right",bbox_to_anchor=(0.7,0))
    fig.savefig('plots/time_vs_frac.png',bbox_inches='tight')   # save the figure to file
    plt.close(fig)    # close the figure window


def get_grid_frac(files):
    grid = np.zeros((len(hmm_list),len(hMM_list)))
    for dict_path in files:
        hMM, hmm = dict_path.split("hMM")[-1].split("-")[0], dict_path.split("hmm")[-1].split("_")[0]
        hMM_idx, hmm_idx = int(float(hMM)*10), int(float(hmm)*10)
        print("hMM: {}, hmm: {}".format(hMM, hmm))
        with open(dict_path, 'rb') as f:
             dict_ = pkl.load(f)
        grid[hmm_idx][hMM_idx] = round(dict_[T-1]["min"] - dict_[T-1]["maj"],2)
    return grid

def get_heatmap_frac(file_path,model=""):  

    plot_directory = "../plots/heatmap/frac"
    if not os.path.exists(plot_directory): os.makedirs(plot_directory)

    all_files = os.listdir(file_path)
    frac_files = [os.path.join(file_path,file_name) for file_name in all_files if "frac" in file_name]
    grid = get_grid_frac(frac_files)
  
    heatmap = grid.T

    hmm_ticks = [np.round(hmm,2) for hmm in hmm_list]
    hMM_ticks = [np.round(hMM,2) for hMM in hMM_list]
    vmin, vmax = -0.3, 0.3
    ax = sns.heatmap(heatmap, cmap=plt.cm.coolwarm,xticklabels=hmm_ticks,yticklabels=hMM_ticks,vmin=vmin,vmax=vmax)
    ax.invert_yaxis()
    ax.set_xlabel("Homophily for Minority Class")
    ax.set_ylabel("Homophily for Majority Class")
    fig = ax.get_figure()
    fig.savefig(plot_directory+"/frac_{}.png".format(model))


def get_for_all_combinations(model, params=[]):
    fig, ax = plt.subplots( nrows=1, ncols=1 ) 
    colors = ["#81B622","#D8A7B1","#38A055","#756AB6","r","b"]
    style = ["solid","solid"]
    for i,param in enumerate(params):
        path = "./{}_{}".format(model,param)
        dict_files = [file_name for file_name in os.listdir(path) if "frac" in file_name]
        for _, dict_path in enumerate(dict_files):
             dict_path = os.path.join(path,dict_path)
             with open(dict_path,"rb") as f:
                 dict_ = pkl.load(f)
        
             hMM, hmm = dict_path.split("hMM")[-1].split("-")[0], dict_path.split("hmm")[-1].split("_")[0]
             label = "hMM:{} , hmm:{}".format(hMM,hmm)
             bet = {t:round(val["bet"]*100.0 ,5) for t,val in dict_.items()}
             dict_ = {t:round((val["min"]-val["maj"]) ,2) for t,val in dict_.items()}
             dict_,bet = dict(sorted(dict_.items())), dict(sorted(bet.items()))
             print("hMM:{}, hmm:{}, bet : {} ".format(hMM,hmm,bet))
            # print("hMM:{}, hmm:{}, T=30 : {} ".format(hMM,hmm,bet[29]*100))
            # ax.plot(dict_.keys(), dict_.values(),marker="o",label=label)
             if hMM == "0.8" and hmm == "0.2":
                 j = 0
                 ax.plot(bet.keys(), bet.values(),linestyle=style[j],color=colors[i])
             else:
                 j = 1
                 ax.plot(bet.keys(), bet.values(),linestyle=style[j],color=colors[i],label="beta {}".format(param))
             
            #  ax2.plot(bet.keys(), bet.values(),marker=style[i],color=colors[i])
    # idxs = [_ for _ in range(len(dict_))]
    # ax.set_xticks(idxs)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Avg betweenness of minority nodes (by 100)")
    ax.legend(loc = "lower right",bbox_to_anchor=(0.,0.2))
    fig.savefig('plots/trial_{}.png'.format(model),bbox_inches='tight')   # save the figure to file
    plt.close(fig)    # close the figure window

def get_for_all_combinations():
    fig, ax = plt.subplots( nrows=1, ncols=1 ) 

    colors = ["#81B622","#D8A7B1","#38A055","#756AB6","r","b"]
    model_list = ["degree_beta_2.0", "indegree_beta_2.0","commonngh","n2v_p_1.0_q_1.0","fw_p_1.0_q_1.0"]
    style = ["solid","solid"]
    for i, model in enumerate(model_list):
        path = "./{}".format(model)
        print(model)
        dict_files = [file_name for file_name in os.listdir(path) if "frac" in file_name]
        for _, dict_path in enumerate(dict_files):
             dict_path = os.path.join(path,dict_path)
             with open(dict_path,"rb") as f:
                 dict_ = pkl.load(f)
        
             hMM, hmm = dict_path.split("hMM")[-1].split("-")[0], dict_path.split("hmm")[-1].split("_")[0]
             label = "hMM:{} , hmm:{}".format(hMM,hmm)
             try:
              bet = {t:round(val["bet"] ,5) for t,val in dict_.items()}
             except Exception as e:
                continue
             dict_ = {t:round((val["min"]-val["maj"]) ,2) for t,val in dict_.items()}
             dict_,bet = dict(sorted(dict_.items())), dict(sorted(bet.items()))
             print("hMM:{}, hmm:{}, bet : {} ".format(hMM,hmm,bet))
            # print("hMM:{}, hmm:{}, T=30 : {} ".format(hMM,hmm,bet[29]*100))
            # ax.plot(dict_.keys(), dict_.values(),marker="o",label=label)
             if hMM == "0.8" and hmm == "0.2":
                 j = 0
                 ax.plot(bet.keys(), bet.values(),linestyle=style[j],color=colors[i],marker="o")
             elif hMM == "0.2" and hmm == "0.8":
                 j = 0
                 ax.plot(bet.keys(), bet.values(),linestyle=style[j],color=colors[i],label=model, marker="o")
             
            #  ax2.plot(bet.keys(), bet.values(),marker=style[i],color=colors[i])
    # idxs = [_ for _ in range(len(dict_))]
    # ax.set_xticks(idxs)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Avg betweenness of minority nodes (by 100)")
    ax.legend(loc = "lower right",bbox_to_anchor=(0.4,0.3))
    fig.savefig('plots/trial.png',bbox_inches='tight')   # save the figure to file
    plt.close(fig)    # close the figure window

def get_edges_starting_at_u(u=None, walks=[]):
    # filter edges starting at u

    walks = [walk for walk in walks if int(walk[0]) in u]

    edge_set = []
    for walk in walks:
        if len(walk) <= 1: continue
        for i, node in enumerate(walk):
            if i == len(walk) - 1: break
            edge_set.append((int(walk[i]), int(walk[i+1])))
    edge_set = list(set(edge_set))
    return edge_set

def visualize_walk(graph_path,model,extra_params=dict()):
    G = nx.read_gpickle(graph_path)
    node2group = {node:G.nodes[node]["m"] for node in G.nodes()}
    nx.set_node_attributes(G, node2group, 'group')

    
    degree_dict = dict(G.degree())

    maj_degree = {n:degree for n, degree in degree_dict.items() if G.nodes(data=True)[n]["m"] == 0}
    min_degree = {n:degree for n, degree in degree_dict.items() if G.nodes(data=True)[n]["m"] == 1}
    maj_degree = sorted(maj_degree.items(), key=operator.itemgetter(1),reverse=True)[:1]
    min_degree = sorted(min_degree.items(), key=operator.itemgetter(1),reverse=True)[:1]
    
    # maj_nodes = [n for n,obj in G.nodes(data=True) if obj['m'] == 0]
    # min_nodes = [n for n,obj in G.nodes(data=True) if obj['m'] == 1]

    maj_nodes = [n for n,_ in maj_degree]
    min_nodes = [n for n,_ in min_degree]
    walks = get_walks(G, model=model, extra_params=extra_params)
    subset_edges_maj = get_edges_starting_at_u(u=maj_nodes, walks=walks)
    subset_edges_min = get_edges_starting_at_u(u=min_nodes, walks=walks)

    fig, ax = plt.subplots(2,1, figsize=(5,10))
    vmin, vmax = (0, 1) 
    cmap = plt.cm.coolwarm
    scale_degree_size = 5
    edgecolors = 'black'
    node2degree = dict(G.degree)

    colors = {'min':'#ec8b67', 'maj':'#6aa8cb'}
    node_color = [colors['min'] if obj['m'] else colors['maj'] for n,obj in G.nodes(data=True)]
    # pos = nx.nx_pydot.graphviz_layout(G, prog='neato')
    # pos = nx.circular_layout(G)
    pos = nx.spring_layout(G, k=0.5, iterations=20)
    alpha = .9

    labeldictmaj = {maj_node:str(maj_node) for maj_node in maj_nodes}
    labeldictmin = {min_node:str(min_node) for min_node in min_nodes}


    nx.draw_networkx_nodes(G, node_color=node_color, cmap=cmap, vmin=vmin, vmax=vmax,
                           pos=pos,
                           node_size=[node2degree[node] * scale_degree_size + 15. for node in G.nodes()], 
                          ax=ax[0], alpha=alpha, linewidths=.2)
    nx.draw_networkx_labels(G, pos=pos, labels=labeldictmaj, ax=ax[0],font_size=8)
    nx.draw_networkx_nodes(G, node_color=node_color, cmap=cmap, vmin=vmin, vmax=vmax,
                           pos=pos, 
                           node_size=[node2degree[node] * scale_degree_size + 15. for node in G.nodes()], 
                          ax=ax[1], alpha=alpha, linewidths=.2)
    nx.draw_networkx_labels(G, pos=pos, labels=labeldictmin, ax=ax[1],font_size=8)
   
    edges_to_color = subset_edges_maj
    edge_colors = 'blue'
    alpha = 0.5
    nx.draw_networkx_edges(G, arrows=True, 
                           edgelist=edges_to_color, 
                           connectionstyle='arc3, rad = -0.1',
                           edge_color=edge_colors, pos=pos, ax=ax[0])

    edges_to_color = subset_edges_min
    edge_colors = 'orange'
    nx.draw_networkx_edges(G, arrows=True,
                           edgelist=edges_to_color, 
                           connectionstyle='arc3, rad = 0.1',
                           edge_color=edge_colors, pos=pos, ax=ax[1])
    fig.savefig("trial_{}.png".format(model), bbox_inches='tight')

def get_label(num):
    if num == 0: return "M"
    else: return "m"

def get_edge_dict(g):
    key_list=["M->M","m->M","M->m","m->m"]
    edge_dict = {key:0 for key in key_list}
    node_attr = nx.get_node_attributes(g, "group")

    for u, v in g.edges():
        key = "{}->{}".format(get_label(node_attr[u]),get_label(node_attr[v]))      
        edge_dict[key]+= 1
    
    result_dict = dict()
    maj_den, min_den = (edge_dict["M->m"]+edge_dict["M->M"]), (edge_dict["m->m"]+edge_dict["m->M"])
    result_dict["maj_inlink"] = edge_dict["M->M"]/maj_den
    result_dict["min_inlink"] = edge_dict["m->m"]/min_den
    result_dict["min_outlink"] = edge_dict["m->M"]/min_den
    result_dict["maj_outlink"] = edge_dict["M->m"]/maj_den
    return result_dict

def get_homo_to_edge_dict(model):
    main_dict = ()
    for hMM in hMM_list:
        for hmm in hmm_list:
            hMM, hmm = np.round(hMM,2), np.round(hmm,2)
            if model.startswith("indegree"):
                path = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/{}/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_29.gpickle".format(model,model,hMM,hmm)
                g = nx.read_gpickle(path)
                edge_dict = get_edge_dict(g)
                main_dict[",".format(hMM,hmm)] = edge_dict

    return main_dict

def plot_scatter_edge_link_ratio(model):
    homo_to_edge_link = get_homo_to_edge_dict(model)


    
if __name__ == "__main__":
    # model = "levy_alpha_-1.0"
    # path = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/{}".format(model)
    # get_line_plot_t_vs_frac(path)
    # "0.2,0.2", "0.8,0.8", "0.2,0.8", "0.8,0.2"
    # get_line_plot_t_vs_frac_bet(path, ["0.2,0.8", "0.8,0.2"], model)
    # get_for_all_combinations("indegree_beta", [-1.0,0.0,1.0,2.0,5.0,10.0])
   # get_for_all_combinations()
   # get_heatmap_frac(path, "fw")

#    path = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/DPAH/DPAH-N100-fm0.3-d0.03-ploM2.5-plom2.5-hMM0.8-hmm0.2-ID0.gpickle"
#    visualize_walk(path,model="fairindegree",extra_params={"beta":2.0})
    get_homo_to_edge_dict(model="indegree_beta_2.0")