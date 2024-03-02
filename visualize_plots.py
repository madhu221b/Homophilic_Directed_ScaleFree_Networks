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
from generate_heatmap_centrality import get_grid

T = 30
hMM_list, hmm_list = np.arange(0,1.1,0.1), np.arange(0,1.1,0.1)


def get_diff_avg_betn(model,g,hMM,hmm):
    centrality_dict = get_centrality_dict(model,hMM,hmm)
    minority_centrality = [val for node, val in centrality_dict.items() if g.nodes[node]["m"] == 1]
    avg_min_val = np.mean(minority_centrality)

    majority_centrality = [val for node, val in centrality_dict.items() if g.nodes[node]["m"] == 0]
    avg_maj_val = np.mean(majority_centrality)
    diff_val =  avg_min_val - avg_maj_val
    return diff_val


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
    model_list = ["degree_beta_2.0", "indegree_beta_2.0_fm_0.3","commonngh","n2v_p_1.0_q_1.0","fw_p_1.0_q_1.0"]
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

def get_edge_dict_v2(g, edge_list=[]):
    key_list=["M->M","m->M","M->m","m->m"]
    edge_dict = {key:0 for key in key_list}
    node_attr = nx.get_node_attributes(g, "group")
    
    if edge_list: itr = edge_list
    else: itr = g.edges()
    for u, v in itr:
        key = "{}->{}".format(get_label(node_attr[u]),get_label(node_attr[v]))      
        edge_dict[key]+= 1
    return edge_dict

def get_homo_to_edge_dict(model):
    main_dict = dict()
    for hMM in hMM_list:
        for hmm in hmm_list:
            hMM, hmm = np.round(hMM,2), np.round(hmm,2)

            if model.startswith("indegree"):
                path = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/{}/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_29.gpickle".format(model,model,hMM,hmm)
            else:
                path = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/{}/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}.gpickle".format(model,model,hMM,hmm)
            g = nx.read_gpickle(path)
            node2group = {node:g.nodes[node]["m"] for node in g.nodes()}
            nx.set_node_attributes(g, node2group, 'group')
            edge_dict = get_edge_dict(g)
            key = "{},{}".format(hMM,hmm)
            main_dict[key] = edge_dict

    return main_dict

def plot_diff_scatter_edge_link_ratio(model1, model2, display_keys=["min_inlink","min_outlink"]):
    homo_to_edge_link_1 = get_homo_to_edge_dict(model1)
    homo_to_edge_link_2 = get_homo_to_edge_dict(model2)
    display_dict = {"min_inlink":"m->m", "min_outlink":"m->M","maj_outlink":"M->m","maj_inlink":"M->M"}
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    cmap = "coolwarm"
    xdata1, ydata1, zdata1, c1 = [], [], [], []
    zdata2, c2 = [], []
    for key, value in homo_to_edge_link_1.items():
        arr = key.split(",")
        hMM, hmm = float(arr[0]), float(arr[1])
        xdata1.append(hmm)
        ydata1.append(hMM)
        zdata1.append(1)
        zdata2.append(2)
        diff1 = value[display_keys[0]] - homo_to_edge_link_2[key][display_keys[0]]
        diff2 = value[display_keys[1]] - homo_to_edge_link_2[key][display_keys[1]]
        c1.append(diff1)
        c2.append(diff2)

    all_data = np.concatenate([c1, c2])
    norm = plt.Normalize(np.min(all_data), np.max(all_data))
    data1 = ax.scatter3D(xdata1, ydata1, zdata1, c=c1, cmap=cmap,marker="o",norm=norm)
    data2 = ax.scatter3D(xdata1, ydata1, zdata2, c=c2, cmap=cmap,marker="^",norm=norm)

    ax.set_xlabel('hmm')
    ax.set_ylabel('hMM')
    ax.set_yticks([0.0,0.5,1.0])
    ax.set_xticks([0.0,0.5,1.0])
    ax.set_zticks([1,2])
    ax.set_zticklabels([display_dict[display_keys[0]], display_dict[display_keys[1]]])
    ax.invert_yaxis()

    #c1.extend(c2)
    # norm = plt.Normalize(np.min(c1), np.max(c1)) norm=norm
    smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(smap, ax=ax, fraction=0.1, shrink = 0.8, orientation="horizontal")

    # sm.set_array([])
    # cb = ax.figure.colorbar(sm,)
    cb.ax.set_title("Variation in Edge Ratio")
    fig.savefig('plots/scatter_diff__{}_{}.pdf'.format(model1,model2),bbox_inches='tight')   # save the figure to file
    plt.close(fig)    # close the figure window


def plot_scatter_edge_link_ratio(model, display_keys=["min_inlink","min_outlink"]):
    homo_to_edge_link = get_homo_to_edge_dict(model)
    display_dict = {"min_inlink":"m->m", "min_outlink":"m->M","maj_outlink":"M->m","maj_inlink":"M->M"}
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    cmap = "coolwarm"
    xdata1, ydata1, zdata1, c1 = [], [], [], []
    zdata2, c2 = [], []
    for key, value in homo_to_edge_link.items():
        arr = key.split(",")
        hMM, hmm = float(arr[0]), float(arr[1])
        xdata1.append(hmm)
        ydata1.append(hMM)
        zdata1.append(1)
        zdata2.append(2)
        c1.append(value[display_keys[0]])
        c2.append(value[display_keys[1]])

    all_data = np.concatenate([c1, c2])
    norm = plt.Normalize(np.min(all_data), np.max(all_data))
    data1 = ax.scatter3D(xdata1, ydata1, zdata1, c=c1, cmap=cmap,marker="o",norm=norm)
    data2 = ax.scatter3D(xdata1, ydata1, zdata2, c=c2, cmap=cmap,marker="^",norm=norm)

    ax.set_xlabel('hmm')
    ax.set_ylabel('hMM')
    ax.set_yticks([0.0,0.5,1.0])
    ax.set_xticks([0.0,0.5,1.0])
    ax.set_zticks([1,2])
    ax.set_zticklabels([display_dict[display_keys[0]], display_dict[display_keys[1]]])
    ax.invert_yaxis()

    #c1.extend(c2)
    # norm = plt.Normalize(np.min(c1), np.max(c1)) norm=norm
    smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(smap, ax=ax, fraction=0.1, shrink = 0.8, orientation="horizontal")

    # sm.set_array([])
    # cb = ax.figure.colorbar(sm,)
    cb.ax.set_title("Strength of Edge Ratio")
    fig.savefig('plots/scatter_{}.pdf'.format(model),bbox_inches='tight')   # save the figure to file
    plt.close(fig)    # close the figure window

    

def get_pearson_betn_centrality_and_edge_link(file_path, model, ratio="maj_outlink"):
    all_files = os.listdir(file_path)
    graph_files = [os.path.join(file_path,file_name) for file_name in all_files if "netmeta" not in file_name and ".gpickle" in file_name and "t_29" in file_name]
    centrality_grid = get_grid(graph_files, model, centrality="betweenness")
    # grid[hmm_idx][hMM_idx] = avg_val

    edge_grid = np.zeros_like(centrality_grid)
    edge_ratio_dict = get_homo_to_edge_dict(model)
    for key, value in edge_ratio_dict.items():
        arr = key.split(",")
        hMM, hmm = float(arr[0]), float(arr[1])
        hMM_idx, hmm_idx = int(float(hMM)*10), int(float(hmm)*10)
        edge_grid[hmm_idx][hMM_idx] = value[ratio]
    
    # r = np.corrcoef(centrality_grid.flatten(), edge_grid.flatten())
    r = np.corrcoef(edge_grid.flatten(), centrality_grid.flatten())

    print(r[0][1])


def get_centrality_dict(model,hMM,hmm,centrality="betweenness"):

    dict_folder = "./centrality/{}/{}".format(centrality,model)  
    dict_file_name = dict_folder+"/_hMM{}_hmm{}.pkl".format(hMM,hmm)
    with open(dict_file_name, 'rb') as f:                
        centrality_dict = pkl.load(f)
        return centrality_dict

def plot_scatter_plots_2_models(model1,model2,hMM, hmm, edge_types=["M->m","m->M","m->m","M->M"]):
    # fig, ax = plt.subplots() 
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    edge_dict_1 = {key:[] for key in edge_types}
    edge_dict_2 = {key:[] for key in edge_types}

    colors = ["#2EC83D", "#3F6A43","#24D8D8", "#058181"]
    
    path1 = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/{}/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_29.gpickle".format(model1,model1,hMM,hmm)   
    path2 = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/{}/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}.gpickle".format(model2,model2,hMM,hmm)
    
    g1 = nx.read_gpickle(path1)
    node2group = {node:g1.nodes[node]["m"] for node in g1.nodes()}
    nx.set_node_attributes(g1, node2group, 'group')
    node_attr_1 = nx.get_node_attributes(g1, "group")

    g2 = nx.read_gpickle(path2)
    node2group = {node:g2.nodes[node]["m"] for node in g2.nodes()}
    nx.set_node_attributes(g2, node2group, 'group')
    node_attr_2 = nx.get_node_attributes(g2, "group")
    
    # edges = []
    for u, v in g1.edges():
        edge_label = "{}->{}".format(get_label(node_attr_1[u]),get_label(node_attr_1[v]))      
        # if edge_label == edge_type:
        # edges.append([u,v])
        if edge_label in edge_types:
           edge_dict_1[edge_label].append([u,v])
    
    for u, v in g2.edges():
        edge_label = "{}->{}".format(get_label(node_attr_2[u]),get_label(node_attr_2[v]))      
        # if edge_label == edge_type:
        # edges.append([u,v])
        if edge_label in edge_types:
           edge_dict_2[edge_label].append([u,v])
    
    centrality_dict_1 = get_centrality_dict(model1,hMM,hmm)
    centrality_dict_2 = get_centrality_dict(model2,hMM,hmm)
    all_values =  list(centrality_dict_1.values()) + list(centrality_dict_2.values())
    max_value, min_value = max(all_values), min(all_values)
    mid_value = (min_value+max_value)/2
    i = 0
    zs = [-0.25,-0.5,0,0.5]
    zs_labels = []
    zs_ticks = []
    for key, value in edge_dict_1.items():
        source_bet, dest_bet = [], []
        for source, dest in value:
            source_bet.append(centrality_dict_1[source])
            dest_bet.append(centrality_dict_1[dest])
        
        source_bet_2, dest_bet_2 = [], []
        for source, dest in edge_dict_2[key]:
            source_bet_2.append(centrality_dict_2[source])
            dest_bet_2.append(centrality_dict_2[dest])
    
        # ax.scatter(source_bet, dest_bet, marker = "o",color=colors[i],label=key)
        ax.scatter3D(source_bet, dest_bet, zs[i], marker = "o",color=colors[i])
        zs_labels.append(key)
        zs_ticks.append(zs[i])
        i+=1 
        ax.scatter3D(source_bet_2, dest_bet_2, zs[i], marker = "^",color=colors[i])
        zs_labels.append("")
        zs_ticks.append(zs[i])
        
       
    
    ax.set_xlabel("Source Node")
    ax.set_ylabel("Destination Node", labelpad=10)
    ax.set_zticks(zs_ticks)
    ax.set_zticklabels(zs_labels)
    ax.set_yticks([min_value, mid_value, max_value])
    ax.set_xticks([min_value, mid_value, max_value])
    ax.invert_yaxis()
    fig.savefig('plots/scatter_2models_hMM_{}_hmm_{}.png'.format(hMM,hmm),bbox_inches='tight')   # save the figure to file
    plt.close(fig)    # close the figure window

def plot_scatter_plots(model,hMM, hmm, edge_types=["M->m","m->M","m->m","M->M"]):
    # fig, ax = plt.subplots() 
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    edge_dict = {key:[] for key in edge_types}
    if "M->M" in edge_types:
        colors = ["#2EC83D", "#3F6A43"]
    else:
        colors = ["#24D8D8", "#058181"]
    if model.startswith("indegree"):
        path = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/{}/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_29.gpickle".format(model,model,hMM,hmm)
    else:
        path = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/{}/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}.gpickle".format(model,model,hMM,hmm)
    
    g = nx.read_gpickle(path)
    node2group = {node:g.nodes[node]["m"] for node in g.nodes()}
    nx.set_node_attributes(g, node2group, 'group')
    node_attr = nx.get_node_attributes(g, "group")
    
    # edges = []
    for u, v in g.edges():
        edge_label = "{}->{}".format(get_label(node_attr[u]),get_label(node_attr[v]))      
        # if edge_label == edge_type:
        # edges.append([u,v])
        if edge_label in edge_types:
           edge_dict[edge_label].append([u,v])
    
    centrality_dict = get_centrality_dict(model,hMM,hmm)
    max_value, min_value = max(centrality_dict.values()), min(centrality_dict.values())
    mid_value = (min_value+max_value)/2
    i = 0
    zs = [1, 2]
    zs_labels = []
    for key, value in edge_dict.items():
        source_bet, dest_bet = [], []
        for source, dest in value:
            source_bet.append(centrality_dict[source])
            dest_bet.append(centrality_dict[dest])
    
        # ax.scatter(source_bet, dest_bet, marker = "o",color=colors[i],label=key)
        ax.scatter3D(source_bet, dest_bet, zs[i], marker = "o",color=colors[i])
        zs_labels.append(key)
        i+=1 
    
    ax.set_xlabel("Source Node")
    ax.set_ylabel("Destination Node", labelpad=10)
    ax.set_zticks(zs)
    ax.set_zticklabels(zs_labels)
    ax.set_yticks([min_value, mid_value, max_value])
    ax.set_xticks([min_value, mid_value, max_value])
    ax.invert_yaxis()
    fig.savefig('plots/scatter_{}_hMM_{}_hmm_{}.pdf'.format(model,hMM,hmm),bbox_inches='tight')   # save the figure to file
    plt.close(fig)    # close the figure window


def standardize(x):
    return (x - np.mean(x)) / np.std(x)

def plot_degree_scatter_plots(model,hMM, hmm, edge_types=["M->m","m->M","m->m","M->M"]):
    # fig, ax = plt.subplots() 
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    edge_dict = {key:[] for key in edge_types}
    if "M->M" in edge_types:
        colors = ["#2EC83D", "#3F6A43"]
    else:
        colors = ["#24D8D8", "#058181"]
    if model.startswith("indegree"):
        path = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/{}_fm_0.3/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_29.gpickle".format(model,model,hMM,hmm)
    else:
        path = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/{}/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}.gpickle".format(model,model,hMM,hmm)
    
    g = nx.read_gpickle(path)
    node2group = {node:g.nodes[node]["m"] for node in g.nodes()}
    nx.set_node_attributes(g, node2group, 'group')
    node_attr = nx.get_node_attributes(g, "group")
    
    # edges = []
    for u, v in g.edges():
        edge_label = "{}->{}".format(get_label(node_attr[u]),get_label(node_attr[v]))      
        # if edge_label == edge_type:
        # edges.append([u,v])
        if edge_label in edge_types:
           edge_dict[edge_label].append([u,v])
    
    in_degree_dict =  dict(g.in_degree())
    out_degree_dict = dict(g.in_degree())
    all_values =  list(in_degree_dict.values()) + list(out_degree_dict.values())
    # all_values = standardize(np.array(all_values))
    max_value, min_value = max(all_values), min(all_values)
 
    i = 0
    zs = [1, 2]
    zs_labels = []
    for key, value in edge_dict.items():
        source_bet, dest_bet = [], []
        for source, dest in value:
            source_bet.append(out_degree_dict[source])
            dest_bet.append(in_degree_dict[dest])
    
        # ax.scatter(source_bet, dest_bet, marker = "o",color=colors[i],label=key)
        ax.scatter3D(source_bet, dest_bet, zs[i], marker = "o",color=colors[i])
        zs_labels.append(key)
        i+=1 
    
    ax.set_xlabel("In Degree Source Node")
    ax.set_ylabel("In Degree Destination Node", labelpad=10)
    ax.set_zticks(zs)
    ax.set_zticklabels(zs_labels)
    
    ax.set_yticks([min_value,  max_value])
    ax.set_xticks([min_value, max_value])
    ax.invert_yaxis()
    fig.savefig('plots/scatter_degree_{}_hMM_{}_hmm_{}.png'.format(model,hMM,hmm),bbox_inches='tight')   # save the figure to file
    plt.close(fig)    # close the figure window


def utility_visualization():
    """
    time vs precision & recall
    """
    homo_configs = ["0.2,0.8"]
    models = ["model_indegree_beta_2.0_fm_0.3", "model_fw_p_1.0_q_1.0_fm_0.3","model_ingroupdegree_beta_1.0_fm_0.3"]
    colors = ["#38A055", "#756AB6", "#2EC83D"]
    linestyles = ["solid", "dashed"]
    
    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax2 = ax1.twinx()

    for i, model in enumerate(models):
        config = homo_configs[0]
        hMM, hmm = config.split(",")
        dict_path = "../Homophilic_Directed_ScaleFree_Networks/utility/{}/seed_42/_hMM{}_hmm{}.pkl".format(model,hMM,hmm)
        with open(dict_path,"rb") as f:
              result_dict = pkl.load(f)

        result_dict = dict(sorted(result_dict.items()))
        x_vals = result_dict.keys()
        y_vals_1 = [value["recall"] for _, value in result_dict.items()]
        y_vals_2 =  [value["precision"] for _, value in result_dict.items()]

        lns1 = ax1.plot(x_vals, y_vals_1, marker="o", color=colors[i], label="Recall", linestyle=linestyles[0])
        lns2 = ax2.plot(x_vals, y_vals_2, marker="o", color=colors[i], label="Precision", linestyle=linestyles[1])
   

    #dummy lines with NO entries, just to create the black style legend
    dummy_lines = []
    for linestyle in linestyles:
        dummy_lines.append(ax1.plot([],[], c="black", ls = linestyle)[0])

    lines = ax1.get_lines()
    ax1.legend([line for line in lines],["Indegree", "Fairwalk", "Cross In-Degree"],loc='upper center', bbox_to_anchor=(0.5, 1))
    ax2.legend([dummy_line for dummy_line in dummy_lines],["Recall", "Precision"],loc='upper center', bbox_to_anchor=(0.3, 1))
    # axes.add_artist(legend1) . 
    #     ax1.legend(lns, labs, loc=0)
    

    ax1.set_xlabel("Timesteps")
    ax1.set_ylim(-0.01,0.25)
    ax2.set_ylim(-0.01,0.75)
    ax1.set_ylabel("Recall")
    ax2.set_ylabel("Precision") 
    

    fig.savefig('utility_{}.png'.format(config),bbox_inches='tight')   # save the figure to file
    plt.close(fig)  


def get_grid_utility(files, metric):
    grid = np.zeros((len(hmm_list),len(hMM_list)))
    for dict_path in files:
        hMM, hmm = dict_path.split("hMM")[-1].split("_")[0], dict_path.split("hmm")[-1].split(".pkl")[0]
        hMM_idx, hmm_idx = int(float(hMM)*10), int(float(hmm)*10)
        print("hMM: {}, hmm: {}".format(hMM, hmm))
        with open(dict_path, 'rb') as f:
             dict_ = pkl.load(f)
        grid[hmm_idx][hMM_idx] = dict_[T-1][metric]
    return grid

def get_heatmap_utility(folder_path,metric=""):  

    plot_directory = "./plots/heatmap/utility"
    if not os.path.exists(plot_directory): os.makedirs(plot_directory)

    all_files = os.listdir(folder_path)
    files = [os.path.join(folder_path,file_name) for file_name in all_files]
    grid = get_grid_utility(files, metric)
  
    heatmap = grid.T

    hmm_ticks = [np.round(hmm,2) for hmm in hmm_list]
    hMM_ticks = [np.round(hMM,2) for hMM in hMM_list]
    vmin, vmax = np.min(heatmap), np.max(heatmap)
    print("vmin: ", vmin, "vmax: ", vmax)
    if metric == "precision":
        vmin, vmax = 0.1, 0.8
    elif metric == "recall":
        vmin, vmax = 0, 0.06
    cmap = plt.cm.get_cmap('BuGn')
    ax = sns.heatmap(heatmap, cmap=cmap,xticklabels=hmm_ticks,yticklabels=hMM_ticks,vmin=vmin,vmax=vmax)
    ax.invert_yaxis()
    ax.set_xlabel("Homophily for Minority Class")
    ax.set_ylabel("Homophily for Majority Class")
    fig = ax.get_figure()
    model = "fw" if "fw" in folder_path else "ind"
    fig.savefig(plot_directory+"/{}_{}.png".format(metric,model))

def get_edge_info(model, hMM, hmm):
    main_directory = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/"
    # mention for what timesteps you are looking for edge ratios
    T = [0,29]

    # file_path = main_directory+"{}_fm_0.3/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_{}.gpickle".format(model,model,hMM,hmm,T[0])
    file_path = main_directory + "DPAH_fm_0.3/DPAH-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}-ID0.gpickle".format(hMM,hmm)
    g = nx.read_gpickle(file_path)
    node2group = {node:g.nodes[node]["m"] for node in g.nodes()}
    nx.set_node_attributes(g, node2group, 'group')
    edge_dict_1 = get_edge_dict(g)
    # edge_dict = get_edge_dict_v2(g)
    # sum_ = sum(edge_dict.values())
    # edge_dict_1 = {k:np.round((v/sum_)*100.0,2) for k,v in edge_dict.items()}
    # print('edge_dict_1: ', edge_dict_1)

    file_path = main_directory+"{}_fm_0.3/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_{}.gpickle".format(model,model,hMM,hmm,T[1])
    g = nx.read_gpickle(file_path)
    edge_dict_2 = get_edge_dict(g)
    # edge_dict = get_edge_dict_v2(g)
    # sum_ = sum(edge_dict.values())
    # edge_dict_2 = {k:np.round((v/sum_)*100.0,2) for k,v in edge_dict.items()}
    print('edge_dict_2: ', edge_dict_2)

    sub_dict = {k:np.round((v-edge_dict_1[k]),2) for k,v in edge_dict_2.items()}
    print(sub_dict)

def plot_hmm_vs_avgminbetn_edgelink(model, hMM, hmm):
    """
    
    # edge_dict = get_edge_dict_v2(g)
    # sum_ = sum(edge_dict.values())
    # edge_dict_1 = {k:np.round((v/sum_)*100.0,2) for k,v in edge_dict.items()}
    # print('edge_dict_1: ', edge_dict_1)

    # edge_dict = get_edge_dict_v2(g)
    # sum_ = sum(edge_dict.values())
    # edge_dict_2 = {k:np.round((v/sum_)*100.0,2) for k,v in edge_dict.items()}
    print('edge_dict_2: ', edge_dict_2)

    """
    main_directory = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/"
    # mention for what timesteps you are looking for edge ratios
    T = [0,29]

    # file_path = main_directory+"{}_fm_0.3/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_{}.gpickle".format(model,model,hMM,hmm,T[0])
    linestyles = {'maj_inlink': "solid", 'min_inlink': "dashed", 'min_outlink':"dotted", 'maj_outlink':"dashdot"}
    colors = ["#38A055", "#756AB6"]
    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax2 = ax1.twinx()
    
    # hMM = 1.0
    hmm = 1.0
    models = [model,"fw_p_1.0_q_1.0"]
    for i, model in enumerate(models):
     
        plot_dict = dict()
        diff_dict = dict()
        # for hmm in hmm_list:
        #     hmm = np.round(hmm,2)
        for hMM in hMM_list:
            hMM = np.round(hMM,2)
            file_path = main_directory + "DPAH_fm_0.3/DPAH-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}-ID0.gpickle".format(hMM,hmm)
            g = nx.read_gpickle(file_path)
            node2group = {node:g.nodes[node]["m"] for node in g.nodes()}
            nx.set_node_attributes(g, node2group, 'group')
            diff_1 = get_diff_avg_betn("DPAH_fm_0.3",g,hMM,hmm)
            edge_dict_1 = get_edge_dict(g)
        
            file_path = main_directory+"{}_fm_0.3/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_{}.gpickle".format(model,model,hMM,hmm,T[1])
            g = nx.read_gpickle(file_path)
            edge_dict_2 = get_edge_dict(g)
            
            model_bet = model+"_fm_0.3"
            diff_2 = get_diff_avg_betn(model_bet,g,hMM,hmm)
            # diff_dict[hmm] = diff_2 
            diff_dict[hMM] = diff_2 

            for k,v in edge_dict_2.items():
                if k not in plot_dict:
                    plot_dict[k] = dict()
                sub = np.round((v-edge_dict_1[k]),2)
                # plot_dict[k][hmm] = sub
                plot_dict[k][hMM] = sub
            

            
        print("plot dict: ", plot_dict)
        ax1.plot(diff_dict.keys(), diff_dict.values(), marker=">",label=model,color=colors[i])
        for edge_link, sub_dict in plot_dict.items():    
             x_vals = sub_dict.keys()
             y_vals = sub_dict.values()
             if edge_link == "maj_inlink" or edge_link == "min_inlink": x_vals, y_vals = [],[]
             ax2.plot(x_vals,y_vals,linestyle=linestyles[edge_link],marker="o",label=edge_link,color=colors[i])           
        

    dummy_lines = []
    for linestyle in linestyles.values():
        dummy_lines.append(ax1.plot([],[], c="black", ls = linestyle)[0])

    lines = ax1.get_lines()
    ax1.legend([line for line in lines],models,loc='upper center', bbox_to_anchor=(0.5, 1))
    ax2.legend([dummy_line for dummy_line in dummy_lines],list(linestyles.keys()),loc='upper center', bbox_to_anchor=(0.3, 1))
    # axes.add_artist(legend1) . 
    #     ax1.legend(lns, labs, loc=0)
    

    ax1.set_xlabel("hmm for Fixed hMM: {}".format(hMM))
    # ax1.set_ylim(-0.01,0.25)
    # ax2.set_ylim(-0.01,0.75)
    ax1.set_ylabel("avg(m) - avg(M) betn centrality")
    ax2.set_ylabel("Delta Edge Link Ratio") 
    

    fig.savefig('trial.png',bbox_inches='tight')   # save the figure to file
    plt.close(fig)  

def get_in_avg(g, deg="min"):
    in_degree_dict =  dict(g.in_degree())
    node_attr = nx.get_node_attributes(g, "group")
    
    if deg=="min":
        res =  np.mean(np.array([val for node, val in in_degree_dict.items() if get_label(node_attr[node]) == "m"]))
        return res
    elif deg=="maj":
        res =  np.mean(np.array([val for node, val in in_degree_dict.items() if get_label(node_attr[node]) == "M"]))
        return res


def plot_hmm_vs_avgdegree(model, hMM, hmm):
    main_directory = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/"
    # mention for what timesteps you are looking for edge ratios

    # file_path = main_directory+"{}_fm_0.3/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_{}.gpickle".format(model,model,hMM,hmm,T[0])
    # linestyles = {'avg_min': "solid", 'avg_maj: "dashed"}
    linestyles = ["solid", "dashed"]
    colors = ["#38A055", "#756AB6"]
    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax2 = ax1.twinx()
    
    hMM = 1.0
    # hmm = 1.0
    models = [model,"fw_p_1.0_q_1.0"]
    for i, model in enumerate(models):
     
        avg_dict = dict()
        betn_dict = dict()
        for hmm in hmm_list:
            hmm = np.round(hmm,2)
       
            file_path = main_directory+"{}_fm_0.3/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_{}.gpickle".format(model,model,hMM,hmm,29)
            g = nx.read_gpickle(file_path)

            min_deg = get_in_avg(g,deg="min")
            maj_deg = get_in_avg(g, deg="maj")
            if hmm not in avg_dict: avg_dict[hmm] = dict()
            
            model_bet = model+"_fm_0.3"
            betn = get_diff_avg_betn(model_bet,g,hMM,hmm)

            # avg_dict[hmm]["min"] = min_deg
            avg_dict[hmm]["min"] = min_deg - maj_deg
            betn_dict[hmm] = betn
    
        ax1.plot(betn_dict.keys(),betn_dict.values(), marker="o",label=model,color=colors[i],linestyle=linestyles[0])
        avg_deg = np.array([_["min"] for _ in avg_dict.values()])
        # avg_deg_2 = 2*(avg_deg  - min(avg_deg )) / ( max(avg_deg ) - min(avg_deg )) - 1
        avg_deg_2 = avg_deg
        print(max(avg_deg_2), min(avg_deg_2))
        ax2.plot(avg_dict.keys(),avg_deg_2, marker="o",label=model,color=colors[i],linestyle=linestyles[1])


    dummy_lines = []
    for linestyle in linestyles:
        dummy_lines.append(ax1.plot([],[], c="black", ls = linestyle)[0])

    lines = ax1.get_lines()
    ax1.legend([line for line in lines],models,loc='upper center', bbox_to_anchor=(0.7, 0.1))
    ax2.legend([dummy_line for dummy_line in dummy_lines],["avg(m) - avg(M) bet centrality ", "avg(m) - avg(M) in-degree"],loc='upper center', bbox_to_anchor=(0.3, 0.1))
    # axes.add_artist(legend1) . 
    #     ax1.legend(lns, labs, loc=0)
    

    ax1.set_xlabel("Varying hmm for Fixed hMM: {}".format(hMM))
    ax1.set_ylim(-0.004,0.004)
    ax2.set_ylim(-40,40)
    ax1.set_ylabel("avg(m)-avg(M) betn centrality")
    ax2.set_ylabel("avg(m) - avg(M) in-degree") 
    

    fig.savefig('trial.png',bbox_inches='tight')   # save the figure to file
    plt.close(fig)  

def plot_hMM_vs_avgdegree(model, hMM, hmm):
    main_directory = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/"
    # mention for what timesteps you are looking for edge ratios

    # file_path = main_directory+"{}_fm_0.3/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_{}.gpickle".format(model,model,hMM,hmm,T[0])
    # linestyles = {'avg_min': "solid", 'avg_maj: "dashed"}
    linestyles = ["solid", "dashed"]
    colors = ["#38A055", "#756AB6"]
    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax2 = ax1.twinx()
    
    hmm = 1.0
    models = [model,"fw_p_1.0_q_1.0"]
    for i, model in enumerate(models):
     
        avg_dict = dict()
        betn_dict = dict()
        for hMM in hMM_list:
            hMM = np.round(hMM,2)
       
            file_path = main_directory+"{}_fm_0.3/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_{}.gpickle".format(model,model,hMM,hmm,29)
            g = nx.read_gpickle(file_path)

            min_deg = get_in_avg(g,deg="min")
            maj_deg = get_in_avg(g, deg="maj")
            if hMM not in avg_dict: avg_dict[hMM] = dict()
            
            model_bet = model+"_fm_0.3"
            betn = get_diff_avg_betn(model_bet,g,hMM,hmm)

            # avg_dict[hmm]["min"] = min_deg
            avg_dict[hMM]["min"] = min_deg - maj_deg
            betn_dict[hMM] = betn
    
        ax1.plot(betn_dict.keys(),betn_dict.values(), marker="o",label=model,color=colors[i],linestyle=linestyles[0])
        avg_deg = np.array([_["min"] for _ in avg_dict.values()])
        # avg_deg_2 = 2*(avg_deg  - min(avg_deg )) / ( max(avg_deg ) - min(avg_deg )) - 1
        avg_deg_2 = avg_deg
        print(max(avg_deg_2), min(avg_deg_2))
        ax2.plot(avg_dict.keys(),avg_deg_2, marker="o",label=model,color=colors[i],linestyle=linestyles[1])


    dummy_lines = []
    for linestyle in linestyles:
        dummy_lines.append(ax1.plot([],[], c="black", ls = linestyle)[0])

    lines = ax1.get_lines()
    ax1.legend([line for line in lines],models,loc='upper center', bbox_to_anchor=(0.7, 0.1))
    ax2.legend([dummy_line for dummy_line in dummy_lines],["avg(m) - avg(M) bet centrality ", "avg(m) - avg(M) in-degree"],loc='upper center', bbox_to_anchor=(0.3, 0.1))
    # axes.add_artist(legend1) . 
    #     ax1.legend(lns, labs, loc=0)
    

    ax1.set_xlabel("Varying hMM for Fixed hmm: {}".format(hmm))
    ax1.set_ylim(-0.004,0.004)
    ax2.set_ylim(-40,40)
    ax1.set_ylabel("avg(m)-avg(M) betn centrality")
    ax2.set_ylabel("avg(m) - avg(M) in-degree") 
    

    fig.savefig('trial.png',bbox_inches='tight')   # save the figure to file
    plt.close(fig)  



def get_diff_of_rate(hmm):
    linestyles = ["solid", "dashed"]
    colors = ["#81B622","#D8A7B1","#38A055","#756AB6"]
    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax2 = ax1.twinx()
    ax2.set_axis_off()

    main_directory = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/"
    
    models = ["fw_p_1.0_q_1.0", "indegree_beta_2.0"]
    
    for i, model in enumerate(models):
        res_dict = dict()
        for hMM in hMM_list:
            hMM = np.round(hMM,2)

            dpah_file = main_directory + "DPAH_fm_0.3/DPAH-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}-ID0.gpickle".format(hMM,hmm)
            g_init = nx.read_gpickle(dpah_file)
            node2group = {node:g_init.nodes[node]["m"] for node in g_init.nodes()}
            nx.set_node_attributes(g_init, node2group, 'group')
            
        

            file_path = main_directory+"{}_fm_0.3/{}-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM{}-hmm{}_t_{}.gpickle".format(model,model,hMM,hmm,29)
            g_final = nx.read_gpickle(file_path)
            new_edges = list(set(g_final.edges()) - set(g_init.edges()))
            edge_dict = get_edge_dict_v2(g_final, new_edges)
            sum_ = sum(edge_dict.values())
    
            for k, v in edge_dict.items():
                val = np.round((v/sum_)*100.0,2)
                if k not in res_dict: res_dict[k] = dict()
                res_dict[k][hMM] = val
        
        for j, (k,sub_dict) in enumerate(res_dict.items()):
            ax1.plot(sub_dict.keys(), sub_dict.values(), marker="o",label=k,color=colors[j],linestyle=linestyles[i])
     

    dummy_lines = []
    for linestyle in linestyles:
        dummy_lines.append(ax1.plot([],[], c="black", ls = linestyle)[0])
    
    lines = ax1.get_lines()
    ax1.legend([line for line in lines if line.get_linestyle()=="-"],[line.get_label() for line in lines if line.get_linestyle()=="-"], bbox_to_anchor=(0.3, 1.0))
    ax2.legend([dummy_line for dummy_line in dummy_lines],models, bbox_to_anchor=(0.7, 1.0))
    ax1.set_xlabel("Varying hMM for Fixed hmm: {}".format(hmm))
    ax1.set_ylabel("Percent of new recommendations")
    ax1.set_ylim(0,100)

    fig.savefig('new_recos_hmm_{}.png'.format(hmm),bbox_inches='tight')   # save the figure to file
    plt.close(fig)  

       
      


if __name__ == "__main__":
    # model = "levy_alpha_-1.0"
    # path = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/{}".format(model)
    # get_line_plot_t_vs_frac(path)
    # "0.2,0.2", "0.8,0.8", "0.2,0.8", "0.8,0.2"
    # get_line_plot_t_vs_frac_bet(path, ["0.2,0.8", "0.8,0.2"], model)
    # get_for_all_combinations("indegree_beta", [-1.0,0.0,1.0,2.0,5.0,10.0])
    #  get_for_all_combinations()
    # get_heatmap_frac(path, "fw")

    #    path = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/DPAH/DPAH-N100-fm0.3-d0.03-ploM2.5-plom2.5-hMM0.8-hmm0.2-ID0.gpickle"
    #    visualize_walk(path,model="fairindegree",extra_params={"beta":2.0})
    # # get_homo_to_edge_dict(model="indegree_beta_2.0")
    # plot_scatter_edge_link_ratio(model="n2v_p_1.0_q_1.0",display_keys=["maj_outlink","min_outlink"])
    # get_homo_to_edge_dict(model="indegree_beta_2.0")
    # plot_scatter_edge_link_ratio(model="indegree_beta_2.0",display_keys=["maj_outlink","min_outlink"])
    #  plot_diff_scatter_edge_link_ratio("indegree_beta_2.0","n2v_p_1.0_q_1.0", display_keys=["maj_inlink","min_inlink"])
   
    # model = "indegree_beta_2.0"
    # model = "indegree_beta_2.0"
    # file_path  = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/{}".format(model)
    # get_pearson_betn_centrality_and_edge_link(file_path, model, ratio="min_inlink")
    
    # model = "indegree_beta_2.0"
    # plot_degree_scatter_plots(model,hMM=0.0, hmm=0.3, edge_types=["M->M","M->m"])
    # plot_scatter_plots_2_models(model1="indegree_beta_2.0",model2 = "n2v_p_1.0_q_1.0",hMM=1.0, hmm=0.1, edge_types=["m->M","M->m"])

    # utility_visualization()
    # folder_path = "../Homophilic_Directed_ScaleFree_Networks/utility/model_fw_p_1.0_q_1.0_fm_0.3/seed_42/"
    # folder_path = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/utility/model_indegree_beta_2.0_fm_0.3/seed_42"
    # get_heatmap_utility(folder_path,metric="recall")

    # model = "indegree_beta_2.0"
    # hMM, hmm = 0.0, 0.3
    # get_edge_info(model, hMM, hmm)
    # plot_hmm_vs_avgdegree(model, hMM, hmm)
    
    model = "indegree_beta_2.0"
    hmm = 0.0
    get_diff_of_rate(hmm)
    