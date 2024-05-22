import os
import numpy as np
import networkx as nx
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics.pairwise import cosine_similarity
import pickle as pkl

import matplotlib.pyplot as plt
import matplotlib as mpl

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    os.environ.__setitem__('DISPLAY', ':0.0')
    mpl.use('Agg')

def get_edge_dict(g,edges=list(),is_neg=False):
    edge_dict = dict()
    node_attr = nx.get_node_attributes(g, "group")
    
    if is_neg is False: edges = g.edges()

    for u, v in edges:
        key = "{}->{}".format(node_attr[u],node_attr[v])    
        if key not in edge_dict:
            edge_dict[key] = [(u,v)]
        else:
            edge_dict[key].append((u,v))
    return edge_dict


def generate_pos_neg_links(g,seed, prop_pos=0.1, prop_neg=0.1):
        """

        Following CrossWalk's methodology to sample test edges

        Select random existing edges in the graph to be postive links,
        and random non-edges to be negative links.
        prop_pos: 0.1,  # Proportion of edges to remove and use as positive samples per edge group
        prop_neg: 0.1  # Number of non-edges to use as negative samples per edge group
        """
        _rnd = np.random.RandomState(seed=seed)
        pos_edge_list, neg_edge_list = [], []

        # Select n edges at random (positive samples)
        n_edges = g.number_of_edges()
        n_nodes = g.number_of_nodes()
        non_edges = [e for e in nx.non_edges(g)]
       
        
        pos_edge_dict = get_edge_dict(g)
        neg_edge_dict = get_edge_dict(g,non_edges,is_neg=True)


        for edge_type, edges in pos_edge_dict.items():
            neg_edges = neg_edge_dict[edge_type]

            n_edges = len(edges)
            npos =  int(prop_pos*n_edges)

            neg = int(prop_neg*n_edges)
            print("Edge Type: {} , total edges: {}, sampling pos links: {} , total neg edges: {}, sampling neg links: {} ".format(edge_type,n_edges,npos,len(neg_edges),neg))
            rnd_pos_inx = _rnd.choice(n_edges, npos, replace=False)
            pos_edge_list.extend([edges[ii] for ii in rnd_pos_inx])


            rnd_neg_inx = _rnd.choice(n_edges, neg, replace=False)
            neg_edge_list.extend([neg_edges[ii] for ii in rnd_neg_inx])
        
        pos_edge_list, neg_edge_list = list(set(pos_edge_list)), list(set(neg_edge_list))
        print("Totally pos set: {}, total neg set: {}".format(len(pos_edge_list),len(neg_edge_list)))
        return pos_edge_list, neg_edge_list

def generate_pos_neg_links_v2(g,seed, prop_pos=0.1, prop_neg=0.1):
        """

        Following CrossWalk's methodology to sample test edges

        Select random existing edges in the graph to be postive links,
        and random non-edges to be negative links.
        prop_pos: 0.1,  # Proportion of edges to remove and use as positive samples per edge group
        prop_neg: 0.1  # Number of non-edges to use as negative samples per edge group
        """
        _rnd = np.random.RandomState(seed=seed)
        pos_edge_list, neg_edge_list = [], []
        edge_dict = get_edge_dict(g)
        print("Total no of Edge Groups: ", len(edge_dict))

        n_edges = g.number_of_edges()
        n_nodes = g.number_of_nodes()
        non_edges = [e for e in nx.non_edges(g)]

        # Select n edges at random (positive samples)
        min_edges = min([len(edges) for _, edges in edge_dict.items()])
        nnpos, nneg = int(prop_pos*min_edges), int(prop_neg*min_edges*len(edge_dict))

    
        rnd_inx = _rnd.choice(len(non_edges), nneg, replace=False)
        neg_edge_list = [non_edges[ii] for ii in rnd_inx]

        for edge_type, edges in edge_dict.items():
       
            n_edges = len(edges)
            npos =  nnpos
            print("Edge Type: {} , total edges: {}, sampling pos links: {} ".format(edge_type,n_edges,npos))
            rnd_pos_inx = _rnd.choice(n_edges, npos, replace=False)
            pos_edge_list.extend([edges[ii] for ii in rnd_pos_inx])
        
        pos_edge_list, neg_edge_list = list(set(pos_edge_list)), list(set(neg_edge_list))
        print("Totally pos set: {}, total neg set: {}".format(len(pos_edge_list),len(neg_edge_list)))
        return pos_edge_list, neg_edge_list


def get_train_test_graph(g, seed):
    """
    Input is graph read at t=0 (DPAH graph)
    
    Return training graph instance and list of pos-neg test edges, and true labels
    """
    pos_edge_list, neg_edge_list = generate_pos_neg_links(g,seed,prop_pos=0.1,prop_neg=0.1)
    g.remove_edges_from(pos_edge_list)
    edges = pos_edge_list + neg_edge_list
    labels = np.zeros(len(edges))
    labels[:len(pos_edge_list)] = 1
    print("!! isolates :" , list(nx.isolates(g)))
    return g, edges, labels

def get_cos_sims(df, test_edges):
    scores = list()
    for u,v  in test_edges:
        source_emb = df[df.index == u]
        target_emb = df[df.index == v]
      
        sim = cosine_similarity(source_emb, target_emb)[0]
        scores.append(sim)
           
    return scores


def get_model_metrics(g,test_edges,y_true):
    """
    Computes Precision & Recall
    - Precision: Quantifies the number of correct positive predictions made.
       Ratio of correctly predicted positive examples divided by the total number of positive examples that were predicted.
     
    - Recall: Calculated as the number of true positives divided by the total number of true positives and false negatives. 

    
    """
    y_pred = [int(g.has_edge(u,v)) for u,v in test_edges]
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print("tn: {},  fp:{},  fn: {},  tp: {}".format(tn, fp, fn, tp))
    return precision, recall, accuracy

def get_model_metrics_v2(embeddings,test_edges,y_true):
    """
    Computes Precision & Recall
    - Precision: Quantifies the number of correct positive predictions made.
       Ratio of correctly predicted positive examples divided by the total number of positive examples that were predicted.
     
    - Recall: Calculated as the number of true positives divided by the total number of true positives and false negatives. 

    
    """
    y_pred = get_cos_sims(embeddings,test_edges)
    auc_score = roc_auc_score(y_true,y_pred)
    print("auc score: ", auc_score)
    

    y_scores = y_pred
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # print("true pos rate: ", tpr)
    # print("false pos rate: ", fpr)
    # # print(thresholds)
    # # print(roc_auc_score(y_true, y_scores))
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print("Threshold value is:", optimal_threshold)

        # accuracy
    threshold = optimal_threshold
    y_pred_vals = np.where(np.array(y_pred) >= threshold,1,0)
    accuracy = accuracy_score(y_true, y_pred_vals)
    precision = precision_score(y_true, y_pred_vals)
    recall = recall_score(y_true, y_pred_vals)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_vals).ravel()
    print("tn: {},  fp:{},  fn: {},  tp: {}".format(tn, fp, fn, tp))
    print("accuracy: ", accuracy, "precision: ", precision, "recall: ", recall)


    return auc_score, precision, recall


def degree_histogram_directed(G, in_degree=False, out_degree=False):
    """Return a list of the frequency of each degree value.

    Parameters
    ----------
    G : Networkx graph
       A graph
    in_degree : bool
    out_degree : bool

    Returns
    -------
    hist : list
       A list of frequencies of degrees.
       The degree values are the index in the list.

    Notes
    -----
    Note: the bins are width one, hence len(list) can be large
    (Order(number_of_edges))
    """
    nodes = G.nodes()
    if in_degree:
        in_degree = dict(G.in_degree())
        degseq=[in_degree.get(k,0) for k in nodes]
    elif out_degree:
        out_degree = dict(G.out_degree())
        degseq=[out_degree.get(k,0) for k in nodes]
    else:
        degseq=[v for k, v in G.degree()]
    dmax=max(degseq)+1
    freq= [ 0 for d in range(dmax) ]
    for d in degseq:
        freq[d] += 1
    return freq

def plot_degree_dist(G,label=""):
    in_degree_freq = degree_histogram_directed(G, in_degree=True)
    out_degree_freq = degree_histogram_directed(G, out_degree=True)
    degrees = range(len(in_degree_freq))

    fig, ax = plt.subplots( nrows=1, ncols=1 ) 
    ax.loglog(range(len(in_degree_freq)), in_degree_freq, 'go-', label='in-degree') 
    ax.loglog(range(len(out_degree_freq)), out_degree_freq, 'bo-', label='out-degree')
    ax.legend(loc = "upper right",bbox_to_anchor=(1,1))
    ax.set_xlabel('Degree')
    ax.set_ylabel('Frequency')

    fig.savefig(label, bbox_inches='tight')   # save the figure to file
    plt.close(fig)


def get_avg_inout_degree(g):
    in_degree = np.array(list(dict(g.in_degree()).values()))
    out_degree = np.array(list(dict(g.out_degree()).values()))
    return np.mean(in_degree), np.mean(out_degree)
    