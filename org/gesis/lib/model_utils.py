import numpy as np
import networkx as nx
from sklearn.metrics import precision_score, recall_score

def get_label(num):
    if num == 0: return "M"
    else: return "m"

def get_edge_dict(g):
    edge_dict = dict()
    node_attr = nx.get_node_attributes(g, "group")

    for u, v in g.edges():
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

        # Select n edges at random (positive samples)
        n_edges = g.number_of_edges()
        n_nodes = g.number_of_nodes()
        non_edges = [e for e in nx.non_edges(g)]
       
        pos_edge_list, neg_edge_list = [], []
        edge_dict = get_edge_dict(g)
        print("Total no of Edge Groups: ", len(edge_dict))
        for edge_type, edges in edge_dict.items():
       
            n_edges = len(edges)
            npos, nneg =  int(prop_pos*n_edges), int(prop_neg*n_edges)
            print("Edge Type: {} , total edges: {}, sampling pos links: {}, neg links: {}".format(edge_type,n_edges,npos, nneg))

            rnd_inx = _rnd.choice(len(non_edges), nneg, replace=False)
            neg_edge_list.extend([non_edges[ii] for ii in rnd_inx])

            rnd_pos_inx = _rnd.choice(len(edges), npos, replace=False)
            pos_edge_list.extend([edges[ii] for ii in rnd_pos_inx])

        return pos_edge_list, neg_edge_list


def get_train_test_graph(g, seed):
    """
    Input is graph read at t=0 (DPAH graph)
    
    Return training graph instance and list of pos-neg test edges, and true labels
    """
    pos_edge_list, neg_edge_list = generate_pos_neg_links(g,seed)
    g.remove_edges_from(pos_edge_list)
    edges = pos_edge_list + neg_edge_list
    labels = np.zeros(len(edges))
    labels[:len(pos_edge_list)] = 1
    return g, edges, labels


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
    return precision, recall
