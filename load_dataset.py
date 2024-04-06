import os
import networkx as nx
from facebook_scrap import get_graph

def get_edge_info(g):
    node_attrs = nx.get_node_attributes(g, "group")
    print("0: ", len([_ for node, _ in node_attrs.items() if _ == 0]))
    print("1: ", len([_ for node, _ in node_attrs.items() if _ == 1]))
    print("2: ", len([_ for node, _ in node_attrs.items() if _ == 2]))
    count_dict = dict()
    for edge in g.edges():
        u_id, v_id = node_attrs[edge[0]], node_attrs[edge[1]]
        label = "{}->{}".format(u_id,v_id)
        if label not in count_dict: count_dict[label] = 0
        count_dict[label] += 1

    print(count_dict)

def load_rice():
    """
    Group 0: Age is 18 or 19
    Group 1: Age is 20
    """
    dataset_path = "./data/rice"
    node_attr_file = os.path.join(dataset_path,"rice_subset.attr")
    edge_file = os.path.join(dataset_path,"rice_subset.links")
    
    node_data, edge_data = list(), list()
    with open(node_attr_file, 'r') as fin:
        for line in fin:
            s = line.split()
            item = (s[0], {"group":int(s[1])})
            node_data.append(item)
    
    with open(edge_file, 'r') as fin:
        for line in fin:
            u,v = line.split()
            edge_data.append((u,v))
            


    print(len(node_data), len(edge_data))
    g = nx.DiGraph()
    g.add_nodes_from(node_data)
    g.add_edges_from(edge_data)
    return g

def load_twitter():
    dataset_path = "./data/twitter"
    node_attr_file = os.path.join(dataset_path,"sample_4000.attr")
    edge_file = os.path.join(dataset_path,"sample_4000.links")
    
    node_data, edge_data = list(), list()
    with open(node_attr_file, 'r') as fin:
        for line in fin:
            s = line.split()
            item = (s[0], {"group":int(s[1])})
            node_data.append(item)
    
    with open(edge_file, 'r') as fin:
        for line in fin:
            line = line.replace("[","").replace("]","").replace(",","").strip()
            u,v,w1, w2 = line.split()
            edge_data.append((u,v,float(w1)))
            edge_data.append((v,u,float(w2)))
    print(len(node_data), len(edge_data))
    g = nx.DiGraph()
    g.add_nodes_from(node_data)
    # g.add_edges_from(edge_data)
    g.add_weighted_edges_from(edge_data)
    return g

def load_facebook(features=["gender"]):
    """
    Assuming only one attribute for now in features arr
    """
    dataset_path = "./data/facebook"
    g = get_graph()
    return g   

def load_dataset(name):
    if name == "rice":
        g = load_rice()
    elif name == "twitter":
        g = load_twitter()
    elif name == "facebook":
        g = load_facebook()
        
    return g

if __name__ == "__main__":
    g = load_dataset(name="facebook")
    get_edge_info(g)