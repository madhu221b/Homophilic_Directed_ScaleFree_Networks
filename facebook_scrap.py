import os
import numpy as np
import pandas as pd
import networkx as nx

ds_path = "./data/facebook"

def gender_extract(feat_arr):
    """
    assuming array size is always 2
    """
    conds = np.where(feat_arr == 1)[0]
    if len(conds) == 0:
        return 0
    else:
        return conds[0]+1

def find_feature_numbers(featfilename, features):
    print("~~~Reading feat file : {}, for features : {}".format(featfilename,features))
    feat2num = dict() # {feature:[no1, no2, ..]}
    featfilepath = os.path.join(ds_path,featfilename)
    with open(featfilepath, "r") as f:
        content = f.readlines()
        for line in content:
            feature = [feature for feature in features if feature in line]
            if feature: feature = feature[0]
            else: continue
            if feature not in feat2num: feat2num[feature] = list()
            
            feature_idx = int(line.split()[0])
            feat2num[feature].append(feature_idx)
    return feat2num, len(content)

def find_idx(item,list_):
    try:
        return list_.index(item)
    except:
        return -1

def read_features(featfile,feat2num,feat_len):
    egonode = int(featfile.split(".")[0])
    featfile = os.path.join(ds_path, featfile)
    egofeatfile = os.path.join(ds_path, str(egonode)+".egofeat")
    featureidxs = feat2num.keys()
    nodedict = dict()
    if egonode not in nodedict: nodedict[egonode] = dict()
    with open(egofeatfile, "r") as fego:
        content_ego = fego.read().split()
        content_ego = np.array([int(item) for item in content_ego])
    with open(featfile, "r") as f:
        content = f.read().split()
        content = np.array([int(item) for item in content])
        start_node = content[0]
        next_idx = 0 
        while next_idx < len(content):
              if next_idx >= len(content): break  
             
              node = content[next_idx] 
              features = np.array(content[next_idx+1:next_idx+1+feat_len])
              if node not in nodedict: nodedict[node] = dict()  
              for feat_name, feat_idxs in feat2num.items():
                  nodedict[node][feat_name] = features[feat_idxs] 
                  if feat_name == "gender": feat_extract_ =  gender_extract
                  nodedict[node][feat_name] = feat_extract_(features[feat_idxs])
                  if egonode in nodedict and feat_name in nodedict[egonode]: pass
                  else: nodedict[egonode][feat_name] = feat_extract_(content_ego[feat_idxs])
              next_idx +=  (feat_len+1)
    return nodedict
              
def read_feat_file(featfile, featfilename, features):
    # find feature numbers
    feat2num, feat_len = find_feature_numbers(featfilename,features)
    node_dict = read_features(featfile, feat2num,feat_len)   
    return node_dict

def read_feature_files(feat_extract=["gender"]):
    """
    Takes a list of features to extract
    Return in format: {"nodeid": {"feature": feature_value}}
    
    
    # convert binary encoded feature in integer class
    """
    all_node_dict = dict()
    feat_files  = [(file, file.split(".")[0]+".featnames") for file in os.listdir(ds_path) if file.endswith(".feat")]
    for feat_file, feat_file_name in feat_files:
        node_dict = read_feat_file(feat_file, feat_file_name, feat_extract)
        all_node_dict.update(node_dict)
    return all_node_dict

def get_edges():
    df = pd.read_csv(
    os.path.join(ds_path, "facebook_combined.txt"),
    sep=" ")
    edges = list(df.itertuples(index=False))
    return edges

def get_graph():
    feats = ["gender"]
    node_dict = read_feature_files(feat_extract=feats)
    edge_data = get_edges()
    node_data = [(node,{"group":dict_[feats[0]]}) for node, dict_ in node_dict.items()]
    g = nx.DiGraph()
    g.add_nodes_from(node_data)
    g.add_edges_from(edge_data)
    return g
    
        