import networkx as nx

EPSILON = 1e-6

def get_common_out_neighbors(g, i, j):
    return set(g.successors(i)).intersection(g.successors(j))

def get_shortest_path_length(g):
    length = dict(nx.all_pairs_dijkstra_path_length(g))
    return length


def get_not_c_indegree_ratio(g,node_attrs,i):
    preds = list(g.predecessors(i))
    c = node_attrs[i]
    num = len([node for node in preds if node_attrs[node] != c])
    denom = len(preds)
    if denom == 0: ratio = 0 
    else: ratio = (num + EPSILON)/denom
   
    return ratio

def get_c_outdegree_ratio(g, node_attrs, i):
    succ = list(g.successors(i))
    c = node_attrs[i]
    num = len([node for node in succ if node_attrs[node] == c])
    denom = len(succ)
    if denom == 0: ratio = 0 
    else: ratio = (num + EPSILON)/denom
    
    return ratio