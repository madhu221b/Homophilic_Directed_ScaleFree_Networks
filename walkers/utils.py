import networkx as nx

def get_common_out_neighbors(g, i, j):
    return set(g.successors(i)).intersection(g.successors(j))

def get_shortest_path_length(g):
    length = dict(nx.all_pairs_dijkstra_path_length(g))
    return length