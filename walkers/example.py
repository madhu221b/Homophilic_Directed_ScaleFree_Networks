
import numpy as np
import networkx as nx
import pandas as pd
import time

from degreewalker import DegreeWalker
from indegreewalker import InDegreeWalker
from commonnghaware import CommonNeighborWalker
from levywalker import LevyWalker
from fairindegreewalker import FairInDegreeWalker
from indegreevarybetawalker import InDegreeVaryBetaWalker

# path = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/DPAH_trial/DPAH-N100-fm0.3-d0.03-ploM2.5-plom2.5-hMM0.8-hmm0.8-ID0.gpickle"
path = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/DPAH_trial/DPAH-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM0.0-hmm0.0-ID0.gpickle"
graph = nx.read_gpickle(path)
node2group = {node:graph.nodes[node]["m"] for node in graph.nodes()}
nx.set_node_attributes(graph, node2group, 'group')

DIM = 64
WALK_LEN = 10
NUM_WALKS = 200

start_time = time.time()
walker = InDegreeVaryBetaWalker(graph, dimensions=DIM, walk_len=WALK_LEN, num_walks=NUM_WALKS, workers=8)
# walker = LevyWalker(graph,alpha=1,workers=8,dimensions=64)
# walker = FairInDegreeWalker(graph, beta=-1, workers=8,dimensions=64)
model = walker.fit() 
end_time = time.time()
print("Time taken: ", (end_time-start_time))
print("InDegree Vary Beta walks:",  walker.walks, file=open("walks.txt","a"))
emb_df = (pd.DataFrame([model.wv.get_vector(str(n)) for n in graph.nodes()], index = graph.nodes))
print("embedding shape: ", emb_df.shape)