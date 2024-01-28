
import numpy as np
import networkx as nx
import pandas as pd

from degreewalker import DegreeWalker
from indegreewalker import InDegreeWalker
from commonnghaware import CommonNeighborWalker
from levywalker import LevyWalker
from fairindegreewalker import FairInDegreeWalker
path = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/DPAH/DPAH-N50-fm0.3-d0.03-ploM2.5-plom2.5-hMM0.8-hmm0.8-ID0.gpickle"

graph = nx.read_gpickle(path)
node2group = {node:graph.nodes[node]["m"] for node in graph.nodes()}
nx.set_node_attributes(graph, node2group, 'group')

# walker = InDegreeWalker(graph, beta=-1, workers=8,dimensions=64)
# walker = LevyWalker(graph,alpha=1,workers=8,dimensions=64)
walker = FairInDegreeWalker(graph, beta=-1, workers=8,dimensions=64)
model = walker.fit() 
emb_df = (pd.DataFrame([model.wv.get_vector(str(n)) for n in graph.nodes()], index = graph.nodes))
print("embedding shape: ", emb_df.shape)