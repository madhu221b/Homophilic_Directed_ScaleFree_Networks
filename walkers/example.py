
import numpy as np
import networkx as nx
import pandas as pd

from degreewalker import DegreeWalker

path = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/DPAH/DPAH-N50-fm0.3-d0.03-ploM2.5-plom2.5-hMM0.8-hmm0.8-ID0.gpickle"
graph = nx.read_gpickle(path)

walker = DegreeWalker(graph, beta=-1, workers=8,dimensions=64)
model = walker.fit() 
emb_df = (pd.DataFrame([model.wv.get_vector(str(n)) for n in graph.nodes()], index = graph.nodes))
print("embedding shape: ", emb_df.shape)