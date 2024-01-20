import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import networkx as nx

color_dict = {'min':'#ec8b67', 'maj':'#6aa8cb'}
path = "/home/mpawar/Homophilic_Directed_ScaleFree_Networks/DPAH/DPAH-N50-fm0.3-d0.03-ploM2.5-plom2.5-hMM0.8-hmm0.8-ID0.gpickle"
fig, ax = plt.subplots() 
g = nx.read_gpickle(path)
node_color = [color_dict['min'] if obj['m'] else color_dict['maj'] for n,obj in g.nodes(data=True)]
# nx.draw_networkx_edges(g,pos,alpha=0.2)

# nx.draw(g, pos,node_color=node_color)
pos = nx.spring_layout(g, k=12 / math.sqrt(g.order()), seed=42)
# nx.draw_networkx_edges(g,pos,alpha=0.2)
# nx.draw_networkx_nodes(g,pos=pos,node_color=node_color,edgecolors="black",linewidths=0.2)
nx.draw_networkx_nodes(g, pos=pos, node_color=node_color,edgecolors="black",linewidths=0.2)
nx.draw_networkx_edges(g,pos,alpha=0.2)
plt.savefig('graph.png', dpi=300, bbox_inches='tight')