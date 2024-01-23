import networkx as nx
import numpy as np

try:
  from .walker import Walker
  from .utils import get_common_out_neighbors
except Exception as error:
    from walker import Walker
    from utils import get_common_out_neighbors



class CommonNeighborWalker(Walker):
    def __init__(self,graph,workers=1,dimensions=64,walk_len=10,num_walks=200):
        print("Common Neigbor Awareness Random Walk")
        super().__init__(graph, workers=workers,dimensions=dimensions,walk_len=walk_len,num_walks=num_walks)
        """pi i-> j =  1 - common_nghs/min(deg(i),deg(j))"""
        self.number_of_nodes = self.graph.number_of_nodes()

        # Transition Prs matrix
        # self.pi = np.zeros((self.number_of_nodes, self.number_of_nodes))
        self.d_graph = {node: {"pr":list(), "ngh":list()} for node in self.graph.nodes()}
        self.degree = dict(self.graph.out_degree()) # note now it is out degree (?)
       
        # compute probabilities
        print("!!!!  Computing Probability Matrix")
        self._precompute_probabilities()
        print("!!!!  Generate Walks")
        self._generate_walks(self.graph, self.d_graph, type="local")


    def _precompute_probabilities(self):
        for i in sorted(self.graph.nodes()):
            # we traverse neighbours only because for non neighbours this value should be zero
            # according to formula 
            
            for j in self.graph.neighbors(i):
                c_ij = len(get_common_out_neighbors(self.graph,i,j))
                min_deg = min(self.degree[i], self.degree[j])
                if min_deg != 0:
                   pr = 1 - c_ij/min_deg
                   self.d_graph[i]["pr"].append(pr)
                   self.d_graph[i]["ngh"].append(j)                   
            
            # normalize
            self.d_graph[i]["pr"] = np.array(self.d_graph[i]["pr"])/sum(self.d_graph[i]["pr"])

 
