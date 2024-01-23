import networkx as nx
import numpy as np

try:
  from .walker import Walker
  from .utils import get_shortest_path_length
except Exception as error:
    from walker import Walker
    from utils import get_shortest_path_length


class LevyWalker(Walker):
    def __init__(self,graph,alpha=1,workers=1,dimensions=64,walk_len=10,num_walks=200):
        print("Levy Random Walk with alpha: ", alpha)
        super().__init__(graph, workers=workers,dimensions=dimensions,walk_len=walk_len,num_walks=num_walks)
        """pi i-> j =  d_ij ** -alpha / sum all nodes l except i d_il ** -alpha  """
        self.number_of_nodes = self.graph.number_of_nodes()

        # Transition Prs matrix
        self.d_graph = {node: {"pr":list(), "ngh":list()} for node in self.graph.nodes()}
        
        length_graph = get_shortest_path_length(self.graph)
        self.length_pow = {node: {sub_node: (np.round(val**-alpha,5) if val != 0 else 0) for sub_node, val in sub_dict.items()} for node, sub_dict in length_graph.items()}
   
        # compute probabilities
        print("!!!!  Computing Probability Matrix")
        self._precompute_probabilities()
        print("!!!!  Generate Walks")
        self._generate_walks(self.graph, self.d_graph, type="local")


    def _precompute_probabilities(self):
        for i in sorted(self.graph.nodes()):
            # we traverse neighbours only because for non neighbours this value should be zero
            # according to formula 
            
            sum_ = sum(self.length_pow[i].values())
            for j in self.length_pow[i].keys():
                if i == j: continue
                num = self.length_pow[i][j]
                pr = num/sum_
                self.d_graph[i]["pr"].append(pr)
                self.d_graph[i]["ngh"].append(j)                   
            

