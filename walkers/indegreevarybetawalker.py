from collections import Counter

import networkx as nx
import numpy as np

try:
  from .walker import Walker
except Exception as error:
    from walker import Walker

class InDegreeVaryBetaWalker(Walker):
    def __init__(self,graph,beta=0,workers=1,dimensions=64,walk_len=10,num_walks=200):
        print("In Degree Walker with varying beta")
        super().__init__(graph, workers=workers,dimensions=dimensions,walk_len=walk_len,num_walks=num_walks)
        """pi i-> j =  A_ij (q_j_indegree^beta) / sum l=1 ^ N Ail q_l_indegree^beta """
        self.number_of_nodes = self.graph.number_of_nodes()
        self.node_attrs = nx.get_node_attributes(graph, "group")
        # Transition Prs matrix
        # self.pi = np.zeros((self.number_of_nodes, self.number_of_nodes))
        self.d_graph = {node: {"pr":list(), "ngh":list()} for node in self.graph.nodes()}
        self.fm_dict = self.get_fm_from_graph(self.graph)
        degree = dict(self.graph.in_degree()) # note now it is indegree
        offset = 1/len(self.fm_dict) # 0.5 for 2 groups
  
        self.degree_pow = dict({node:(degree**(self.fm_dict[self.node_attrs[node]] - offset) if degree != 0 else 0) 
                           for node, degree in degree.items()})

        # compute probabilities
        print("!!!!  Computing Probability Matrix")
        self._precompute_probabilities()
        print("!!!!  Generate Walks")
        self._generate_walks(self.graph, self.d_graph, type="local")


    def _precompute_probabilities(self):
        for i in self.graph.nodes():
            # we traverse neighbours only because for non neighbours this value should be zero
            # according to formula 
            neighbors = list(self.graph.successors(i))
            _sum = sum([v for k,v in self.degree_pow.items() if k in neighbors])
            if _sum != 0: # denominator is non zero
               for (_,j) in np.ndenumerate(neighbors):                
                    num = self.degree_pow[j]
                    pr = num/_sum
                    self.d_graph[i]["pr"].append(pr)
                    self.d_graph[i]["ngh"].append(j)  


    def get_fm_from_graph(self, graph):
       count_dict = Counter(self.node_attrs.values())
       fm_sum = sum(count_dict.values())
       count_dict = {i: fm/fm_sum for i, fm in count_dict.items()}
       return count_dict

 
