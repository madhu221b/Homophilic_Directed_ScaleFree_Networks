import networkx as nx
import numpy as np

try:
  from .walker import Walker
except Exception as error:
    from walker import Walker

class DegreeWalker(Walker):
    def __init__(self,graph,beta=0,workers=1,dimensions=64,walk_len=10,num_walks=200):
        print("Degree Walker with beta = ", beta)
        super().__init__(graph, workers=workers,dimensions=dimensions,walk_len=walk_len,num_walks=num_walks)
        """pi i-> j =  A_ij (q_j^beta) / sum l=1 ^ N Ail q_l^beta """
        self.graph = self.graph.to_undirected()
        self.number_of_nodes = self.graph.number_of_nodes()

        # Transition Prs matrix
        # self.pi = np.zeros((self.number_of_nodes, self.number_of_nodes))
        self.d_graph = {node: {"pr":list(), "ngh":list()} for node in self.graph.nodes()}
        self.A = nx.to_numpy_array(self.graph, nodelist=sorted(self.graph.nodes())) 
        degree = dict(nx.degree(self.graph))
        self.degree_pow = dict({node: (np.round(degree**beta,5) if degree != 0 else 0) for node, degree in degree.items()})
    
        # compute probabilities
        print("!!!!  Computing Probability Matrix")
        self._precompute_probabilities()
        print("!!!!  Generate Walks")
        self._generate_walks(self.graph, self.d_graph, type="local")


    def _precompute_probabilities(self):
        for i in sorted(self.graph.nodes()):
            # we traverse neighbours only because for non neighbours this value should be zero
            # according to formula 
            neighbors = np.array(np.nonzero(self.A[i] !=  0))
            _sum = sum([v for k,v in self.degree_pow.items() if k in neighbors])
            if _sum != 0: # denominator is non zero
               for (_,j) in np.ndenumerate(neighbors):
                    num = self.degree_pow[j]
                    pr = num/_sum
                    self.d_graph[i]["pr"].append(pr)
                    self.d_graph[i]["ngh"].append(j)                   


 
