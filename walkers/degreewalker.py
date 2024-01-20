import networkx as nx
import numpy as np
from .walker import Walker

class DegreeWalker(Walker):
    def __init__(self,graph,beta=0,workers=1,dimensions=64,walk_len=10,num_walks=200):
        print("Degree Walker with beta = ", beta)
        super().__init__(graph, workers=workers,dimensions=dimensions,walk_len=walk_len,num_walks=num_walks)
        """pi i-> j =  A_ij (q_j^beta) / sum l=1 ^ N Ail q_l^beta """
        self.graph = self.graph.to_undirected()
        self.number_of_nodes = self.graph.number_of_nodes()
        """
        pi_(i->j) =  
         [[0,0, 0,1, ....., 0,j],
          [1,0, 1,1, ....., 1,j],
          ..
          ..
          [i,0, i,1, ..... i,j]]
          (i spans along rows, j spans along columns)
        """
        # Transition Prs matrix
        self.pi = np.zeros((self.number_of_nodes, self.number_of_nodes))
        self.A = nx.to_numpy_array(self.graph, nodelist=sorted(self.graph.nodes())) 
        degree = dict(nx.degree(self.graph))
        self.degree_pow = dict({node: (np.round(degree**beta,5) if degree != 0 else 0) for node, degree in degree.items()})
                
        # compute probabilities
        print("!!!!  Computing Probability Matrix")
        self._precompute_probabilities()
        print("!!!!  Generate Walks")
        self._generate_walks(self.graph, self.pi, type="local")


    def _precompute_probabilities(self):
        for i in sorted(self.graph.nodes()):
            # we traverse neighbours only because for non neighbours this value should be zero
            # according to formula 
            neighbors = np.array(np.nonzero(self.A[i] !=  0))
            _sum = sum([v for k,v in self.degree_pow.items() if k in neighbors])
            if _sum != 0: # denominator is non zero
               for (_,j) in np.ndenumerate(neighbors):
                   num = self.degree_pow[j]
                   self.pi[i][j] = num/_sum
                   


 
