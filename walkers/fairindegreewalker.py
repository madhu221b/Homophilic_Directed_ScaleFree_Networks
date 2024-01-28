import networkx as nx
import numpy as np

try:
  from .walker import Walker
except Exception as error:
    from walker import Walker

class FairInDegreeWalker(Walker):
    def __init__(self,graph,beta=0,workers=1,dimensions=64,walk_len=10,num_walks=200):
        print("Fair In Degree Walker with beta = ", beta)
        super().__init__(graph, workers=workers,dimensions=dimensions,walk_len=walk_len,num_walks=num_walks)
        """pi i-> j =  A_ij (q_j_indegree^beta) / sum l=1 ^ N Ail q_l_indegree^beta """
        self.number_of_nodes = self.graph.number_of_nodes()

        # Transition Prs matrix
        # self.pi = np.zeros((self.number_of_nodes, self.number_of_nodes))
        
        self.node2groups = nx.get_node_attributes(self.graph, "group")
        self.groups = np.unique(list(self.node2groups.values()))
        
        # initialise d_graph
        self.d_graph = dict()
        for node in self.graph.nodes():
            self.d_graph[node] = {"pr":dict(), "ngh":dict()}
            for group in self.groups:
               self.d_graph[node]["pr"][group] = list()
               self.d_graph[node]["ngh"][group] = list()


        degree = dict(self.graph.in_degree()) # note now it is indegree
        self.degree_pow = dict({node: (np.round(degree**beta,5) if degree != 0 else 0) for node, degree in degree.items()})
    
        # compute probabilities
        print("!!!!  Computing Probability Matrix")
        self._precompute_probabilities()
        print("!!!!  Generate Walks")
        self._generate_walks(self.graph, self.d_graph, type="fair_local")


    def _precompute_probabilities(self):
        for i in sorted(self.graph.nodes()):
            
            # find unnormalized weights
            neighbors = list(self.graph.successors(i))
            for j in neighbors:  
                identity = self.node2groups[j]              
                unnormalized_wgt = self.degree_pow[j]
                self.d_graph[i]["pr"][identity].append(unnormalized_wgt)
                self.d_graph[i]["ngh"][identity].append(j)                   
           
            # calculated normalizec weights
            for group in self.groups:
                unnormalized_wgts = self.d_graph[i]["pr"][group]
                sum_ = sum(unnormalized_wgts)
                self.d_graph[i]["pr"][group] = np.array(unnormalized_wgts)/sum_



 
