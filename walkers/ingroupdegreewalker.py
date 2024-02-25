import networkx as nx
import numpy as np

try:
  from .walker import Walker
  from .utils import get_not_c_indegree_ratio, get_c_outdegree_ratio
except Exception as error:
    from walker import Walker
    from utils import get_not_c_indegree_ratio, get_c_outdegree_ratio

class InGroupDegreeWalker(Walker):
    def __init__(self,graph,beta=0,workers=1,dimensions=64,walk_len=10,num_walks=200):
        print("In Group Degree Walker")
        super().__init__(graph, workers=workers,dimensions=dimensions,walk_len=walk_len,num_walks=num_walks)
       
        self.number_of_nodes = self.graph.number_of_nodes()
        self.node_attrs = nx.get_node_attributes(graph, "group")
        self.scores = dict()
        # Transition Prs matrix
        # self.pi = np.zeros((self.number_of_nodes, self.number_of_nodes))
        self.d_graph = {node: {"pr":list(), "ngh":list()} for node in self.graph.nodes()}
        
        print("!!!!!! Computing out-flow score of information for every node")
        self._compute_information_score()
        # compute probabilities
        print("!!!!  Computing Probability Matrix")
        self._precompute_probabilities()
        print("!!!!  Generate Walks")
        self._generate_walks(self.graph, self.d_graph, type="local")

    
    def _compute_information_score(self):
        for i in self.graph.nodes():
           in_ratio =  get_not_c_indegree_ratio(self.graph,self.node_attrs,i)
           out_ratio =  get_c_outdegree_ratio(self.graph,self.node_attrs,i)
           self.scores[i] = in_ratio*out_ratio



    def _precompute_probabilities(self):
        for i in self.graph.nodes():
            # we traverse neighbours only because for non neighbours this value should be zero
            # according to formula 
            neighbors = list(self.graph.successors(i))
            _sum = sum([v for k,v in self.scores.items() if k in neighbors])
            if _sum != 0: # denominator is non zero
               for (_,j) in np.ndenumerate(neighbors):                
                    num = self.scores[j]
                    pr = num/_sum
                    self.d_graph[i]["pr"].append(pr)
                    self.d_graph[i]["ngh"].append(j)                   


 
