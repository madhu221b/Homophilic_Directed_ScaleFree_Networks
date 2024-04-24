import networkx as nx
import numpy as np
import random
from joblib import Parallel, delayed
from tqdm import tqdm

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
        
        self.node_attrs = nx.get_node_attributes(self.graph, "group")
        self.groups = np.unique(list(self.node_attrs.values()))
        
        # initialise d_graph
        self.d_graph = dict()
        for node in self.graph.nodes():
            self.d_graph[node] = {"pr":dict(), "ngh":dict()}
            for group in self.groups:
               self.d_graph[node]["pr"][group] = list()
               self.d_graph[node]["ngh"][group] = list()
               

        degree = dict(self.graph.in_degree()) # note now it is indegree
        self.degree_pow = dict({node: (np.round(degree**beta,5) if degree != 0 else 0) for node, degree in degree.items()})
        
        print("!!! Computing Edge Dict")
        self.ratio_dict = self._get_edge_dict()
        print("Ratio Dict: ", self.ratio_dict)
        
        # compute probabilities
        print("!!!!  Computing Probability Matrix")
        self._precompute_probabilities()
        print("!!!!  Generate Walks")
        self._generate_walks(self.graph, self.d_graph,self.ratio_dict)

    
    def _get_edge_dict(self):
        g = self.graph
        node_attrs = self.node_attrs
        edge_dict = dict()

        for u,v in g.edges():
            label = "{}->{}".format(node_attrs[u],node_attrs[v])
            if label not in edge_dict: edge_dict[label] = 0
            edge_dict[label] += 1
        
        print("edge dict: ", edge_dict)
        ratio_dict = dict()
        for u_group in self.groups:
            for v_group in self.groups:
                label = "{}->{}".format(u_group,v_group)
                num = edge_dict[label]
                if u_group not in ratio_dict: ratio_dict[u_group] = {"pr":[],"groups":[]}
                ratio_dict[u_group]["pr"].append(num)
                ratio_dict[u_group]["groups"].append(v_group)
        
        # normalise
        for u_group in ratio_dict:
            items = ratio_dict[u_group]["pr"]
            ratios = np.array(items)/np.sum(items)
            inv_ratios = 1/ratios
            prs = inv_ratios/np.sum(inv_ratios)
            ratio_dict[u_group]["pr"] = prs
        return ratio_dict

    def _precompute_probabilities(self):
        
        for i in sorted(self.graph.nodes()):
            
            # find unnormalized weights
            neighbors = list(self.graph.successors(i))
            for j in neighbors:  
                identity = self.node_attrs[j]            
                unnormalized_wgt = self.degree_pow[j]
                self.d_graph[i]["pr"][identity].append(unnormalized_wgt)
                self.d_graph[i]["ngh"][identity].append(j)                   
           
            # calculated normalizec weights
            for group in self.groups:
                unnormalized_wgts = self.d_graph[i]["pr"][group]
                sum_ = sum(unnormalized_wgts)
                self.d_graph[i]["pr"][group] = np.array(unnormalized_wgts)/sum_

    def _generate_walks(self, graph, d_graph, ratio_dict, type="local") -> list:
        """
        Generates the random walks which will be used as the skip-gram input.
        :return: List of walks. Each walk is a list of nodes.
        """

        flatten = lambda l: [item for sublist in l for item in sublist]
       
        # Split num_walks for each worker
        num_walks_lists = np.array_split(range(self.num_walks), self.workers)
        parallel_generate_walks = self.local_generate_walk

        walk_results = Parallel(n_jobs=self.workers)(
            delayed(parallel_generate_walks)(graph, d_graph,ratio_dict, idx, len(num_walks))
                                        for idx, num_walks
            in enumerate(num_walks_lists, 1))

        walks = flatten(walk_results)
        self.walks = walks

    def local_generate_walk(self, graph, d_graph,ratio_dict, cpu_num, num_walks):
        walks = list()
        pbar = tqdm(total=num_walks, desc='[Fair local] Generating walks (CPU: {})'.format(cpu_num))

        for n_walk in range(num_walks):
  
            pbar.update(1)

            shuffled_nodes = list(graph.nodes())
            random.shuffle(shuffled_nodes)

            # Start a random walk from every node
            for source in shuffled_nodes:
  
                walk = [source]
                while len(walk) < self.walk_len:
                       last_node = walk[-1]
                       walk_options = d_graph[last_node]["ngh"]
                    #    all_possible_groups = list(walk_options.keys())

                       # note this is where we choose a group randomly
                       all_possible_groups = ratio_dict[self.node_attrs[last_node]]["groups"]
                       all_possible_prs = ratio_dict[self.node_attrs[last_node]]["pr"]
                       random_group = np.random.choice(all_possible_groups,p=all_possible_prs,size=1)[0]
                       walk_options = walk_options[random_group]
                       probabilities = d_graph[last_node]["pr"][random_group]
  
                       if len(probabilities) == 0: break  # skip nodes with no ngs
                       next_node = np.random.choice(walk_options, size=1, p=probabilities)[0]
                       walk.append(next_node)

                walk = list(map(str, walk))  # Convert all to strings
                walks.append(walk)

    
        pbar.close()
        return walks 



 
