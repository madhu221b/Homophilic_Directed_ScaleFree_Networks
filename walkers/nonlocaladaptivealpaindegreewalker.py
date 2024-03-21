from collections import Counter
import networkx as nx
import random
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

"""
The non-local pr for jumps is dependent on node identity. And not on individual node's
local metrics. 
"""
try:
  from .walker import Walker
except Exception as error:
    from walker import Walker

class NonLocalAdaptiveInDegreeWalker(Walker):
    def __init__(self,graph,beta=0,workers=1,dimensions=64,walk_len=10,num_walks=200):
        print("Non Local Adaptive In Degree Walker with beta = {} ".format(beta))
        super().__init__(graph, workers=workers,dimensions=dimensions,walk_len=walk_len,num_walks=num_walks)

        self.number_of_nodes = self.graph.number_of_nodes()
        self.node_attrs = nx.get_node_attributes(graph, "group")

        # Populate nodes by group
        self._get_group_to_node_dict()
        
        # Transition Prs matrix
        # self.pi = np.zeros((self.number_of_nodes, self.number_of_nodes))
        walk_types =  ["local","nonlocal"]
        self.d_graph = dict()
        
        
        for node in self.graph.nodes():
            self.d_graph[node] = dict()
            for w_type in walk_types:
                self.d_graph[node][w_type] = {"pr":list(), "ngh":list()}
    
        degree = dict(self.graph.in_degree()) # note now it is indegree
        self.indegree_df = pd.DataFrame.from_dict(degree, orient='index', columns=['degree'])
        degree_pow = dict({node: (np.round(degree**beta,5) if degree != 0 else 0) for node, degree in degree.items()})
        
        self.degree_pow_df = pd.DataFrame.from_dict(degree_pow, orient='index', columns=['degree_pow'])


        # compute probabilities
        print("!!!! Computing non-local jump probability")
        self.walk_alpha_pr = dict()
        self._precompute_alpha()

        print("!!!!  Computing Probability Matrix")
        self._precompute_probabilities()
        print("!!!!  Generate Walks")
        self._generate_walks(self.graph, self.d_graph, self.walk_alpha_pr)
    

    def _get_group_to_node_dict(self):
        self.group2node = dict()
        for node, node_id in self.node_attrs.items():
            if node_id not in self.group2node:
                self.group2node[node_id] = list()
            self.group2node[node_id].append(node)

    def avg_indegree_due_to_grp(self, grp):
        g = self.graph
        itr = [node for node, _ in self.node_attrs.items() if _ == grp]
        total_len = len(itr)
        sum_ = 0
        for i in itr:      
            neighbors = list(g.predecessors(i))
            diff_nghs = len([ngh for ngh in neighbors if self.node_attrs[ngh] != grp])
            sum_ += diff_nghs
        avg_indg = sum_/total_len
        return avg_indg
     
    def _precompute_alpha(self):
        uniquegroups = self.group2node.keys()
        group2alpha = dict()
        epsilon = 1e-6

        for uniquegroup in uniquegroups:
            group2alpha[uniquegroup] = self.avg_indegree_due_to_grp(uniquegroup)
        unnormalized_prs = {k:1/(v+epsilon) for k,v in group2alpha.items()}
        sum_ = sum(unnormalized_prs.values())
        group2alpha = {k:v/sum_ for k,v in unnormalized_prs.items()}
        print("Group2Alpha: ", group2alpha)
        for i in self.graph.nodes():
            self.walk_alpha_pr[i] = group2alpha[self.node_attrs[i]]

    def _get_non_local_successors(self, node, successors):
       """
        Sampling with 
        max_val = max(group_size of nghs)
        sample size = max val - group size of group i in ngh
        Design choice - nghs picked at random 

        this wont work.

        Trying another approach - degree wise
       """
       k = 5
       
       predecessors = self.graph.predecessors(node)
       non_local_nodes = []
        
       all_nodes = self.group2node[self.node_attrs[node]]
       all_nodes = list(set(all_nodes) - set(set(predecessors) | set(successors) | set([node])))
       sample_size = min(len(successors),len(all_nodes))
       if sample_size == 0: sample_size = k
           
       unnormalized_prs = self.degree_pow_df.loc[all_nodes, "degree_pow"]
       unnormalized_prs += 1e-6
       _sum = unnormalized_prs.sum()
 
       prs = unnormalized_prs/_sum
       non_local_choice = np.random.choice(all_nodes, size=sample_size, p=prs, replace=False)
       non_local_nodes.extend(non_local_choice)
        

       return non_local_nodes

    def _precompute_probabilities(self):
        for i in self.graph.nodes():
            local_neighbors = list(self.graph.successors(i))
            non_local_neighbors = self._get_non_local_successors(i, local_neighbors)

            unnormalized_prs_local = self.degree_pow_df.loc[local_neighbors, "degree_pow"]
            unnormalized_prs_nonlocal = self.degree_pow_df.loc[non_local_neighbors, "degree_pow"]
                        
            if len(local_neighbors) != 0:
                _sum = unnormalized_prs_local.sum()
                if _sum == 0:  
                    unnormalized_prs_local = unnormalized_prs_local + 1e-6
                    _sum = unnormalized_prs_local.sum()

                prs = unnormalized_prs_local/_sum
                self.d_graph[i]["local"]["pr"] = list(prs)
                self.d_graph[i]["local"]["ngh"] = local_neighbors

            if len(non_local_neighbors) != 0:
                _sum = unnormalized_prs_nonlocal.sum()
                if _sum == 0: 
                    unnormalized_prs_nonlocal = unnormalized_prs_nonlocal + 1e-6
                    _sum = unnormalized_prs_nonlocal.sum()

                prs = unnormalized_prs_nonlocal/_sum
                self.d_graph[i]["nonlocal"]["pr"] = list(prs)
                self.d_graph[i]["nonlocal"]["ngh"] = non_local_neighbors



    def _generate_walks(self, graph, d_graph, walk_alpha_pr) -> list:
        """
        Generates the random walks which will be used as the skip-gram input.
        :return: List of walks. Each walk is a list of nodes.
        """

        flatten = lambda l: [item for sublist in l for item in sublist]
       
        # Split num_walks for each worker
        num_walks_lists = np.array_split(range(self.num_walks), self.workers)
        
        parallel_generate_walks = self.local_generate_walk

        walk_results = Parallel(n_jobs=self.workers)(
            delayed(parallel_generate_walks)(graph, d_graph,walk_alpha_pr, idx, len(num_walks))
                                        for idx, num_walks
            in enumerate(num_walks_lists, 1))

        walks = flatten(walk_results)
        self.walks = walks

    def local_generate_walk(self, graph, d_graph, walk_alpha_pr, cpu_num, num_walks):
        walks = list()
        pbar = tqdm(total=num_walks, desc='Generating walks (CPU: {})'.format(cpu_num))
        possible_walks = ["local", "nonlocal"]
        # walks_pr = [1-alpha, alpha] # pr of selecting high degree nodes, low degree nodes
        
        for n_walk in range(num_walks):
            # random_group = np.random.choice(possible_walks, p=walks_pr, size=1)[0]
            pbar.update(1)

            shuffled_nodes = list(graph.nodes())
            random.shuffle(shuffled_nodes)
         
            # Start a random walk from every node
            for source in shuffled_nodes:
  
                walk = [source]
                alpha = walk_alpha_pr[source]
                walks_pr = [1-alpha, alpha]
                random_group = np.random.choice(possible_walks, p=walks_pr, size=1)[0]
                
                while len(walk) < self.walk_len:
                       last_node = walk[-1]
                       walkgroups = [group for group in possible_walks if len(d_graph[last_node][group]["pr"]) > 0]
                       
                       
                       if random_group not in walkgroups: break

                       walk_options = list(d_graph[last_node][random_group]["ngh"])
                       probabilities = d_graph[last_node][random_group]["pr"]
                       next_node = np.random.choice(walk_options, size=1, p=probabilities)[0]
                       walk.append(next_node)

                walk = list(map(str, walk))  # Convert all to strings
                walks.append(walk)

    
        pbar.close()
        return walks

 
        


 
