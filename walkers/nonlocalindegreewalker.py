from collections import Counter
import networkx as nx
import random
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

try:
  from .walker import Walker
except Exception as error:
    from walker import Walker

class NonLocalInDegreeWalker(Walker):
    def __init__(self,graph,beta=0,alpha=0,workers=1,dimensions=64,walk_len=10,num_walks=200):
        print("Non Local In Degree Walker with beta = {}, alpha = {} ".format(beta,alpha))
        super().__init__(graph, workers=workers,dimensions=dimensions,walk_len=walk_len,num_walks=num_walks)
        """pi i-> j =  A_ij (q_j_indegree^beta) / sum l=1 ^ N Ail q_l_indegree^beta """
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
        print("!!!!  Computing Probability Matrix")
        self._precompute_probabilities()
        print("!!!!  Generate Walks")
        self._generate_walks(self.graph, self.d_graph, alpha)
    

    
    def _get_group_to_node_dict(self):
        self.group2node = dict()
        for node, node_id in self.node_attrs.items():
            if node_id not in self.group2node:
                self.group2node[node_id] = list()
            self.group2node[node_id].append(node)

    def _get_avgdegree_by_group(self,successors):
        grouptodegree = dict()
        for node in successors:
            node_id = self.node_attrs[node]
            if node_id not in grouptodegree: grouptodegree[node_id] = list()
            ind = self.indegree_df.loc[node_id,"degree"]
            grouptodegree[node_id].append(ind)
        
        grouptodegree = {k:avg(v) for k,v in grouptodegree.items()}
        return group2degree

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
        
    #    ids = [self.node_attrs[node] for node in successors]
    #    count_dict = Counter(ids)
    #    count_dict = {k:(count_dict[k] if k in count_dict else 0) for k,_ in self.group2node.items()}
    #    max_val = max(count_dict.values())


    #    for group, count in count_dict.items():
    #        delta_val = (max_val - count)
    #        if max_val == 0: sample_size = k
    #        else: sample_size = delta_val

    #        all_nodes = self.group2node[group]
    #        all_nodes = list(set(all_nodes) - set(set(predecessors) | set(successors) | set([node])))
    #        sample_size = min(sample_size,len(all_nodes))
    #        unnormalized_prs = self.degree_pow_df.loc[all_nodes, "degree_pow"]
    #        unnormalized_prs += 1e-6
    #        _sum = unnormalized_prs.sum()

    #        prs = unnormalized_prs/_sum
    #        non_local_choice = np.random.choice(all_nodes, size=sample_size, p=prs, replace=False)
    #        non_local_nodes.extend(non_local_choice)
        
    #    random.shuffle(non_local_nodes)
    #    node_ids = [self.node_attrs[node] for node in non_local_nodes]
    #    node_ids_succ = [self.node_attrs[node] for node in successors]
    #    print("~~~~~~~ for node identity: {}, local ids: {} , non local ids: {}".format(self.node_attrs[node],node_ids_succ,node_ids),file=open("output.txt","a"))
    #    non_local_majids = [_ for _ in successors if self.node_attrs[_] == 0]
    #    non_local_minids = [_ for _ in successors if self.node_attrs[_] == 1]
    #    if non_local_minids: avg_min_degree = self.indegree_df.loc[non_local_minids,"degree"].mean()
    #    else: avg_min_degree = "null" 
    #    if non_local_majids: avg_maj_degree = self.indegree_df.loc[non_local_majids,"degree"].mean()
    #    else: avg_maj_degree = "null"
    #    print("non local ids: avg min: {}, avg maj:{}".format(avg_min_degree,avg_maj_degree),file=open("output.txt","a"))
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



    def _generate_walks(self, graph, d_graph, alpha) -> list:
        """
        Generates the random walks which will be used as the skip-gram input.
        :return: List of walks. Each walk is a list of nodes.
        """

        flatten = lambda l: [item for sublist in l for item in sublist]
       
        # Split num_walks for each worker
        num_walks_lists = np.array_split(range(self.num_walks), self.workers)
        
        parallel_generate_walks = self.local_generate_walk

        walk_results = Parallel(n_jobs=self.workers)(
            delayed(parallel_generate_walks)(graph, d_graph, idx, len(num_walks), alpha)
                                        for idx, num_walks
            in enumerate(num_walks_lists, 1))

        walks = flatten(walk_results)
        self.walks = walks

    def local_generate_walk(self, graph, d_graph, cpu_num, num_walks, alpha):
        walks = list()
        pbar = tqdm(total=num_walks, desc='Generating walks (CPU: {})'.format(cpu_num))
        possible_walks = ["local", "nonlocal"]
        walks_pr = [1-alpha, alpha] # pr of selecting high degree nodes, low degree nodes
        
        for n_walk in range(num_walks):
            random_group = np.random.choice(possible_walks, p=walks_pr, size=1)[0]
            pbar.update(1)

            shuffled_nodes = list(graph.nodes())
            random.shuffle(shuffled_nodes)
         
            # Start a random walk from every node
            for source in shuffled_nodes:
  
                walk = [source]
                
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

 
        


 
