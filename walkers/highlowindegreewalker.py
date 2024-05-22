import networkx as nx
import random
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

# import warnings
# from pandas.core.common import SettingWithCopyWarning
# warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# with warnings.catch_warnings():
#     warnings.filterwarnings('ignore', category=pd.core.common.SettingWithCopyWarning) 

try:
  from .walker import Walker
except Exception as error:
    from walker import Walker

class HighLowInDegreeWalker(Walker):
    def __init__(self, graph, alpha=0.5, workers=1, dimensions=64, walk_len=10, num_walks=200):
        print("High Low In  Degree Walker with alpha = ", alpha)
        super().__init__(graph, workers=workers,dimensions=dimensions,walk_len=walk_len,num_walks=num_walks)
      
        self.number_of_nodes = self.graph.number_of_nodes()
        self.node_attrs = nx.get_node_attributes(graph, "group")

        # Init Probability Dict
        groups = ["high","low"]
        
        # self.d_graph = {node:{group : {"pr":list(), "ngh":list()}} for node in self.graph.nodes() for group in groups}
        self.d_graph = dict()
        for node in self.graph.nodes():
            self.d_graph[node] = dict()
            for group in groups:
                self.d_graph[node][group] = {"pr":list(), "ngh":list()}
 
        indegree = dict(self.graph.in_degree()) # note now it is indegree
        self.indegree_df = pd.DataFrame.from_dict(indegree, orient='index', columns=['indegree'])
        
        print("!!!!!! Computing relative degree for every node")
        # self._compute_rel_degree(indegree)
        self._compute_rel_degree_global(indegree)
      
        # compute probabilities
        print("!!!!  Computing Probability Matrix")
        self._precompute_probabilities()
        print("!!!!  Generate Walks")
        self._generate_walks(self.graph, self.d_graph, alpha)

    def _compute_rel_degree_global(self, indegree):
        prop_dict = dict()
        
        vals = [v for k, v in indegree.items()]
        sum_degree_ = np.sum(vals)

        for node, ind in indegree.items():    
            val = ind/sum_degree_
            prop_dict[node] = val
        
        # self.df_degree = pd.DataFrame.from_dict(prop_dict, orient='index', columns=['prop_indegree'])
        df = pd.DataFrame(list(prop_dict.items()), columns=['node','prop_indegree']) 
        df.set_index('node')
        self.df_degree = df

    def _compute_rel_degree(self, indegree):
        prop_dict = dict()
        
        unique_groups = set(self.node_attrs.values())
        sum_degree_dict = {group:0 for group in unique_groups} 
        for group in unique_groups:
            vals = [v for k, v in indegree.items() if self.node_attrs[k] == group]
            sum_degree_dict[group] = np.sum(vals)

        for node, ind in indegree.items():
            node_id = self.node_attrs[node]
            denom = sum_degree_dict[node_id]

            if denom != 0: val = ind/denom
            else: val = 0
            
            prop_dict[node] = val
        
        # self.df_degree = pd.DataFrame.from_dict(prop_dict, orient='index', columns=['prop_indegree'])
        df = pd.DataFrame(list(prop_dict.items()), columns=['node','prop_indegree']) 
        df.set_index('node')
        self.df_degree = df
        
    def _precompute_probabilities(self):
        for i in self.graph.nodes():
            # we traverse neighbours only because for non neighbours this value should be zero
            # according to formula 
            # print(self.df_degree)
            di = self.df_degree.loc[i]
            neighbors = list(self.graph.successors(i))
            ind_nghs = self.df_degree.loc[neighbors]
            diff = di - ind_nghs
            pos_diff = diff.loc[diff['prop_indegree'] > 0]
            neg_diff = diff.loc[diff['prop_indegree'] <= 0]
            
            # Design Choice - Uniform Pr
            # pos_diff["pr"] = 1
            # neg_diff["pr"] = 1
            
            # Design Choice - Indegree beta = 2 pr
            beta1, beta2 = 2, 2
            pos_idx, neg_idx = list(pos_diff.index), list(neg_diff.index)
            pos_diff.loc[pos_idx,"pr"] =  1/(self.indegree_df.loc[pos_idx, "indegree"]**beta2)
            neg_diff.loc[neg_idx, "pr"] = self.indegree_df.loc[neg_idx, "indegree"]**beta1
            # Normalize the prs
            sum_pos, sum_neg = pos_diff['pr'].sum(), neg_diff['pr'].sum()
   
            if sum_pos != 0:
                prs = pos_diff["pr"].tolist()
                self.d_graph[i]["low"]["pr"] = prs/sum_pos
                self.d_graph[i]["low"]["ngh"] = pos_diff.index.tolist()
                 
            if sum_neg != 0:
               prs = neg_diff["pr"].tolist()        
               self.d_graph[i]["high"]["pr"] = prs/sum_neg
               self.d_graph[i]["high"]["ngh"] = neg_diff.index.tolist()  

            
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
        possible_groups = ["high", "low"]
        group_pr = [alpha, 1-alpha] # pr of selecting high degree nodes, low degree nodes
        
        for n_walk in range(num_walks):
  
            pbar.update(1)

            shuffled_nodes = list(graph.nodes())
            random.shuffle(shuffled_nodes)

            # Start a random walk from every node
            for source in shuffled_nodes:
  
                walk = [source]
                while len(walk) < self.walk_len:
                       last_node = walk[-1]
                       group2neighbors = d_graph[last_node].keys() # "high", "low"
                       all_possible_groups = [group for group in group2neighbors if len(d_graph[last_node][group]["pr"]) > 0]
                       random_group = np.random.choice(possible_groups, p=group_pr, size=1)[0]
                       if random_group not in all_possible_groups: break

                       walk_options = list(d_graph[last_node][random_group]["ngh"])
                       probabilities = d_graph[last_node][random_group]["pr"]
                       if len(probabilities) == 0: break  # skip nodes with no ngs
                       next_node = np.random.choice(walk_options, size=1, p=probabilities)[0]
                       walk.append(next_node)

                walk = list(map(str, walk))  # Convert all to strings
                walks.append(walk)

    
        pbar.close()
        return walks

 
