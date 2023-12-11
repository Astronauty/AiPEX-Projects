import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import math
from bond_graph import *
from itertools import permutations
import random

class BondGraphEnv(gym.Env):
    def __init__(self, max_bond_graphs, max_nodes):
        num_node_types = len(BondGraphElementTypes)
        
        num_edge_actions = math.permute(max_nodes, 2)*2 # Multiply by 2 for bond causality
        num_node_actions = num_node_types**max_nodes # Max number of nodes allowed in bg times number of node types
        
        self.possible_edge_list = permutations(range(max_nodes), 2)
        
        # self.edge_action_space = spaces.Discrete(num_edge_actions)
        # self.node_action_space = spaces.Discrete(num_node_actions)
        
        ## Actions
        # Add node
        # Connect two existing nodes
        
        self.bond_graph_space = spaces.Discrete(max_bond_graphs)
        self.bond_graph_list = []
    
        # self.node_type_space = spaces.Discrete(num_node_types)
        # self.edge_space = spaces.MultiDiscrete([max_nodes, max_nodes]) 
        # self.edge_type_space = spaces.Discrete(2) # 2 different types of causality
        
        
        add_node_or_edge = spaces.Discrete(2) # 0 for add node, 1 for add edge
        node_space = spaces.Discrete(num_node_types)        
        edge_space = spaces.MultiDiscrete([max_nodes, max_nodes, 2]) 

        # (2, num_node_types, possible_edges_to_add)
        self.action_space = spaces.Graph(add_node_or_edge, node_space, edge_space)
        
        self.bg = BondGraph()
        self.bg_list = {}
        
    def _action_to_bond_graph_action():
        # Return action converting discrete space to either a node or edge addition
        if action < num_edge_actions:
            return 
        else:
            return
        pass
    
    def _get_action_mask(self, add_node_or_edge):
        if add_node_or_edge == 0:
            
            return 
        else:
            return
            
        return 
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.n
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
        pass
    
    def step(self, action):
        return observation, reward, terminated, info
        
    
    def _get_obs(self):
        return list(self.bg.nodes.data()), list(self.bg.edges.data())
    
    def _get_info(self):
        pass