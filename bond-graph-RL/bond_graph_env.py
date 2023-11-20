import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import math
from bond_graph import *
from itertools import permutations

class BondGraphEnv(gym.env):
    def __init__(self, max_nodes):
        num_node_types = len(BondGraphPortTypes)
        
        num_edge_actions = math.comb(max_nodes, 2)*2 # Multiply by 2 for bond causality
        num_node_actions = num_node_types**max_nodes
        possible_edge_list = permutations(range(max_nodes), 2)
        self.action_space = spaces.Discrete(num_edge_actions + num_node_actions)
        

        # Number of states equals number of possible bond graphs
        self.observation_space = spaces.Graph(node_space=spaces.Discrete(num_node_types), edge_space=spaces.Discrete(2))
        
        
        self.bond_graph_dict = nx.DiGraph()
    
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.
        
        return observation, info
        pass
        
    
    def _get_obs(self):
        return self.bond_graph