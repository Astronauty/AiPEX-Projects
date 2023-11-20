import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import math
from bond_graph import *
from itertools import permutations

class BondGraphEnv(gym.env):
    def __init__(self, max_nodes, num_node_types):
        num_edge_actions = math.comb(max_nodes, 2)*2 # Multiply by 2 for bond causality
        num_node_actions = num_node_types**max_nodes
        
        possible_edge_list = permutations(range(max_nodes), 2)
        
        self.action_space = spaces.Discrete(num_edge_actions + num_node_actions)
        
        
        BondGraph(max_nodes, num_node_types)
    
    
    def reset(self):
        pass
        
        