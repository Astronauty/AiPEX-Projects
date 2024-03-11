import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import math
from bond_graph import *
from bond_graph_nodes import*
from itertools import permutations
import random
import copy
from gymnasium.envs.registration import register



class BondGraphEnv(gym.Env):
    def __init__(self, seed, seed_graph, max_nodes, default_params):
        # super(BondGraphEnv, self).__init__()
        self.seed_graph = seed_graph
        self.max_nodes = max_nodes 
        self.default_params = default_params
        
        self.bond_graph = copy.deepcopy(seed_graph)
        
        num_node_types = len(BondGraphElementTypes)
        
        # num_edge_actions = math.permute(max_nodes, 2)*2 # Multiply by 2 for bond causality
        # num_node_actions = num_node_types**max_nodes # Max number of nodes allowed in bg times number of node types
        
        # self.possible_edge_list = permutations(range(max_nodes), 2)

        # self.bond_graph_space = spaces.Discrete(max_bond_graphs)
        # self.bond_graph_list = []
    

        # Action space definition
        add_node_space = spaces.Discrete(num_node_types-1, start=0, seed=seed) # node additions correspond to choosing what type you want, don't include the NONE type for adding
        add_edge_space = spaces.MultiDiscrete([max_nodes, max_nodes, 2], seed=seed) # edge additions sample space

        self.action_space = spaces.Dict(
            {
                "add_node_space": add_node_space,
                "add_edge_space": add_edge_space,
            }
        )
        
        # Observation space definition
        adjacency_matrix_space = spaces.Box(low=0, high=1, shape=(max_nodes, max_nodes), dtype=np.int32) # represents the flow-causal adacency matrix
        node_feature_space = spaces.MultiDiscrete([max_nodes, num_node_types], seed=seed) # look at up to the number of max_nodes
        
        self.observation_space = spaces.Dict(
            {
                "adjacency_matrix_space": adjacency_matrix_space,
                "node_feature_space": node_feature_space,
            }
        )
        
        self.render_mode = None
        

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset the bond graph to the seed state
        self.bond_graph = copy.deepcopy(self.seed_graph)
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        
        element_addition_mask = self.bond_graph.get_element_addition_mask()
        
        match action:
            case BondGraphElementTypes.CAPACITANCE:
                self.bond_graph.add_element(Capacitance(capacitance = self.default_params['C']))
                
            case BondGraphElementTypes.INERTANCE:
                self.bond_graph.add_element(Inertance(inertance = self.default_params['I']))

            case BondGraphElementTypes.RESISTANCE:
                self.bond_graph.add_element(Resistance(resistance = self.default_params['R']))
                
            case BondGraphElementTypes.EFFORT_SOURCE:
                self.bond_graph.add_element(EffortSource())
                    
            case BondGraphElementTypes.FLOW_SOURCE:
                self.bond_graph.add_element(FlowSource())
                
            case BondGraphElementTypes.ZERO_JUNCTION:
                self.bond_graph.add_element(ZeroJunction())
                
            case BondGraphElementTypes.ONE_JUNCTION:
                self.bond_graph.add_element(OneJunction())
            case _: # Default case
                raise ValueError("Invalid action applied in BondGraphEnv.")
        
        observation = self._get_obs()
        
        info = self._get_info()
        
        if self.bond_graph.is_valid_solution():
            reward = self.bond_graph.reward()
        else:
            reward = -1
            
        terminated = self.bond_graph.at_max_node_size()

        return observation, reward, terminated, False, info
        
    
    def _get_obs(self):
        element_types = [self.bond_graph.flow_causal_graph.nodes[node]['element_type'] for node in self.bond_graph.flow_causal_graph.nodes]
        element_types_vec = np.array(element_types)
        
        adjacency_matrix = np.array(nx.adjacency_matrix(self.bond_graph.flow_causal_graph).todense(), dtype=np.int32)
        
        return {'node_feature_space': element_types_vec, 'adjacency_matrix_space': adjacency_matrix}
    
    def _get_info(self):
        num_nodes = self.bond_graph.flow_causal_graph.number_of_nodes()
        node_info = self.bond_graph.flow_causal_graph.nodes(data=True)
        
        return {
            "num_nodes": num_nodes,
            "node_info": node_info,
        }
        
