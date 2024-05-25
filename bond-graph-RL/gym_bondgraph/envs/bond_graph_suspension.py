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
from gymnasium.wrappers import FlattenObservation
from collections import *

OBJECTIVE_REWARD_SCALING = 1000
VALID_SOLUTION_REWARD = 1000
INVALID_SOLUTION_REWARD = -1
MASKED_ACTION_PENALTY = -1

MIN_PARAM_VAL = 1
MAX_PARAM_VAL = 10

class BondGraphSuspEnv(gym.Env):
    def __init__(self, seed, seed_graph, max_nodes, default_params):
        # super(BondGraphEnv, self).__init__()
        self.seed_graph = seed_graph
        self.max_nodes = max_nodes 
        self.default_params = default_params

        self.bond_graph = copy.deepcopy(seed_graph)
        self.num_node_types = len(BondGraphElementTypes)
        
        ## Create custom mapping of integer actions to composite actions
        # self.action_space_indices = []
        # self.create_action_space_integer_mapping()
        # self.action_space_size = len(self.action_space_indices)

        self.R = 1576.855
        self.K = 8098.479


        ## Define suspension params
        self.Rmin = 500
        self.Rmax = 4500
        self.Kmin = 5000
        self.Kmax = 50000
        
        ## Action space definition
        self.action_space = spaces.Discrete(4, start=0, seed=seed) # increment/decrement the two suspension parameters
        
        
        # Observation space definition
        # adjacency_matrix_space = spaces.Box(low=0, high=1, shape=(max_nodes, max_nodes), dtype=np.int64) # represents the flow-causal adacency matrix
        # node_type_space = spaces.MultiDiscrete([self.num_node_types]*max_nodes, seed=seed) # look at up to the number of max_nodes
        # node_parameter_space = spaces.MultiDiscrete([MAX_PARAM_VAL]*max_nodes, seed=seed)
        
        # self.observation_space = spaces.Dict(
        #     {
        #         "adjacency_matrix_space": adjacency_matrix_space,
        #         "node_type_space": node_type_space,
        #         "node_param_space": node_parameter_space
        #     }
        # )
            
        self.observation_space = spaces.Box(low=np.array([self.Rmin, 1.0/self.Kmax]), high=np.array([self.Rmax , 1.0/self.Kmin]), dtype=np.float32)
        # self.flattened_observation_space = spaces.utils.flatten_space(self.observation_space)
        
        self.render_mode = None

    # Convert the composite action space into a discrete space for compatibility with DQN
    def create_action_space_integer_mapping(self): 
        # 0: node or bond
        # 1: node type
        # 2: node parameter value
        # 3: first node for bond creation
        # 4: second node for bond creation
        # 5: bond sign

        for node_or_bond in range(2):
            for node_type in range(3, self.num_node_types):
                for node_param in range(MIN_PARAM_VAL, MAX_PARAM_VAL):
                    for bond1 in range(self.max_nodes):
                        for bond2 in range(self.max_nodes):
                            for bond_sign in range(2):
                                action_index = [node_or_bond, node_type, node_param, bond1, bond2, bond_sign]
                                self.action_space_indices.append(action_index)
                                



    # def composite_to_integer_action(self, action_array):
    #     integer_action = self.action_space_indices.index(action_array)
    #     return integer_action

    # def integer_to_composite_action(self, integer_action):
    #     action_array = self.action_space_indices[integer_action]
    #     action = {
    #         'node_or_bond': action_array[0], # 0 for add node, 1 for add edge
    #         "node_type": action_array[1],
    #         "bond": [action_array[3], action_array[4], action_array[5]], # 0 for negative bond sign, 1 for positive
    #         "node_param": action_array[2]
    #     }
    #     return action


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset the bond graph to the seed state
        self.bond_graph = copy.deepcopy(self.seed_graph)
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
        # return spaces.utils.flatten(self.observation_space, observation), info

    def step(self, action):  
        observation = self._get_obs()
        info = self._get_info()
        
        # Several conditions for terminating episode: no edge additions possible, no element additions possible, or max node size
        # terminated = self.bond_graph.at_max_node_size() and np.all(causal_adjacency_mask == 0) and np.all(element_addition_mask == 0)

        if action  == 0:
            self.bond_graph.flow_causal_graph.nodes[8]["params"]["R"] -= 1.0
        elif action == 1:
            self.bond_graph.flow_causal_graph.nodes[8]["params"]["R"] += 1.0
        elif action == 2:
            self.bond_graph.flow_causal_graph.nodes[7]["params"]["C"] -= 0.001
        elif action == 3:
            self.bond_graph.flow_causal_graph.nodes[7]["params"]["C"] += 0.001
            
        self.bond_graph.update_state_space_matrix()
        
        
        terminated = self.bond_graph.flow_causal_graph.nodes[8]["params"]["R"] < self.Rmin \
            or self.bond_graph.flow_causal_graph.nodes[8]["params"]["R"] > self.Rmax \
            or 1.0/self.bond_graph.flow_causal_graph.nodes[7]["params"]["C"] < self.Kmin \
            or 1.0/self.bond_graph.flow_causal_graph.nodes[7]["params"]["C"] > self.Kmax
            
        reward = self.bond_graph.reward()
        
        return observation, reward, terminated, False, info
        
        # return spaces.utils.flatten(self.observation_space, observation), reward, terminated, False, info
    def speed_bump_excitation(t, L, H, v):
        if t <= L/v:
            return -(H/2)*(np.cos(2*np.pi*v*t/L)-1)
            # return (np.pi*H*v/L)*(np.sin(2*np.pi*v*t/L))
            # return 0.0
        else:
            return 0.0
    

    
    def _get_obs(self):
        # Element type of nodes
        element_types = [self.bond_graph.flow_causal_graph.nodes[node]['element_type'].value for node in self.bond_graph.flow_causal_graph.nodes]
        element_types_vec = np.array(element_types)
        
        num_unfilled_nodes = self.max_nodes - len(element_types_vec)
        element_types_vec = np.pad(element_types_vec, (0, num_unfilled_nodes),'constant', constant_values=BondGraphElementTypes.NONE.value)
        
        # Paramter values of nodes
        node_params = self.bond_graph.get_parameters()
      
        observation = np.array([self.bond_graph.flow_causal_graph.nodes[8]["params"]["R"], self.bond_graph.flow_causal_graph.nodes[7]["params"]["C"]], dtype=np.float32)
        
        return observation
        # return element_types_vec, node_param_space, adjacency_matrix

    def _get_info(self):
        # num_nodes = self.bond_graph.flow_causal_graph.number_of_nodes()
        valid_solution = self.bond_graph.is_valid_solution()
        
        
        return {
            "bond_graph": self.bond_graph,
            "valid_solution": valid_solution,
        }
        
