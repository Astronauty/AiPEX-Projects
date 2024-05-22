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

OBJECTIVE_REWARD_SCALING = 100
VALID_SOLUTION_REWARD = 100
INVALID_SOLUTION_REWARD = -1
MASKED_ACTION_PENALTY = -10

MIN_PARAM_VAL = 1
MAX_PARAM_VAL = 10

class BondGraphEnv(gym.Env):
    def __init__(self, seed, seed_graph, max_nodes, default_params):
        # super(BondGraphEnv, self).__init__()
        self.seed_graph = seed_graph
        self.max_nodes = max_nodes 
        self.default_params = default_params

        self.bond_graph = copy.deepcopy(seed_graph)
        self.num_node_types = len(BondGraphElementTypes)
        
        # Create custom mapping of integer actions to composite actions
        self.action_space_indices = []
        self.create_action_space_integer_mapping()
        self.action_space_size = len(self.action_space_indices)


        # Action space definition
        add_node_space = spaces.Discrete(self.num_node_types-3, start=3, seed=seed) # node additions correspond to choosing what type you want, don't include the NONE type for adding
        add_edge_space = spaces.MultiDiscrete([max_nodes, max_nodes, 2], seed=seed) # edge additions sample space

        bond_sign = spaces.Discrete(2, start=0, seed=seed)

        self.action_space = spaces.Dict(
            {
                'node_or_bond': spaces.Discrete(2, start=0, seed=seed),
                'node_param': spaces.Discrete(MAX_PARAM_VAL, start=1, seed=seed),
                "node_type": add_node_space,
                "bond": add_edge_space,
            }
        )
        
        self.integer_action_space = spaces.Discrete(self.action_space_size, seed=seed)


        # self.flattened_action_space = spaces.utils.flatten_space(self.action_space)
        
        # Observation space definition
        adjacency_matrix_space = spaces.Box(low=0, high=1, shape=(max_nodes, max_nodes), dtype=np.int64) # represents the flow-causal adacency matrix
        node_type_space = spaces.MultiDiscrete([self.num_node_types]*max_nodes, seed=seed) # look at up to the number of max_nodes
        node_parameter_space = spaces.MultiDiscrete([MAX_PARAM_VAL]*max_nodes, seed=seed)
        
        self.observation_space = spaces.Dict(
            {
                "adjacency_matrix_space": adjacency_matrix_space,
                "node_type_space": node_type_space,
                "node_param_space": node_parameter_space
            }
        )
    
        self.flattened_observation_space = spaces.utils.flatten_space(self.observation_space)
        
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


    def composite_to_integer_action(self, action_array):
        integer_action = self.action_space_indices.index(action_array)
        return integer_action

    def integer_to_composite_action(self, integer_action):
        action_array = self.action_space_indices[integer_action]
        action = {
            'node_or_bond': action_array[0], # 0 for add node, 1 for add edge
            "node_type": action_array[1],
            "bond": [action_array[3], action_array[4], action_array[5]], # 0 for negative bond sign, 1 for positive
            "node_param": action_array[2]
        }
        return action


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset the bond graph to the seed state
        self.bond_graph = copy.deepcopy(self.seed_graph)
        
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
        # return spaces.utils.flatten(self.observation_space, observation), info

    def step(self, action):  
        element_addition_mask = self.bond_graph.get_element_addition_mask()
        causal_adjacency_mask, power_flow_adjacency_mask = self.bond_graph.get_bond_addition_mask()
        
        if action['node_or_bond'] == 0: # Node addition
            # element_addition_mask = self.bond_graph.get_element_addition_mask()            
            if element_addition_mask[action['node_type']] == 1:
                match action['node_type']:
                    case BondGraphElementTypes.CAPACITANCE.value:
                        # self.bond_graph.add_element(Capacitance(capacitance = self.default_params['C']))
                        self.bond_graph.add_element(Capacitance(capacitance = action['node_param']))
                        
                    case BondGraphElementTypes.INERTANCE.value:
                        # self.bond_graph.add_element(Inertance(inertance = self.default_params['I']))
                        self.bond_graph.add_element(Inertance(inertance = action['node_param']))

                    case BondGraphElementTypes.RESISTANCE.value:
                        # self.bond_graph.add_element(Resistance(resistance = self.default_params['R']))
                        self.bond_graph.add_element(Resistance(resistance = action['node_param']))
                        
                    case BondGraphElementTypes.EFFORT_SOURCE.value:
                        self.bond_graph.add_element(EffortSource())
                            
                    case BondGraphElementTypes.FLOW_SOURCE.value:
                        self.bond_graph.add_element(FlowSource())
                        
                    case BondGraphElementTypes.ZERO_JUNCTION.value:
                        self.bond_graph.add_element(ZeroJunction())
                        
                    case BondGraphElementTypes.ONE_JUNCTION.value:
                        self.bond_graph.add_element(OneJunction())
                        
                    case _: # Default case
                        print(action['node_type'])
                        raise ValueError("Invalid node addition applied in BondGraphEnv.")
                    
                if self.bond_graph.is_valid_solution():
                    reward = VALID_SOLUTION_REWARD + self.bond_graph.reward()*OBJECTIVE_REWARD_SCALING
                else:
                    reward = INVALID_SOLUTION_REWARD
            else: 
                reward = MASKED_ACTION_PENALTY # penalize adding elements that are masked heavily

        else: # Bond addition
            # causal_adjacency_mask, power_flow_adjacency_mask = self.bond_graph.get_bond_addition_mask()
            if causal_adjacency_mask[action['bond'][0], action['bond'][1]] == 1: # check if the causality assignment is valid
                if action['bond'][2] == 0: # 0 corresponds to negative power sign
                    power_flow_adjacency_mask = power_flow_adjacency_mask.transpose()
                elif action['bond'][2] == 1:
                    power_flow_adjacency_mask = power_flow_adjacency_mask
                else:
                    raise ValueError("Incorrect value of bond power sign detected.")
                
                if power_flow_adjacency_mask[action['bond'][0], action['bond'][1]] == 1: # check if power flows are valid
                    if action['bond'][2] == 1: #
                        self.bond_graph.add_bond(action['bond'][0], action['bond'][1], 1)
                        if self.bond_graph.is_valid_solution():
                            reward = VALID_SOLUTION_REWARD +  self.bond_graph.reward()*OBJECTIVE_REWARD_SCALING
                        else:
                            reward = INVALID_SOLUTION_REWARD   
                    else:
                        self.bond_graph.add_bond(action['bond'][0], action['bond'][1], -1)
                        if self.bond_graph.is_valid_solution():
                            reward =  VALID_SOLUTION_REWARD + self.bond_graph.reward()*OBJECTIVE_REWARD_SCALING
                        else:
                            reward = INVALID_SOLUTION_REWARD   
                else:
                    reward = MASKED_ACTION_PENALTY
            else:
                reward = MASKED_ACTION_PENALTY

        observation = self._get_obs()
        info = self._get_info()
        
        # Several conditions for terminating episode: no edge additions possible, no element additions possible, or max node size
        terminated = self.bond_graph.at_max_node_size() and np.all(causal_adjacency_mask == 0) and np.all(element_addition_mask == 0)


        return observation, reward, terminated, False, info
        
        # return spaces.utils.flatten(self.observation_space, observation), reward, terminated, False, info
        
    
    def _get_obs(self):
        # Element type of nodes
        element_types = [self.bond_graph.flow_causal_graph.nodes[node]['element_type'].value for node in self.bond_graph.flow_causal_graph.nodes]
        element_types_vec = np.array(element_types)
        
        num_unfilled_nodes = self.max_nodes - len(element_types_vec)
        element_types_vec = np.pad(element_types_vec, (0, num_unfilled_nodes),'constant', constant_values=BondGraphElementTypes.NONE.value)
        
        # Paramter values of nodes
        node_params = self.bond_graph.get_parameters()
      
        # Adjacency matrix and padding
        adjacency_matrix = np.array(nx.adjacency_matrix(self.bond_graph.flow_causal_graph).todense(), dtype=np.int64)
        num_rows_to_pad = self.max_nodes - adjacency_matrix.shape[0]
        num_cols_to_pad = num_rows_to_pad
        adjacency_matrix = np.pad(adjacency_matrix, ((0, num_rows_to_pad), (0, num_cols_to_pad)), 'constant')
        
        # observation = {'node_type_space': element_types_vec, 'node_param_space': node_param_space,'adjacency_matrix_space': adjacency_matrix}
        
        # print(type(element_types_vec))
        # print(type(node_param_space))
        # print(type(adjacency_matrix))
        
        observation = OrderedDict([('adjacency_matrix_space', adjacency_matrix.astype(np.int64)),('node_param_space', node_params.astype(np.int64)), ('node_type_space', element_types_vec.astype(np.int64))])
        # observation = {
        #     "adjacency_matrix_space": adjacency_matrix,
        #     "node_type_space": element_types_vec,
        #     "node_param_space": node_params
        # }

        # flattened_observation = spaces.utils.flatten(self.observation_space, observation)
        
        return observation
        # return element_types_vec, node_param_space, adjacency_matrix

    def _get_info(self):
        # num_nodes = self.bond_graph.flow_causal_graph.number_of_nodes()
        valid_solution = self.bond_graph.is_valid_solution()
        
        
        return {
            "bond_graph": self.bond_graph,
            "valid_solution": valid_solution,
        }
        
