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

VALID_SOLUTION_REWARD_SCALING = 10
INVALID_SOLUTION_REWARD = -1
MASKED_ACTION_PENALTY = -100

MAX_PARAM_VAL = 50

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
        add_node_space = spaces.Discrete(num_node_types-3, start=3, seed=seed) # node additions correspond to choosing what type you want, don't include the NONE type for adding
        add_edge_space = spaces.MultiDiscrete([max_nodes, max_nodes, 2], seed=seed) # edge additions sample space

        self.action_space = spaces.Dict(
            {
                'node_or_bond': spaces.Discrete(2, start=0, seed=seed),
                'node_param': spaces.Discrete(MAX_PARAM_VAL, start=1, seed=seed),
                "node_type": add_node_space,
                "bond": add_edge_space,
            }
        )
        
        # Observation space definition
        adjacency_matrix_space = spaces.Box(low=0, high=1, shape=(max_nodes, max_nodes), dtype=np.int32) # represents the flow-causal adacency matrix
        node_type_space = spaces.MultiDiscrete([num_node_types]*max_nodes, seed=seed) # look at up to the number of max_nodes
        node_parameter_space = spaces.MultiDiscrete([MAX_PARAM_VAL]*max_nodes, seed=seed)
        
        self.observation_space = spaces.Dict(
            {
                "adjacency_matrix_space": adjacency_matrix_space,
                "node_type_space": node_type_space,
                "node_param_space": node_parameter_space
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
        
        if action['node_or_bond'] == 0: # Node addition
            element_addition_mask = self.bond_graph.get_element_addition_mask()
            
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
                    reward = self.bond_graph.reward()*VALID_SOLUTION_REWARD_SCALING
                else:
                    reward = INVALID_SOLUTION_REWARD
            else: 
                reward = MASKED_ACTION_PENALTY # penalize adding elements that are masked heavily

        else: # Bond addition
            causal_adjacency_mask, power_flow_adjacency_mask = self.bond_graph.get_bond_addition_mask()
            
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
                            reward = self.bond_graph.reward()*VALID_SOLUTION_REWARD_SCALING
                        else:
                            reward = INVALID_SOLUTION_REWARD   
                    else:
                        self.bond_graph.add_bond(action['bond'][0], action['bond'][1], -1)
                        if self.bond_graph.is_valid_solution():
                            reward = self.bond_graph.reward()*VALID_SOLUTION_REWARD_SCALING
                        else:
                            reward = INVALID_SOLUTION_REWARD   
                else:
                    reward = MASKED_ACTION_PENALTY
            else:
                reward = MASKED_ACTION_PENALTY

        observation = self._get_obs()
        info = self._get_info()
        

            
        terminated = self.bond_graph.at_max_node_size()

        return observation, reward, terminated, False, info
        
    
    def _get_obs(self):
        # Element type of nodes
        element_types = [self.bond_graph.flow_causal_graph.nodes[node]['element_type'].value for node in self.bond_graph.flow_causal_graph.nodes]
        element_types_vec = np.array(element_types)
        
        num_unfilled_nodes = self.max_nodes - len(element_types_vec)
        element_types_vec = np.pad(element_types_vec, (0, num_unfilled_nodes),'constant', constant_values=BondGraphElementTypes.NONE.value)
        
        # Paramter values of nodes
        node_param_space = self.bond_graph.get_parameters()
      
        # Adjacency matrix and padding
        adjacency_matrix = np.array(nx.adjacency_matrix(self.bond_graph.flow_causal_graph).todense(), dtype=np.int32)
        num_rows_to_pad = self.max_nodes - adjacency_matrix.shape[0]
        num_cols_to_pad = num_rows_to_pad
        adjacency_matrix = np.pad(adjacency_matrix, ((0, num_rows_to_pad), (0, num_cols_to_pad)), 'constant')
        
        return {'node_type_space': element_types_vec, 'node_param_space': node_param_space,'adjacency_matrix_space': adjacency_matrix}
    
    def _get_info(self):
        num_nodes = self.bond_graph.flow_causal_graph.number_of_nodes()
        node_info = self.bond_graph.flow_causal_graph.nodes(data=True)
        
        return {
            "num_nodes": num_nodes,
            "node_info": node_info,
        }
        
