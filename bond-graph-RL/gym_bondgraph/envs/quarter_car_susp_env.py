import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import math
# from bond_graph import *
# from bond_graph_nodes import*
from itertools import permutations
from scipy import *
import random
import copy
from gymnasium.envs.registration import register
from gymnasium.wrappers import FlattenObservation
from collections import *


class QuarterCarSuspEnv(gym.Env):
    def __init__(self, seed, t):
        self.t = t
        
        ## Optimization variables
        # self.R = 1576.855
        # self.K = 8098.479
        self.R = 1861.0
        self.K = 19660.0
        
        self.Ktire = 155900.0
        
        self.z = []
        

        ## Define suspension params
        self.m_wheel = 28.58
        self.m_body = 288.9
        
        self.Rmin = 500
        self.Rmax = 4500
        self.Kmin = 5000
        self.Kmax = 50000
        
        ## Action space definition
        self.action_space = spaces.Discrete(4, start=0, seed=seed) # increment/decrement the two suspension parameters
            
        self.observation_space = spaces.Box(low=np.array([self.Rmin, 1.0/self.Kmax]), high=np.array([self.Rmax , 1.0/self.Kmin]), dtype=np.float64)

        self.render_mode = None


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # self.R = 1576.855
        # self.K = 8098.479
        self.R = 1861.0
        self.K = 19660.0
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info


    def step(self, action):  
        observation = self._get_obs()
        info = self._get_info()
        
        # Several conditions for terminating episode: no edge additions possible, no element additions possible, or max node size

        if action  == 0:
            # self.bond_graph.flow_causal_graph.nodes[8]["params"]["R"] -= 1.0
            self.R -= 1.0
        elif action == 1:
            # self.bond_graph.flow_causal_graph.nodes[8]["params"]["R"] += 1.0
            self.R += 1.0
        elif action == 2:
            # self.bond_graph.flow_causal_graph.nodes[7]["params"]["C"] -= 0.001
            self.K -= 1.0
        elif action == 3:
            # self.bond_graph.flow_causal_graph.nodes[7]["params"]["C"] += 0.001
            self.K += 1.0
            
        
        # terminated = self.bond_graph.flow_causal_graph.nodes[8]["params"]["R"] < self.Rmin \
        #     or self.bond_graph.flow_causal_graph.nodes[8]["params"]["R"] > self.Rmax \
        #     or 1.0/self.bond_graph.flow_causal_graph.nodes[7]["params"]["C"] < self.Kmin \
        #     or 1.0/self.bond_graph.flow_causal_graph.nodes[7]["params"]["C"] > self.Kmax
        reward = 100.0/self.quarter_suspension_reward()
        
        terminated = self.R < self.Rmin \
            or self.R > self.Rmax \
            or self.Kmin < self.Kmin \
            or self.Kmax > self.Kmax \
                
        if terminated:
            reward = -10
            
            
        return observation, reward, terminated, False, info
        
        # return spaces.utils.flatten(self.observation_space, observation), reward, terminated, False, info
    def speed_bump_excitation(self, t):
        H = 0.075
        L = 0.5
        KPH_to_MPS = 1.0/3.6
        v = 10 * KPH_to_MPS
        
        if t <= L/v:
            return -(H/2)*(np.cos(2*np.pi*v*t/L)-1)
        else:
            return 0.0
    
    def quarter_car_dynamics(self, z, t, u):
        xw = z[0]
        xw_dot = z[1]
        xb = z[2]
        xb_dot = z[3]

        xw_ddot = (1.0/self.m_wheel)*(-self.R*(xw_dot - xb_dot) - self.K*(xw - xb) - self.Ktire*xw + self.Ktire*u(t))
        xb_ddot = (1.0/self.m_body)*(-self.R*(xb_dot - xw_dot) - self.K*(xb - xw))
        
        return [xw_dot, xw_ddot, xb_dot, xb_ddot]
    
    def quarter_suspension_reward(self):
        x0 = [0.0, 0.0, 0.0, 0.0] 
        z = integrate.odeint(self.quarter_car_dynamics, np.array(x0), self.t, args=(self.speed_bump_excitation,))
        self.z = z
 
        
        dt = self.t[1] - self.t[0]
        xw = z[:, 0]
        vw = z[:, 1]
        xb = z[:, 2]
        vb = z[:, 3]
        
        ### Compute cost
        ## J1: Peak accel of vehicle body
        ab = np.gradient(vb, dt)
        J1 = np.linalg.norm(ab, np.inf)
        # print("J1: ", J1)
        u = np.array([self.speed_bump_excitation(t_i) for t_i in self.t])

        ## J2: Peak dynamic load
        J2 = np.linalg.norm(self.Ktire*(xw - u), np.inf)
        # print("J2: ", J2)

        ## J3: Suspension working space peak val
        J3 = max(xb - xw)
        # print("J3: ", J3)

        ## J4: Settling time
        crossing_times = [self.t[i] for i in range(len(xb)) if abs(xb[i]) > 0.0001]
        last_crossing_time = crossing_times[-1] if crossing_times else None
        # print(last_crossing_time)
        J4 = last_crossing_time
        # print("J4: ", J4)

        ## Weights
        w1 = 35
        w2 = 35
        w3 = 50
        w4 = 50

        J = [J1, J2, J3, J4]
        w = [w1, w2, w3, w4]


        J_lower = [3.18, 1719, 0.044, 0.08]
        J_upper = [18.75, 5499, 0.08, 3.0]


        # J_overall = J_lower + J_upper
        J_overall = 0

        for i in range(len(J)):
            J_overall += abs(w[i] * (J[i] - J_lower[i])/(J_upper[i] - J_lower[i]))
            # print(w[i] * (J[i] - J_lower[i])/(J_upper[i] - J_lower[i]))
            # J_overall += w[i] * (J[i] - J_lower[i])/(J_upper[i] - )
            

        # print("J Overall: ", J_overall)
        return J_overall
    
    
    def _get_obs(self):
        observation = np.array([self.R, self.K])
        return observation


    def _get_info(self):
        return {"z": self.z}
        
