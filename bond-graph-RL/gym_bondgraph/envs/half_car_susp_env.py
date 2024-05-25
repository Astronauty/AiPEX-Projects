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


class HalfCarSuspEnv(gym.Env):
    def __init__(self, seed, t):
        self.t = t
        self.z = []
        
        ### Suspension params
        self.m_b = 1794.0
        self.I = 3443.05
        self.m_wf = 87.15
        self.m_wr = 140.04

        self.k_tf = 155900.0
        self.k_tr = 200000.0

        self.b1 = 1.271
        self.b2 = 1.713

        ### Nominal
        # self.C_sf = 1190
        # self.C_sr = 1000
        # self.k_sf = 66824
        # self.k_sr = 18615

        ### Optimal 
        self.C_sf = 2497.9
        self.C_sr = 2494.5
        self.k_sf = 28949.4
        self.k_sr = 11115.7

        ### Parameter Ranges
        self.C_sf_min = 1000
        self.C_sr_min = 1000
        self.k_sf_min = 10000
        self.k_sr_min = 10000 

        self.C_sf_max = 2500
        self.C_sr_max = 2500
        self.k_sf_max = 70000
        self.k_sr_max = 70000

        
        ### Action space definition
        self.action_space = spaces.Discrete(8, start=0, seed=seed) # increment/decrement the two suspension parameters
            
        self.observation_space = spaces.Box(low=np.array([self.C_sf_min, self.C_sr_min, self.k_sf_min, self.k_sr_min]), high=np.array([self.C_sf_max, self.C_sr_max, self.k_sf_max, self.k_sr_max]), dtype=np.float64)

        self.render_mode = None

        self.max_reward = -99999

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.C_sf = 1190
        self.C_sr = 1000
        self.k_sf = 66824
        self.k_sr = 18615
        
        observation = self._get_obs()
        info = self._get_info()
        self.max_reward = -99999
        
        return observation, info


    def step(self, action):  
        observation = self._get_obs()
        info = self._get_info()

        if action == 0:
            # Decrement C_sf
            self.C_sf -= 1.0
        elif action == 1:
            # Increment C_sf
            self.C_sf += 1.0
        elif action == 2:
            # Decrement C_sr
            self.C_sr -= 1.0
        elif action == 3:
            # Increment C_sr
            self.C_sr += 1.0
        elif action == 4:
            # Decrement k_sf
            self.k_sf -= 5.0
        elif action == 5:
            # Increment k_sf
            self.k_sf += 5.0
        elif action == 6:
            # Decrement k_sr
            self.k_sr -= 5.0
        elif action == 7:
            # Increment k_sr
            self.k_sr += 5.0




        
        # terminated = self.bond_graph.flow_causal_graph.nodes[8]["params"]["R"] < self.Rmin \
        #     or self.bond_graph.flow_causal_graph.nodes[8]["params"]["R"] > self.Rmax \
        #     or 1.0/self.bond_graph.flow_causal_graph.nodes[7]["params"]["C"] < self.Kmin \
        #     or 1.0/self.bond_graph.flow_causal_graph.nodes[7]["params"]["C"] > self.Kmax
        # reward = 1000.0/self.half_suspension_reward()
        # reward = 1000*np.power(self.half_suspension_reward(), -2)
        reward = -self.half_suspension_reward()
        
        if reward > self.max_reward:
            self.max_reward = reward

        
        terminated = self.C_sf < self.C_sf_min \
            or self.C_sf > self.C_sf_max \
            or self.C_sr < self.C_sr_min \
            or self.C_sr > self.C_sr_max \
            or self.k_sf < self.k_sf_min \
            or self.k_sf > self.k_sf_max \
            or self.k_sr < self.k_sr_min \
            or self.k_sr > self.k_sr_max
                
        # if terminated:
        #     reward = -10
            
            
        return observation, self.max_reward, terminated, False, info
        
    def speed_bump_excitation(self, t):
        H = 0.075
        L = 0.5
        KPH_to_MPS = 1.0/3.6
        v = 10 * KPH_to_MPS

        if t >= 0:
            # y1 = -(H/2)*(np.cos(2*np.pi*v*t/L)-1), 
            # return (np.pi*H*v/L)*(np.sin(2*np.pi*v*t/L))
            # return 0.0
            y1 = 0.1
        else:
            y1 = 0.0
        
        # if t >= v/(b1+b2):
        if t >= 0.1:
            y2 = 0.1
        else:
            y2 = 0.0

        return np.array([y1, y2])
    
    def half_car_dynamics(self, z, t, u):
        xf = z[0]
        xf_dot = z[1]
        xr = z[2]
        xr_dot = z[3]
        phi = z[4]
        phi_dot = z[5]
        xb = z[6]
        xb_dot = z[7]
        
        xf_ddot = (1/self.m_wf)*(-self.C_sf*(xf_dot+self.b1*phi_dot-xb_dot) - self.k_sf*(xf+self.b1*phi-xb) - self.k_tf*xf + self.k_tf*self.speed_bump_excitation(t)[0])
        xr_ddot = (1/self.m_wr)*(-self.C_sr*(xr_dot+self.b2*phi_dot-xb_dot) - self.k_sr*(xr-self.b2*phi-xb) - self.k_tr*xr + self.k_tr*self.speed_bump_excitation(t)[1])
        phi_ddot = (1/self.I)*(-self.C_sf*self.b1*(self.b1*phi_dot+xf_dot-xb_dot) - self.C_sr*self.b2*(self.b2*phi_dot-xr_dot+xb_dot) - self.k_sf*self.b1*(self.b1*phi+xf-xb) - self.k_sr*self.b2*(self.b2*phi-xr+xb))
        xb_ddot = (1/self.m_b)*(-self.C_sf*(xb_dot-self.b1*phi_dot-xf_dot) - self.C_sr*(xb_dot+self.b2*phi_dot-xr_dot) - self.k_sf*(xb-self.b1*phi-xf) - self.k_sr*(xb+self.b2*phi-xr))
        return[xf_dot, xf_ddot, xr_dot, xr_ddot, phi_dot, phi_ddot, xb_dot, xb_ddot]
        
    def half_suspension_reward(self):
        z0 = np.zeros(8)
        z = integrate.odeint(self.half_car_dynamics, np.array(z0), self.t, args=(self.speed_bump_excitation,))
        self.z = z
        
        dt = self.t[1] - self.t[0]

        xf = z[:, 0]
        xf_dot = z[:, 1]
        xr = z[:, 2]
        xr_dot = z[:, 3]
        phi = z[:, 4]
        phi_dot = z[:, 5]
        xb = z[:, 6]
        xb_dot = z[:, 7]


        ## J1: Peak accel of vehicle body
        xb_ddot = np.gradient(xb_dot, dt) 
        phi_ddot = np.gradient(phi_dot, dt)

        J1 = np.linalg.norm(xb_ddot, np.inf) + np.linalg.norm(phi_ddot, np.inf)
        # print("J1: ", J1)

        ## J2: Peak dynamic load
        u = np.array([self.speed_bump_excitation(t_i) for t_i in self.t])

        # print(u[:,0].shape)
        # print(xf.shape)
        J2 = np.linalg.norm(self.k_tf*(xf - u[:, 0]), np.inf) + np.linalg.norm(self.k_tr*(xr - u[:, 1]), np.inf)
        # print("J2: ", J2)

        ## J3: Suspension working space peak val
        J3 = max(xb - xf) + max(xb - xr)
        # print("J3: ", J3)

        ## J4: Settling time
        crossing_times = [self.t[i] for i in range(len(xb)) if abs(xb[i]) >= 0.0001]
        last_crossing_time = crossing_times[-1] if crossing_times else None
        # print(last_crossing_time)
        J4 = last_crossing_time
        # print("J4: ", J4)

        # ## Weights
        w1 = 100
        w2 = 100
        w3 = 100
        w4 = 50

        J = [J1, J2, J3, J4]
        w = [w1, w2, w3, w4]

        # # print("J:", J)

        J_lower = [3.844, 2293.4, 0.1531, 5.9042]
        J_upper = [10.518, 31772, 0.2343, 8]

        # # J_overall = J_lower + J_upper
        J_overall = 0

        for i in range(len(J)):
            J_overall += abs(w[i] * (J[i] - J_lower[i])/(J_upper[i] - J_lower[i]))
            # print(w[i] * (J[i] - J_lower[i])/(J_upper[i] - J_lower[i]))
            # J_overall += w[i] * (J[i] - J_lower[i])/(J_upper[i] - )
            

        # print("J Overall: ", J_overall)
        return J_overall
    
    def _get_obs(self):
        observation = np.array([self.C_sf, self.C_sr, self.k_sf, self.k_sr])
        return observation


    def _get_info(self):
        return {"z": self.z}
        
