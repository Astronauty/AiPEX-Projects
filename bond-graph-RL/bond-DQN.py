import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class BondDQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(BondDQN, self).__init__()