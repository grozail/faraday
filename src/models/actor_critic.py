import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from src.visualization.tbx import board

def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.1)


class ActorCritic(nn.Module):
    def __init__(self, input_dim, out_dim, max_output_value):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, out_dim),
            nn.Hardtanh(-max_output_value, max_output_value)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
        self.max_output_value = max_output_value
        self.log_std = nn.Parameter(torch.ones(1, out_dim) * 2.71)
        self.apply(_init_weights)

    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value
