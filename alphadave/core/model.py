import numpy as np
from torch import nn


class DQN(nn.Module):
    def __init__(self, in_states, out_actions):
        super().__init__()

        hidden_size = 4 * in_states

        self.layers = nn.Sequential(
            nn.Linear(in_states, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_actions),
        )

    def forward(self, x, mask=None):
        '''
        mask: True for not mask
        '''
        qvals = self.layers(x)
        if isinstance(mask, np.ndarray):
            qvals[np.logical_not(mask)] = qvals.min()
        return qvals
