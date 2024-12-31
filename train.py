import random

import gymnasium as gym
import numpy as np
import torch

from alphadave.agent import AlphaDave
from alphadave.wrapper import TorchEnv

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

env = TorchEnv(gym.make('PlantsVsZombies'))
env.action_space.seed(seed)


dave = AlphaDave(env)
dave.train(
    seed=seed,
    batch_size=128,
    learning_rate=0.0003,
    replay_buffer_capacity=100_000,
    replay_buffer_alpha=0.5,
    replay_buffer_beta=0.5,
    network_sync_interval=100,
    epsilon_start=1.0,
    epsilon_final=0.05,
    epsilon_decay=3_000,
    gamma=0.99,
    save_dir='results',
    save_interval=1_000,
    stop_at_episode=None,
    from_checkpoint=None,
)
