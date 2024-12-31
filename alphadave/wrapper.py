from itertools import chain

import torch
from gymnasium import ObservationWrapper

from pvzgym.envs import PvZEnv


class TorchEnv(ObservationWrapper):

    def __init__(self, env: PvZEnv):
        ObservationWrapper.__init__(self, env)

        self.num_states = sum([value.shape[0] for value in env.observation_space.values()])
        self.num_actions = env.action_space.n

    def observation(self, observation: dict):
        '''
        将字典形式的observation转为tensor
        '''
        return torch.tensor(list(chain(*observation.values())))
