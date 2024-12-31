import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def append(self, transition):
        self.buffer.append(transition)

    def sample(self, sample_size):
        return random.sample(self.buffer, sample_size)

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer(ReplayBuffer):

    def __init__(self, capacity, alpha, beta):
        super().__init__(capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.beta = beta

    def append(self, transition):
        super().append(transition)
        self.priorities.append(max(self.priorities, default=1))

    def sample(self, sample_size):
        # get sample probabilities
        probs = self.get_probabilities()
        # get samples based on probabilities
        indices = np.random.choice(len(self.buffer), sample_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        # get weights
        weights = self.get_weights(probs[indices])
        return samples, indices, weights

    def get_probabilities(self):
        probs = np.array(self.priorities) ** self.alpha
        probs /= probs.sum()
        return probs

    def get_weights(self, probs):
        weights = (len(self.buffer) * probs) ** (-self.beta)
        weights /= weights.max()
        return weights

    def set_priorities(self, indices, priorities):
        for i, priority in zip(indices, priorities):
            self.priorities[i] = priority + 1e-10
