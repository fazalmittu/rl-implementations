"""Experience replay buffer.

Shared infrastructure used by all off-policy algorithms (TD3, SAC, DDPG, ...).
A fixed-size circular buffer of transitions; `sample` returns a random minibatch
as torch tensors ready to feed into a network.
"""

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, capacity=1_000_000, device="cpu"):
        self.capacity = capacity
        self.device = device

        # Pre-allocated numpy arrays — a circular buffer. Far cheaper than a
        # list of tuples and trivial to sample from by index.
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

        self.ptr = 0   # where the next transition is written
        self.size = 0  # how many valid transitions currently stored

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity      # wrap around when full
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        to_t = lambda x: torch.as_tensor(x[idx], device=self.device)
        return (
            to_t(self.obs),
            to_t(self.action),
            to_t(self.reward),
            to_t(self.next_obs),
            to_t(self.done),
        )

    def __len__(self):
        return self.size
