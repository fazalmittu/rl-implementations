import torch
import torch.nn as nn
import torch.nn.functional as F

from architectures.diffusion_policy.obs_encoder import ObsEncoder

class DiffusionPolicy(nn.Module):

    def __init__(
        self, 
        hidden_size: int = 128, 
        state_dim: int = 6,
        action_dim: int = 6,
        action_pred_horizon: int = 16,
        obs_horizon: int = 2,
        action_horizon: int = 8,
        batch_size: int = 4,
        diffusion_timesteps: int = 10
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_pred_horizon = action_pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.batch_size = batch_size

        self.obs_encoder = ObsEncoder(hidden_size=self.hidden_size, state_dim=self.state_dim)




if __name__ == "__main__":

    print("hello world")