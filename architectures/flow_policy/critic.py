import torch
import torch.nn as nn

from common.networks import mlp


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        self.net = mlp([obs_dim + action_dim, hidden_size, hidden_size, hidden_size, 1])

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


class REDQCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_qs: int = 10,
        hidden_size: int = 256,
    ):
        super().__init__()
        self.num_qs = num_qs
        self.qs = nn.ModuleList(
            [QNetwork(obs_dim, action_dim, hidden_size) for _ in range(num_qs)]
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.stack([q(obs, action) for q in self.qs], dim=0)

    def aggregate(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        method: str = "mean",
        num_min: int = 2,
    ) -> torch.Tensor:
        if method == "mean":
            return self(obs, action).mean(dim=0)
        if method == "min":
            return self(obs, action).min(dim=0).values
        if method == "subsample":
            return self.min(obs, action, num_min)
        raise ValueError(f"unknown q aggregation: {method}")

    def min(self, obs: torch.Tensor, action: torch.Tensor, num_min: int = 2) -> torch.Tensor:
        if num_min >= self.num_qs:
            values = self(obs, action)
        else:
            # redq trick: always take min over 2 random critics, avoids being too pessimistic to lower outliers
            idx = torch.randperm(self.num_qs)[:num_min].tolist()
            values = torch.stack([self.qs[i](obs, action) for i in idx], dim=0)
        return values.min(dim=0).values
