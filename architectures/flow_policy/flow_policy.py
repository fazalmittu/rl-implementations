import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t[None]
        t = t.float()

        half_dim = self.dim // 2
        freqs = torch.exp(
            torch.arange(half_dim, device=t.device, dtype=t.dtype)
            * (-math.log(10_000.0) / max(half_dim - 1, 1))
        )
        args = t[:, None] * freqs[None, :] * (2.0 * math.pi)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class FlowPolicy(nn.Module):
    """Small conditional flow-matching policy for action chunks.

    The policy learns a velocity field v_theta(x_t, t, obs) for a linear
    interpolant x_t = (1 - t) * noise + t * action_chunk. Sampling integrates
    from t=0 noise to t=1 action_chunk with a simple Euler solver.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_horizon: int = 4,
        hidden_size: int = 256,
        num_layers: int = 4,
        time_embed_dim: int = 64,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.time_embed_dim = time_embed_dim

        self.time_encoder = SinusoidalTimeEmbedding(time_embed_dim)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
        )

        action_flat_dim = action_horizon * action_dim
        input_dim = hidden_size + action_flat_dim + time_embed_dim

        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(dim, hidden_size),
                    nn.SiLU(),
                    nn.LayerNorm(hidden_size),
                ]
            )
            dim = hidden_size
        layers.append(nn.Linear(hidden_size, action_flat_dim))
        self.velocity_net = nn.Sequential(*layers)

    def forward(
        self,
        obs: torch.Tensor,
        noisy_actions: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        obs:           (B, obs_dim)
        noisy_actions: (B, action_horizon, action_dim)
        t:             (B,) in [0, 1]

        returns: (B, action_horizon, action_dim)
        """
        if t.dim() == 0:
            t = t.expand(obs.shape[0])

        obs_emb = self.obs_encoder(obs)
        time_emb = self.time_encoder(t)
        action_flat = noisy_actions.flatten(start_dim=1)

        velocity = self.velocity_net(torch.cat([obs_emb, action_flat, time_emb], dim=-1))
        return velocity.view(obs.shape[0], self.action_horizon, self.action_dim)

    def flow_matching_loss(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(actions)
        t = torch.rand(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t_view = t.view(-1, 1, 1)

        noisy_actions = (1.0 - t_view) * noise + t_view * actions
        target_velocity = actions - noise
        pred_velocity = self(obs, noisy_actions, t)

        return F.mse_loss(pred_velocity, target_velocity, reduction="mean")

    @staticmethod
    def gaussian_log_prob(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        var = std.pow(2)
        log_prob = -0.5 * (
            (value - mean).pow(2) / var
            + 2.0 * torch.log(std)
            + math.log(2.0 * math.pi)
        )
        return log_prob.flatten(start_dim=1).sum(dim=1)

    def sample(
        self,
        obs: torch.Tensor,
        num_steps: int = 10,
        return_chain: bool = True,
        noise: torch.Tensor | None = None,
        noise_std: float = 0.0,
        return_log_probs: bool = False,
        chain: torch.Tensor | None = None,
        debias_noise: bool = False,
    ):
        if return_log_probs and noise_std <= 0.0:
            raise ValueError("noise_std must be positive when return_log_probs=True")

        was_training = self.training
        self.eval()

        try:
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            obs = obs.to(device=device, dtype=dtype)
            batch_size = obs.shape[0]

            replay_chain = chain
            if replay_chain is not None:
                replay_chain = replay_chain.to(device=device, dtype=dtype).detach()
                if replay_chain.shape[1] != num_steps + 1:
                    raise ValueError(f"chain must have {num_steps + 1} states")
                x = replay_chain[:, 0]
            elif noise is None:
                x = torch.randn(
                    batch_size,
                    self.action_horizon,
                    self.action_dim,
                    device=device,
                    dtype=dtype,
                )
            else:
                x = noise.to(device=device, dtype=dtype)

            chain_out = [x] if return_chain and replay_chain is None else None
            log_probs = [] if return_log_probs else None
            dt = 1.0 / float(num_steps)
            std = torch.as_tensor(noise_std, device=device, dtype=dtype)

            for step in range(num_steps):
                t = torch.full((batch_size,), step * dt, device=device, dtype=dtype)
                x_prev = replay_chain[:, step] if replay_chain is not None else x
                velocity = self(obs, x_prev, t)
                step_std = std
                if noise_std > 0.0 and debias_noise:
                    t_view = t.view(-1, 1, 1)
                    step_std = std * torch.sqrt((1.0 - t_view).clamp_min(0.0))
                    correction = 0.5 * std.pow(2) * (t_view * velocity - x_prev)
                    velocity = velocity + correction
                mean = x_prev + dt * velocity

                if replay_chain is not None:
                    x = replay_chain[:, step + 1]
                    if log_probs is not None:
                        log_probs.append(self.gaussian_log_prob(x, mean, step_std))
                elif noise_std > 0.0:
                    x_next = mean.detach() + step_std * torch.randn_like(mean)
                    if log_probs is not None:
                        log_probs.append(self.gaussian_log_prob(x_next, mean, step_std))
                    x = x_next.detach()
                else:
                    x = mean

                if chain_out is not None:
                    chain_out.append(x)

            if return_log_probs:
                log_probs = torch.stack(log_probs, dim=1)
                if return_chain:
                    chain_tensor = replay_chain if replay_chain is not None else torch.stack(chain_out, dim=1)
                    return x, chain_tensor, log_probs
                return x, log_probs
            if return_chain:
                return x, replay_chain if replay_chain is not None else torch.stack(chain_out, dim=1)
            return x
        finally:
            if was_training:
                self.train()

if __name__ == "__main__":

    flow = FlowPolicy(
        10, 6, 4
    )

    # print(flow(torch.randn((4, 10)), torch.randn((4, 4, 6)), torch.randn((4))).shape)
    flow.sample(torch.randn((4, 10)))

# python3 -m architectures.flow_policy.flow_policy
