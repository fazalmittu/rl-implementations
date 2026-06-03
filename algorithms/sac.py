"""SAC — Soft Actor Critic algorithm

SAC trains a stochastic policy as well was 2 clipped Q functions 
- key idea: entropy regularization term in critic loss so model trades off between expected return and randomness in policy
- this obviously connects back to the whole exploration-exploitation debate; higher entropy means more exploration, higher return means exploitation
- soft updates to the target networks for critics (polyak averaging)

Run:  python -m algorithms.sac
"""

import copy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.networks import mlp
from common.replay_buffer import ReplayBuffer
from common.logger import Logger

# hyperparams
GAMMA = 0.99          # discount
TAU = 0.005           # target-network Polyak averaging rate
LR = 3e-4             # learning rate (both actor and critic)
HIDDEN = 256          # hidden units per layer
BATCH_SIZE = 256

class Actor(nn.Module):
    """stochastic policy"""

    def __init__(self, obs_dim, action_dim, max_action):
        super().__init__()
        self.net = mlp([obs_dim, HIDDEN, HIDDEN, HIDDEN], output_activation=nn.ReLU)
        self.max_action = max_action

        # 2 heads, mean + std
        self.mean = torch.nn.Linear(HIDDEN, action_dim)
        self.std = torch.nn.Linear(HIDDEN, action_dim)

    def forward(self, obs):
        feat = self.net(obs)
        log_std = self.std(feat).clamp(-5, 2)  # clamp so std can't run away -> NaN
        return self.mean(feat), log_std.exp()  # mean, std


class Critic(nn.Module):
    """Twin Q-networks. Both live here so a single object holds the pair; the
    `min` over them is what tames overestimation."""

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.q1 = mlp([obs_dim + action_dim, HIDDEN, HIDDEN, 1])
        self.q2 = mlp([obs_dim + action_dim, HIDDEN, HIDDEN, 1])

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)


class SAC:
    def __init__(self, obs_dim, action_dim, max_action):
        self.max_action = max_action

        self.actor = Actor(obs_dim, action_dim, max_action)
        self.critic = Critic(obs_dim, action_dim)

        # target net starts as copy
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=LR)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=LR)

        self.q_value = 0.0  # most recent mean critic estimate (for logging)

        # learnable temperature: auto-tunes the entropy coefficient
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=LR)

    def reparamaterize(self, mean, std):
        eps = torch.randn_like(std)
        action = mean + std * eps

        return action

    def sample(self, mean, std):
        """ reparam sample -> tanh-squashed action (scaled) + tanh-corrected log prob """
        u = self.reparamaterize(mean, std)                    
        squashed = nn.functional.tanh(u)
        log_prob = torch.distributions.Normal(mean, std).log_prob(u) - torch.log(1 - squashed.pow(2) + 1e-6)              
        return squashed * self.max_action, log_prob.sum(-1, keepdim=True)

    @torch.no_grad()
    def act(self, obs):
        """ sample an action from the policy for the given obs """
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        action, _ = self.sample(*self.actor(obs))
        return action.squeeze(0)

    def update(self, buffer):
        """ handles actor + critic backprop  """
        obs, action, reward, next_obs, done = buffer.sample(BATCH_SIZE)

        # --- Critic update -------------------------------------------------
        with torch.no_grad():
            next_action, next_logp = self.sample(*self.actor(next_obs))

            target_q1, target_q2 = self.critic_target(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2)
            # Bellman target. (1 - done) zeroes the bootstrap at episode end.
            target_q = reward + GAMMA * (1 - done) * (target_q - self.log_alpha.exp() * next_logp)

        # Regress BOTH critics toward the same target.
        current_q1, current_q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.q_value = current_q1.mean().item()  # log the critic's own estimate

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        action, logp = self.sample(*self.actor(obs))
        alpha = self.log_alpha.exp()

        q1, q2 = self.critic(obs, action)
        actor_loss = (-torch.min(q1, q2) + alpha.detach() * logp).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # push policy entropy toward target_entropy
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # Soft-update both target networks (Polyak averaging).
        self._soft_update(self.critic, self.critic_target)

    def _soft_update(self, net, target):
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.mul_(1 - TAU).add_(TAU * p.data)


# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------
def evaluate(agent, env_id, episodes=5):
    """Greedy (no-noise) return, averaged over a few episodes."""
    env = gym.make(env_id)
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            obs, reward, term, trunc, _ = env.step(agent.act(obs))
            done = term or trunc
            ep_ret += reward
        returns.append(ep_ret)
    env.close()
    return float(np.mean(returns))


def train(env_id="Pendulum-v1", total_steps=30_000, start_steps=1_000,
          eval_every=2_000, expl_noise=0.1, seed=0, label=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = SAC(obs_dim, action_dim, max_action)
    buffer = ReplayBuffer(obs_dim, action_dim)
    logger = Logger(label or f"SAC-{env_id}")
    # data series returned for plotting/comparison
    hist = {"steps": [], "returns": [], "q_values": []}

    obs, _ = env.reset(seed=seed)
    for step in range(1, total_steps + 1):
        # Warm up with random actions to seed the buffer, then act on-policy
        # with exploration noise.
        if step < start_steps:
            action = env.action_space.sample()
        else:
            action = agent.act(obs)

        next_obs, reward, term, trunc, _ = env.step(action)

        # print(obs.shape, action.shape, reward.shape, next_obs.shape)
        buffer.add(obs, action, reward, next_obs, float(term))
        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()

        if step >= start_steps:
            agent.update(buffer)

        if step % eval_every == 0:
            ret = evaluate(agent, env_id)
            logger.log(step, ret)
            hist["steps"].append(step)
            hist["returns"].append(ret)
            hist["q_values"].append(agent.q_value)

    env.close()
    hist["agent"] = agent
    return hist


if __name__ == "__main__":
    hist = train()
    logger = Logger("SAC-Pendulum-v1")
    logger.steps, logger.values = hist["steps"], hist["returns"]
    logger.plot("algorithms/sac_Pendulum-v1.png")
