"""TD3 — Twin Delayed Deep Deterministic policy gradient.

TD3 is DDPG plus three tricks that fix Q-value overestimation:

1. TWIN CRITICS (clipped double-Q): train 2 critics and use the smaller of the two when forming the target so overestimation doesn't propagate
2. DELAYED POLICY UPDATES: update the actor (and all target nets) less often than the critics so the policy chases a more stable value estimate.
3. TARGET POLICY SMOOTHING: add noise to the target action so the Q function is more smooth and critic can't exploit large spikes

Run:  python -m algorithms.td3
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
POLICY_NOISE = 0.2    # std of target-smoothing noise        
NOISE_CLIP = 0.5      # how far that noise can reach          
POLICY_DELAY = 2      # actor updated every N critic updates

class Actor(nn.Module):
    """Deterministic policy: state -> action, tanh-squashed and scaled to the
    environment's action bounds (assumed symmetric, as in Pendulum/MuJoCo)."""

    def __init__(self, obs_dim, action_dim, max_action):
        super().__init__()
        self.net = mlp([obs_dim, HIDDEN, HIDDEN, action_dim], output_activation=nn.Tanh)
        self.max_action = max_action

    def forward(self, obs):
        return self.max_action * self.net(obs)


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


class TD3:
    def __init__(self, obs_dim, action_dim, max_action,
                 use_target_actor=True, use_target_critic=True):
        self.max_action = max_action
        # ablation switches (both True == normal TD3)
        self.use_target_actor = use_target_actor
        self.use_target_critic = use_target_critic

        self.actor = Actor(obs_dim, action_dim, max_action)
        self.critic = Critic(obs_dim, action_dim)

        # target nets start as copies
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=LR)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=LR)

        self.total_it = 0  # counts critic updates, drives the delayed actor step
        self.q_value = 0.0  # most recent mean critic estimate (for logging)

    @torch.no_grad()
    def act(self, obs, noise=0.0):
        """ function agent uses to take an action given an obs with some noise added + clipping """
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        action = self.actor(obs).numpy().flatten()
        if noise:  # exploration noise during data collection
            action += noise * self.max_action * np.random.randn(*action.shape)
        return action.clip(-self.max_action, self.max_action)

    def update(self, buffer):
        """ handles actor + critic backprop  """
        self.total_it += 1
        obs, action, reward, next_obs, done = buffer.sample(BATCH_SIZE)

        # ablation experiment: which nets build the target. Default = the slow target copies
        actor_net = self.actor_target if self.use_target_actor else self.actor
        critic_net = self.critic_target if self.use_target_critic else self.critic

        # --- Critic update -------------------------------------------------
        with torch.no_grad():
            # trick 3: gaussian smoothing of action
            noise = (torch.randn_like(action) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
            next_action = (actor_net(next_obs) + noise).clamp(
                -self.max_action, self.max_action)

            # trick 1: take min of the 2 critics
            target_q1, target_q2 = critic_net(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2)
            # bellman target. (1 - done) zeroes the bootstrap at episode end
            target_q = reward + GAMMA * (1 - done) * target_q

        # Regress BOTH critics toward the same target
        current_q1, current_q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.q_value = current_q1.mean().item()  # log the critic's own estimate

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # --- Delayed actor + target update ---------------------------------
        # TRICK 2: only every POLICY_DELAY critic steps.
        if self.total_it % POLICY_DELAY == 0:
            # deterministic policy gradient: push actions toward higher Q
            # use just the first critic
            q1, _ = self.critic(obs, self.actor(obs))
            actor_loss = -q1.mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # Soft-update both target networks (Polyak averaging).
            self._soft_update(self.critic, self.critic_target)
            self._soft_update(self.actor, self.actor_target)

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
          eval_every=2_000, expl_noise=0.1, seed=0,
          use_target_actor=True, use_target_critic=True, label=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3(obs_dim, action_dim, max_action,
                use_target_actor=use_target_actor, use_target_critic=use_target_critic)
    buffer = ReplayBuffer(obs_dim, action_dim)
    logger = Logger(label or f"TD3-{env_id}")
    # data series returned for plotting/comparison
    hist = {"steps": [], "returns": [], "q_values": []}

    obs, _ = env.reset(seed=seed)
    for step in range(1, total_steps + 1):
        # warm up with random actions to seed the buffer, then act on-policy with exploration noise.
        if step < start_steps:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, noise=expl_noise)

        next_obs, reward, term, trunc, _ = env.step(action)
        # only `term` is a true environment terminal; `trunc` (time limit) must
        # NOT zero the bootstrap, or we'd teach the agent the world ends at the
        # time limit. pendulum never truly terminates so done is always 0 here.
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
    logger = Logger("TD3-Pendulum-v1")
    logger.steps, logger.values = hist["steps"], hist["returns"]
    logger.plot("algorithms/td3_Pendulum-v1.png")
