"""GAE — generalized advantage estimator

- so A2C tried to fix the issue of the return being too high variance across different episodes however it still isn't enough
- the fact that we use G_t at all in the critic loss function leads to high variance since it's such an unpredictable value
- if we stop for a second and think about why it's so unstable, it's because we are trying to accumulate rewards across a potentially very long episode
- obviously the value is going to be very unpredictable since it can change drastically at so many different turns
- so let's step back and think what's the opposite end of this spectrum?
- well instead of using all t steps in the episode, what if we just used 1?
- as in:
    A_t = r_t + GAMMA * V(s_t+1) - V(s_t) [1 TD step]
    
    this is lower variance but higher bias since now we count a lot more on the critic's value (might not be accurate)
    remember high bias = high systematic error, high variance = high noise

- so we want to find some balance between these high variance/low variance methods of estimating advantage

GAE introduces:

    A_t = sum (i = 0 -> inf) of [(GAMMA * LAMBDA)^i * (r_t + GAMMA * V(s_t+i) - V(s_t+i))

    LAMBDA = 0: we have A_t = r_t + GAMMA * V(s_t+1) - V(s_t) [1 TD step]
    LAMBDA = 1: we have A_t ~ G_t - V(s_t) [A2C advantage function]

    so we can tweak LAMBDA to whatever we desire to get a value suitable for whatever task we are doing

Run:  python -m algorithms.gae
"""

import copy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from common.networks import mlp
from common.logger import Logger

# hyperparams
GAMMA = 0.99
LAMBDA = 0.95
LR = 1e-2
HIDDEN = 128


class Actor(nn.Module):
    """maps a state to a categorical distribution over discrete actions"""

    def __init__(self, obs_dim, n_actions):
        super().__init__()

        self.obs_dim = obs_dim
        self.n_actions = n_actions

        self.net = mlp([obs_dim, HIDDEN, HIDDEN, n_actions], output_activation=nn.Identity)  # want identity as final activation instead of ReLU since we don't want to throw away negative logits

    def forward(self, obs):
        return torch.log_softmax(self.net(obs), 0)  # more numerically stable softmax (same output)

class Critic(nn.Module):
    """ returns a value for the current state """
    def __init__(self, obs_dim):
        super().__init__()

        self.obs_dim = obs_dim

        self.net = mlp([obs_dim, HIDDEN, HIDDEN, 1], output_activation=nn.Identity)
    
    def forward(self, obs):
        return self.net(obs)

class GAE:
    def __init__(self, obs_dim, n_actions):
        self.actor = Actor(obs_dim, n_actions)
        self.critic = Critic(obs_dim)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=LR)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=LR)

    def act(self, obs):
        action_probs = self.actor(torch.as_tensor(obs))
        action_dist = torch.distributions.Categorical(action_probs.exp())

        action = action_dist.sample()

        return action, action_dist.log_prob(action)


    def update(self, log_probs, rewards, values):
        T = len(log_probs)
        A = 0
        # t = len(log_probs) - 2
        delta_t = torch.zeros(T)
        A_t = torch.zeros(T)

        # compute all delta_t's
        for i in range(len(values)):                     
            next_value = values[i + 1] if i + 1 < len(values) else 0.0
            delta_t[i] = rewards[i] + GAMMA * next_value - values[i]

        for t in reversed(range(T)):
            A = delta_t[t] + GAMMA * LAMBDA * A
            A_t[t] = A

        # we detach A_t because we just need it to be a weight in the actor loss
        # if we don't detach, then torch will compute a gradient for A_t
        # when we do actor_loss.backward(), it will adjust the critic params but we don't want that
        # that's considered a "leak", critic will get changed by the wrong objective

        target = (A_t + torch.stack(values)).detach() 
        critic_loss = torch.mean((torch.stack(values) - target).pow(2))
        actor_loss = -torch.mean(torch.stack(log_probs) * A_t.detach())

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()


# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------
def evaluate(agent, env_id, episodes=10):
    """Greedy (argmax) return, averaged over a few episodes."""
    env = gym.make(env_id)
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, ep_ret = False, 0.0
        while not done:
            with torch.no_grad():
                action = agent.actor(torch.as_tensor(obs, dtype=torch.float32)).argmax().item()
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            ep_ret += reward
        returns.append(ep_ret)
    env.close()
    return float(np.mean(returns))


def train(env_id="CartPole-v1", episodes=1000, eval_every=10, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = GAE(obs_dim, n_actions)
    logger = Logger(f"GAE-{env_id}")
    hist = {"steps": [], "returns": []}
    best_ret, best_state = -float("inf"), None

    for ep in range(1, episodes + 1):
        # --- collect ONE fresh episode with the current actor ---
        obs, _ = env.reset(seed=seed + ep)
        log_probs, rewards, values = [], [], []
        done = False
        while not done:
            a, log_prob = agent.act(obs)
            next_obs, r, term, trunc, _ = env.step(a.item())
            log_probs.append(log_prob)
            rewards.append(r)
            values.append(agent.critic(torch.as_tensor(obs)))
            obs = next_obs

            if term or trunc:
                done = True
                # rewards.append(0)
                # values.append(0)

        # --- one update, then the episode is discarded (on-policy) ---
        agent.update(log_probs, rewards, values)

        if ep % eval_every == 0:
            ret = evaluate(agent, env_id)
            logger.log(ep, ret)
            hist["steps"].append(ep)
            hist["returns"].append(ret)
            if ret > best_ret:  # snapshot the best policy seen so far
                best_ret, best_state = ret, copy.deepcopy(agent.actor.state_dict())

    env.close()
    logger.steps, logger.values = hist["steps"], hist["returns"]
    logger.plot(f"algorithms/gae_{env_id}.png")
    if best_state is not None:
        agent.actor.load_state_dict(best_state)  # restore the best, not the last
    hist["agent"] = agent
    return hist


if __name__ == "__main__":
    train()
