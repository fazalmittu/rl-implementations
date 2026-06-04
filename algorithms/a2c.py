"""A2C — advantage actor critic.

REINFORCE weights each action's log-prob by the raw return value
- this has extremely high variance as the reward can take on drastically small/large values that lead to slightly unstable gradients
- G_t usually ends up being a noisy number which we want to avoid
    - it can be high bc we took a good action or because the state we began in was already good and REINFORCE can't really differentiate

CORE IDEA: maintain a function that is able to "evaluate" the value of the current state and subtract that from G_t
- this means that our algorithm can now differentiate between just starting in a good state and taking an action that actually leads to one
- A(s_t, a_t) = G_t - V(s_t)

    grad J = E[ sum_t  grad log pi(a_t | s_t) * A_t ]

- subtracting V(s_t) helps reduce variance and keeps the value we use in the loss function more stable

V(s) is approximated by what we call a "critic" in Actor Critic methods

Run:  python -m algorithms.a2c
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

class A2C:
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
        G = 0
        t = len(log_probs) - 1
        G_t = torch.Tensor(size=(len(log_probs),))
        while t >= 0:
            G = rewards[t] + GAMMA * G
            G_t[t] = G
            t -= 1
        
        A_t = (G_t - torch.stack(values)).detach()  
        # A_t = (A_t - A_t.mean()) / (A_t.std() + 1e-8)

        # we detach A_t because we just need it to be a weight in the actor loss
        # if we don't detach, then torch will compute a gradient for A_t
        # when we do actor_loss.backward(), it will adjust the critic params but we don't want that
        # that's considered a "leak", critic will get changed by the wrong objective

        critic_loss = torch.mean((torch.stack(values) - G_t).pow(2))
        actor_loss = -torch.mean(torch.stack(log_probs) * A_t)

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

    agent = A2C(obs_dim, n_actions)
    logger = Logger(f"A2C-{env_id}")
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
    logger.plot(f"algorithms/a2c_{env_id}.png")
    if best_state is not None:
        agent.actor.load_state_dict(best_state)  # restore the best, not the last
    hist["agent"] = agent
    return hist


if __name__ == "__main__":
    train()
