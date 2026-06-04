"""REINFORCE — vanilla policy gradient (Williams, 1992).

Base of the on-policy ladder (REINFORCE -> A2C -> PPO). Core idea: increase the
log-probability of actions that led to high return, decrease it for actions that
led to low return.

    grad J = E[ sum_t  grad log pi(a_t | s_t) * G_t ]      G_t = return-to-go

ON-POLICY: collect a fresh episode with the current policy, do ONE update, then
discard it. No replay buffer.

loss: 
    -mean_t(log pi(a_t|s_t) * G_t)
        log pi(a_t|s_t): how probable was the action that we took under the current policy
        G_t: a measure of how good that action turned out to actually be

    this loss function is pretty simple to interpret, expected return across all time steps in episode
    minimizing the loss is equivalent to increasing probability of good actions (get higher reward) and minimizing probability of bad actions (that get lower reward)
    we are doing MLE (maximizing return-weighted log-likelihood of actions we chose)

gradient: 
    the gradient of the log prob indicates the direction we should move our parameters to lead to largest change in probability of selecting the action we did
    we weight that gradient by the return G_t; so for good actions, there's a positive weight and we want params to move heavily in that direction
    alternatively, for bad actions there's a negative G_t and we don't really want to push params in that direction

    policy ends up changing params in a way that increases prob of good actions and decreases prob of bad ones

Run:  python -m algorithms.reinforce
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
LR = 1e-5
HIDDEN = 128


class Policy(nn.Module):
    """maps a state to a categorical distribution over discrete actions"""

    def __init__(self, obs_dim, n_actions):
        super().__init__()

        self.obs_dim = obs_dim
        self.n_actions = n_actions

        self.net = mlp([obs_dim, HIDDEN, HIDDEN, n_actions], output_activation=nn.Identity)  # want identity as final activation instead of ReLU since we don't want to throw away negative logits

    def forward(self, obs):
        return torch.log_softmax(self.net(obs), 0)  # more numerically stable softmax (same output)


class REINFORCE:
    def __init__(self, obs_dim, n_actions):
        self.policy = Policy(obs_dim=obs_dim, n_actions=n_actions)

        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=LR)

    def act(self, obs):
        """ sample an action; return it + its log-prob (kept for the update) """
        action_probs = self.policy(torch.as_tensor(obs))
        action_dist = torch.distributions.Categorical(action_probs.exp())

        action = action_dist.sample()

        return action, action_dist.log_prob(action)

    def update(self, log_probs, rewards):
        G = 0
        t = len(log_probs) - 1
        G_t = torch.Tensor(size=(len(log_probs),))
        while t >= 0:
            G = rewards[t] + GAMMA * G
            G_t[t] = G
            t -= 1
        
        G_t = (G_t - G_t.mean()) / (G_t.std() + 1e-8)  # need standardization so that ~half of the actions have negative returns 

        loss = -torch.mean(torch.stack(log_probs) * G_t)  # use mean instead of sum so that gradient magnitude is comparable between small/large number of steps in episode we are using

        self.policy_opt.zero_grad()
        loss.backward()
        self.policy_opt.step()


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
                action = agent.policy(torch.as_tensor(obs, dtype=torch.float32)).argmax().item()
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

    agent = REINFORCE(obs_dim, n_actions)
    logger = Logger(f"REINFORCE-{env_id}")
    hist = {"steps": [], "returns": []}
    best_ret, best_state = -float("inf"), None  # REINFORCE is unstable -> keep the best policy, not the last

    for ep in range(1, episodes + 1):
        # --- collect ONE fresh episode with the current policy ---
        obs, _ = env.reset(seed=seed + ep)
        log_probs, rewards = [], []
        done = False
        while not done:
            a, log_prob = agent.act(obs)
            next_obs, r, term, trunc, _ = env.step(a.item())
            log_probs.append(log_prob)
            rewards.append(r)
            obs = next_obs

            if term or trunc:
                done = True

        # --- one update, then the episode is discarded (on-policy) ---
        agent.update(log_probs, rewards)

        if ep % eval_every == 0:
            ret = evaluate(agent, env_id)
            logger.log(ep, ret)
            hist["steps"].append(ep)
            hist["returns"].append(ret)
            if ret > best_ret:  # snapshot the best policy seen so far
                best_ret, best_state = ret, copy.deepcopy(agent.policy.state_dict())

    env.close()
    logger.steps, logger.values = hist["steps"], hist["returns"]
    logger.plot(f"algorithms/reinforce_{env_id}.png")
    if best_state is not None:
        agent.policy.load_state_dict(best_state)  # restore the best, not the last
    hist["agent"] = agent
    return hist


if __name__ == "__main__":
    train()
