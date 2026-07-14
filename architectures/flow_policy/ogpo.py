import argparse
import copy
from collections import deque
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from architectures.flow_policy.critic import REDQCritic
from architectures.flow_policy.eval_rollout import (
    eval_action_fn,
    is_success,
    make_env,
    obs_to_tensor,
    unnormalize_action,
)
from architectures.flow_policy.flow_policy import FlowPolicy
from architectures.flow_policy.train import get_device, to_cpu
from common.replay_buffer import ReplayBuffer


class SuccessBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int, device: str):
        self.capacity = capacity
        self.device = device
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action):
        self.obs[self.ptr] = obs
        self.action[self.ptr] = action
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_episode(self, transitions: list[dict]):
        for transition in transitions:
            self.add(transition["obs"], transition["action"])

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], device=self.device),
            torch.as_tensor(self.action[idx], device=self.device),
        )

    def __len__(self):
        return self.size


class OGPO(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_horizon: int = 4,
        hidden_size: int = 256,
        num_layers: int = 4,
        time_embed_dim: int = 64,
        num_qs: int = 10,
        num_min: int = 2,
        critic_hidden_size: int | None = None,
        q_agg: str = "mean",
        gamma: float = 0.99,
        tau: float = 0.05,
        lr: float = 3e-4,
        ppo_lr: float | None = None,
        critic_lr: float | None = None,
        weight_decay: float = 0.0,
        actor_weight_decay: float | None = None,
        critic_weight_decay: float | None = None,
        grad_clip: float | None = 10.0,
        batch_size: int = 256,
        sample_steps: int = 10,
        discount_steps: int | None = None,
        actor_samples: int = 4,
        best_of_n: int = 1,
        flow_noise_std: float = 0.1,
        debias_flow_noise: bool = True,
        ppo_clip: float = 0.2,
        actor_update_epochs: int = 2,
        adv_strategy: str = "vanilla",
        normalize_group: bool = False,
        bc_coeff: float = 0.0,
        actor_scheduler: str = "constant",
        actor_warmup_steps: int = 2000,
        actor_decay_steps: int = 50000,
        actor_end_value: float = 2e-5,
        critic_scheduler: str = "constant",
        critic_warmup_steps: int = 500,
        critic_decay_steps: int = 5000,
        critic_end_value: float = 1e-8,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.flat_action_dim = action_horizon * action_dim
        self.num_min = num_min
        self.q_agg = q_agg
        self.gamma = gamma
        self.discount = gamma ** (discount_steps or action_horizon)
        self.tau = tau
        self.grad_clip = grad_clip
        self.batch_size = batch_size
        self.sample_steps = sample_steps
        self.actor_samples = actor_samples
        self.best_of_n = best_of_n
        self.flow_noise_std = flow_noise_std
        self.debias_flow_noise = debias_flow_noise
        self.ppo_clip = ppo_clip
        self.actor_update_epochs = actor_update_epochs
        self.adv_strategy = adv_strategy
        self.normalize_group = normalize_group
        self.bc_coeff = bc_coeff
        self.actor_scheduler = actor_scheduler
        self.actor_warmup_steps = actor_warmup_steps
        self.actor_decay_steps = actor_decay_steps
        self.actor_end_value = actor_end_value
        self.critic_scheduler = critic_scheduler
        self.critic_warmup_steps = critic_warmup_steps
        self.critic_decay_steps = critic_decay_steps
        self.critic_end_value = critic_end_value
        self.actor_updates = 0
        self.critic_updates = 0
        self.actor_lr = ppo_lr or lr
        self.critic_lr = critic_lr or lr

        self.actor = FlowPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_horizon=action_horizon,
            hidden_size=hidden_size,
            num_layers=num_layers,
            time_embed_dim=time_embed_dim,
        )

        self.critic = REDQCritic(
            obs_dim=obs_dim,
            action_dim=self.flat_action_dim,
            num_qs=num_qs,
            hidden_size=critic_hidden_size or hidden_size,
        )
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(),
            lr=self.actor_lr,
            weight_decay=actor_weight_decay if actor_weight_decay is not None else weight_decay,
        )
        self.critic_opt = torch.optim.AdamW(
            self.critic.parameters(),
            lr=self.critic_lr,
            weight_decay=critic_weight_decay if critic_weight_decay is not None else weight_decay,
        )

    @staticmethod
    def scheduled_lr(
        scheduler: str,
        step: int,
        base_lr: float,
        warmup_steps: int,
        decay_steps: int,
        end_value: float,
    ) -> float:
        if scheduler == "constant":
            return base_lr
        if scheduler != "cosine":
            raise ValueError(f"unknown scheduler: {scheduler}")

        period = max(decay_steps, 1)
        step = step % period
        if warmup_steps > 0 and step < warmup_steps:
            pct = float(step + 1) / float(warmup_steps)
            return end_value + pct * (base_lr - end_value)

        denom = max(period - warmup_steps, 1)
        pct = min(max((step - warmup_steps) / denom, 0.0), 1.0)
        return end_value + 0.5 * (base_lr - end_value) * (1.0 + math.cos(math.pi * pct))

    @staticmethod
    def set_lr(optimizer: torch.optim.Optimizer, lr: float):
        for group in optimizer.param_groups:
            group["lr"] = lr

    def current_actor_lr(self) -> float:
        return self.scheduled_lr(
            self.actor_scheduler,
            self.actor_updates,
            self.actor_lr,
            self.actor_warmup_steps,
            self.actor_decay_steps,
            self.actor_end_value,
        )

    def current_critic_lr(self) -> float:
        return self.scheduled_lr(
            self.critic_scheduler,
            self.critic_updates,
            self.critic_lr,
            self.critic_warmup_steps,
            self.critic_decay_steps,
            self.critic_end_value,
        )

    def compute_advantage(self, q_full: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q_full = q_full.squeeze(-1).view(self.critic.num_qs, self.batch_size, self.actor_samples)
        q_mean = q_full.mean(dim=0)

        if self.adv_strategy == "conservative":
            adv_per_q = q_full - q_full.mean(dim=2, keepdim=True)
            adv_min = adv_per_q.min(dim=0).values
            adv_max = adv_per_q.max(dim=0).values
            advantage = adv_min * (adv_min > 0.0) + adv_max * (adv_max < 0.0)
        elif self.adv_strategy == "vanilla":
            q = self.aggregate_sample_q(q_full)
            advantage = q - q.mean(dim=1, keepdim=True)
            if self.normalize_group:
                advantage = advantage / (q.std(dim=1, keepdim=True, unbiased=False) + 1e-6)
        else:
            raise ValueError(f"unknown advantage strategy: {self.adv_strategy}")

        return advantage.flatten(), q_mean

    def aggregate_sample_q(self, q_full: torch.Tensor) -> torch.Tensor:
        if self.q_agg == "mean":
            return q_full.mean(dim=0)
        if self.q_agg == "min":
            return q_full.min(dim=0).values
        if self.q_agg == "subsample":
            if self.num_min >= self.critic.num_qs:
                return q_full.min(dim=0).values
            idx = torch.randperm(self.critic.num_qs, device=q_full.device)[: self.num_min]
            return q_full[idx].min(dim=0).values
        raise ValueError(f"unknown q aggregation: {self.q_agg}")

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        stochastic: bool = False,
        best_of_n: int | None = None,
    ) -> torch.Tensor:
        n = best_of_n or self.best_of_n
        if n <= 1:
            return self.actor.sample(
                obs,
                num_steps=self.sample_steps,
                return_chain=False,
                noise_std=self.flow_noise_std if stochastic else 0.0,
                debias_noise=self.debias_flow_noise,
            )

        batch_size = obs.shape[0]
        obs_rep = obs.repeat_interleave(n, dim=0)
        actions = self.actor.sample(
            obs_rep,
            num_steps=self.sample_steps,
            return_chain=False,
            noise_std=self.flow_noise_std if stochastic else 0.0,
            debias_noise=self.debias_flow_noise,
        )
        q = self.critic_target.aggregate(
            obs_rep,
            actions.flatten(start_dim=1),
            method=self.q_agg,
            num_min=self.num_min,
        ).squeeze(-1)
        best_idx = q.view(batch_size, n).argmax(dim=1)
        actions = actions.view(batch_size, n, self.action_horizon, self.action_dim)
        return actions[torch.arange(batch_size, device=obs.device), best_idx]

    def update(
        self,
        buffer: ReplayBuffer,
        success_buffer: SuccessBuffer | None = None,
    ) -> dict[str, float]:
        obs, action, reward, next_obs, done = buffer.sample(self.batch_size)
        # obs, next_obs: (B, obs_dim)
        # action:        (B, action_horizon * action_dim)
        # reward, done:  (B, 1)

        # --- offline / replay update: REDQ critic TD learning ---
        with torch.no_grad():
            next_action = self.actor.sample(
                next_obs,
                num_steps=self.sample_steps,
                return_chain=False,
            ).flatten(start_dim=1)
            target_q = self.critic_target.aggregate(
                next_obs,
                next_action,
                method=self.q_agg,
                num_min=self.num_min,
            )
            target_q = reward + (1.0 - done) * self.discount * target_q

        current_q = self.critic(obs, action)
        critic_loss = F.mse_loss(current_q, target_q.expand_as(current_q))

        critic_lr = self.current_critic_lr()
        self.set_lr(self.critic_opt, critic_lr)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_opt.step()
        self.critic_updates += 1
        self.soft_update(self.critic, self.critic_target)

        # --- online update: score sampled flow chains with Q and update actor ---
        obs_rep = obs.repeat_interleave(self.actor_samples, dim=0)
        # obs_rep: (B * actor_samples, obs_dim)

        with torch.no_grad():
            sampled_actions, chain, old_log_probs = self.actor.sample(
                obs_rep,
                num_steps=self.sample_steps,
                return_chain=True,
                noise_std=self.flow_noise_std,
                return_log_probs=True,
                debias_noise=self.debias_flow_noise,
            ) # sample actor_samples chains per obs and freeze their old log probs (bc torch.no_grad())

        # sampled_actions: (B * actor_samples, action_horizon, action_dim)
        # chain:           (B * actor_samples, sample_steps + 1, action_horizon, action_dim)
        # old_log_probs:   (B * actor_samples, sample_steps)
        sampled_actions = sampled_actions.flatten(start_dim=1)
        old_log_prob = old_log_probs.sum(dim=1)     # sum log probs across each action chain
        # sampled_actions: (B * actor_samples, action_horizon * action_dim)
        # old_log_prob:    (B * actor_samples,)

        with torch.no_grad():
            q_full = self.critic(obs_rep, sampled_actions)    # we repeated obs bc same obs used for each actor_sample we do
            advantage, q = self.compute_advantage(q_full)
            # q:         (B, actor_samples)
            # advantage: (B * actor_samples,)

        bc_loss = torch.tensor(0.0, device=obs.device)
        success_obs = success_action = None
        if self.bc_coeff > 0.0 and success_buffer is not None and len(success_buffer) > 0:
            success_obs, success_action = success_buffer.sample(self.batch_size)
            success_action = success_action.view(self.batch_size, self.action_horizon, self.action_dim)

        # here's the actual online update in which we do self.actor_update_epochs per update()
        # for each higher level update() we take output chains from our flow model (B * actor_sample times)
        # for each actor_update, we want to see how the new actor's log probs compare to the higher level chains 
        # as more epochs happen we want to push actor to give higher advantage actions more likelihood of happening
        actor_loss = torch.tensor(0.0, device=obs.device)
        pg_loss = torch.tensor(0.0, device=obs.device)
        actor_lr = self.current_actor_lr()
        for _ in range(self.actor_update_epochs):
            actor_lr = self.current_actor_lr()
            self.set_lr(self.actor_opt, actor_lr)
            _, log_probs = self.actor.sample(
                obs_rep,
                num_steps=self.sample_steps,
                return_chain=False,
                noise_std=self.flow_noise_std,
                return_log_probs=True,
                chain=chain,
                debias_noise=self.debias_flow_noise,
            ) 
            log_prob = log_probs.sum(dim=1)

            ratio = torch.exp(log_prob - old_log_prob)
            clipped_ratio = ratio.clamp(1.0 - self.ppo_clip, 1.0 + self.ppo_clip)
            pg_loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()
            if success_obs is not None:
                bc_loss = self.actor.flow_matching_loss(success_obs, success_action)
            actor_loss = pg_loss + self.bc_coeff * bc_loss

            self.actor_opt.zero_grad()
            actor_loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.actor_opt.step()
            self.actor_updates += 1

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "pg_loss": pg_loss.item(),
            "bc_loss": bc_loss.item(),
            "q": current_q.mean().item(),
            "target_q": target_q.mean().item(),
            "reward": reward.mean().item(),
            "actor_q": q.mean().item(),
            "advantage_abs": advantage.abs().mean().item(),
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "success_buffer": float(len(success_buffer)) if success_buffer is not None else 0.0,
        }

    def soft_update(self, net: nn.Module, target: nn.Module):
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)


def load_agent(args: argparse.Namespace, device: str):
    checkpoint = torch.load(args.policy_checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    args.execute_horizon = args.execute_horizon or config["action_horizon"]

    agent = OGPO(
        obs_dim=config["obs_dim"],
        action_dim=config["action_dim"],
        action_horizon=config["action_horizon"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        time_embed_dim=config["time_embed_dim"],
        num_qs=args.num_qs,
        num_min=args.num_min,
        critic_hidden_size=args.critic_hidden_size or args.hidden_size,
        q_agg=args.q_agg,
        gamma=args.gamma,
        tau=args.tau,
        lr=args.lr,
        ppo_lr=args.ppo_lr,
        critic_lr=args.critic_lr,
        weight_decay=args.weight_decay,
        actor_weight_decay=args.actor_weight_decay,
        critic_weight_decay=args.critic_weight_decay,
        grad_clip=args.grad_clip,
        batch_size=args.batch_size,
        sample_steps=args.sample_steps,
        discount_steps=args.execute_horizon,
        actor_samples=args.actor_samples,
        best_of_n=args.best_of_n,
        flow_noise_std=args.flow_noise_std,
        debias_flow_noise=args.debias_flow_noise,
        ppo_clip=args.ppo_clip,
        actor_update_epochs=args.actor_update_epochs,
        adv_strategy=args.adv_strategy,
        normalize_group=args.normalize_group,
        bc_coeff=args.bc_coeff if args.use_bc_regularization else 0.0,
        actor_scheduler=args.actor_scheduler,
        actor_warmup_steps=args.actor_warmup_steps,
        actor_decay_steps=args.actor_decay_steps,
        actor_end_value=args.actor_end_value,
        critic_scheduler=args.critic_scheduler,
        critic_warmup_steps=args.critic_warmup_steps,
        critic_decay_steps=args.critic_decay_steps,
        critic_end_value=args.critic_end_value,
    ).to(device)
    agent.actor.load_state_dict(checkpoint["model_state_dict"])
    if args.resume is not None:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        agent.load_state_dict(resume["agent_state_dict"])
        if "actor_optimizer_state_dict" in resume:
            agent.actor_opt.load_state_dict(resume["actor_optimizer_state_dict"])
        if "critic_optimizer_state_dict" in resume:
            agent.critic_opt.load_state_dict(resume["critic_optimizer_state_dict"])
        train_config = resume.get("train_config", {})
        chunks = int(train_config.get("chunks", 0))
        updates_per_chunk = int(train_config.get("updates_per_chunk", args.updates_per_chunk))
        actor_update_epochs = int(train_config.get("actor_update_epochs", args.actor_update_epochs))
        agent.critic_updates = int(resume.get("critic_updates", chunks * updates_per_chunk))
        agent.actor_updates = int(resume.get("actor_updates", agent.critic_updates * actor_update_epochs))
        print(f"resumed OGPO checkpoint {args.resume}", flush=True)
    agent.train()

    return checkpoint, agent


def collect_chunk(
    env,
    obs: dict,
    episode_steps: int,
    agent: OGPO,
    obs_keys: tuple[str, ...],
    checkpoint: dict,
    args: argparse.Namespace,
    device: str,
) -> dict:
    obs_tensor = obs_to_tensor(obs, obs_keys, checkpoint, device)
    action_norm = agent.act(obs_tensor, stochastic=args.stochastic_rollout)[0]
    action_env = unnormalize_action(action_norm, checkpoint).detach().cpu().numpy()
    if args.clip_actions:
        action_env = np.clip(action_env, -1.0, 1.0)

    next_obs = obs
    reward_sum = 0.0
    raw_reward_sum = 0.0
    success = False
    done = False
    steps = 0

    for i, action in enumerate(action_env[: args.execute_horizon]):
        next_obs, reward, env_done, _ = env.step(action)
        reward = float(reward)
        reward_sum += (args.gamma ** i) * reward
        raw_reward_sum += reward
        steps += 1
        success = is_success(env)
        done = (
            success
            or (env_done and not args.ignore_done)
            or episode_steps + steps >= args.horizon
        )
        if done:
            break

    next_obs_tensor = obs_to_tensor(next_obs, obs_keys, checkpoint, device)
    return {
        "obs": obs_tensor.squeeze(0).detach().cpu().numpy(),
        "action": action_norm.flatten().detach().cpu().numpy(),
        "reward": reward_sum,
        "raw_reward": raw_reward_sum,
        "next_obs": next_obs_tensor.squeeze(0).detach().cpu().numpy(),
        "done": float(done),
        "success": float(success),
        "steps": steps,
        "next_env_obs": next_obs,
    }


def add_to_buffer(buffer: ReplayBuffer, transition: dict):
    buffer.add(
        transition["obs"],
        transition["action"],
        transition["reward"],
        transition["next_obs"],
        transition["done"],
    )


def seed_replay(
    env,
    buffer: ReplayBuffer,
    success_buffer: SuccessBuffer | None,
    agent: OGPO,
    obs_keys: tuple[str, ...],
    checkpoint: dict,
    args: argparse.Namespace,
    device: str,
) -> dict[str, float]:
    obs = env.reset()
    episode_steps = 0
    episode_return = 0.0
    episodes = 0
    successes = []
    returns = []
    total_steps = 0
    episode_transitions = []

    while len(buffer) < args.start_transitions:
        transition = collect_chunk(
            env, obs, episode_steps, agent, obs_keys, checkpoint, args, device
        )
        add_to_buffer(buffer, transition)
        episode_transitions.append(transition)
        episode_steps += transition["steps"]
        episode_return += transition["raw_reward"]
        total_steps += transition["steps"]

        if transition["done"]:
            if transition["success"] and success_buffer is not None:
                success_buffer.add_episode(episode_transitions)
            episodes += 1
            successes.append(transition["success"])
            returns.append(episode_return)
            obs = env.reset()
            episode_steps = 0
            episode_return = 0.0
            episode_transitions = []
        else:
            obs = transition["next_env_obs"]

    return {
        "episodes": float(episodes),
        "steps": float(total_steps),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
    }


def save_checkpoint(
    path: str | Path,
    agent: OGPO,
    args: argparse.Namespace,
    env_steps: int,
    chunks: int,
    buffer_size: int,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "agent_state_dict": to_cpu(agent.state_dict()),
            "actor_optimizer_state_dict": to_cpu(agent.actor_opt.state_dict()),
            "critic_optimizer_state_dict": to_cpu(agent.critic_opt.state_dict()),
            "actor_updates": agent.actor_updates,
            "critic_updates": agent.critic_updates,
            "policy_checkpoint": args.policy_checkpoint,
            "train_config": {
                **vars(args),
                "env_steps": env_steps,
                "chunks": chunks,
                "buffer_size": buffer_size,
            },
        },
        path,
    )


def init_wandb(args: argparse.Namespace, checkpoint: dict, agent: OGPO, device: str):
    if not args.wandb:
        return None

    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        config={
            **vars(args),
            "device": device,
            "obs_dim": agent.obs_dim,
            "action_dim": agent.action_dim,
            "action_horizon": agent.action_horizon,
            "flat_action_dim": agent.flat_action_dim,
            "policy_model_config": checkpoint["model_config"],
            "policy_train_config": checkpoint.get("train_config", {}),
        },
    )
    wandb.define_metric("agent_steps")
    wandb.define_metric("seed/*", step_metric="agent_steps")
    wandb.define_metric("train/*", step_metric="agent_steps")
    wandb.define_metric("rollout/*", step_metric="agent_steps")
    wandb.define_metric("replay/*", step_metric="agent_steps")
    wandb.define_metric("update/*", step_metric="agent_steps")
    wandb.define_metric("eval/*", step_metric="agent_steps")
    return run


def run_eval(
    args: argparse.Namespace,
    agent: OGPO,
    checkpoint: dict,
    device: str,
    agent_steps: int,
) -> dict[str, float]:
    was_training = agent.training
    agent.eval()
    try:
        metrics = eval_action_fn(
            lambda obs: agent.act(obs, stochastic=False)[0],
            checkpoint,
            args.dataset,
            device,
            episodes=args.eval_episodes,
            horizon=args.eval_horizon,
            execute_horizon=args.eval_execute_horizon or args.execute_horizon,
            clip_actions=args.clip_actions,
            ignore_done=args.ignore_done,
            seed=args.eval_seed,
        )
    finally:
        if was_training:
            agent.train()

    print(
        f"eval agent_steps {agent_steps}"
        f" | success_rate {metrics['success_rate']:.3f}"
        f" | mean_return {metrics['mean_return']:.3f}"
        f" | mean_length {metrics['mean_length']:.1f}",
        flush=True,
    )
    return metrics


def train(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device(args.device)
    checkpoint, agent = load_agent(args, device)
    args.execute_horizon = args.execute_horizon or agent.action_horizon
    if args.execute_horizon <= 0 or args.execute_horizon > agent.action_horizon:
        raise ValueError(f"execute_horizon must be in [1, {agent.action_horizon}]")

    obs_keys = tuple(checkpoint["dataset"]["obs_keys"])
    buffer = ReplayBuffer(
        agent.obs_dim,
        agent.flat_action_dim,
        capacity=args.replay_size,
        device=device,
    )
    success_buffer = None
    if args.use_success_buffer:
        success_buffer = SuccessBuffer(
            agent.obs_dim,
            agent.flat_action_dim,
            capacity=args.success_buffer_size,
            device=device,
        )
    env = make_env(args.dataset, record_video=False)
    wandb_run = init_wandb(args, checkpoint, agent, device)

    print(
        f"OGPO: start_transitions={args.start_transitions}"
        f" | total_env_steps={args.total_env_steps}"
        f" | action_horizon={agent.action_horizon}"
        f" | execute_horizon={args.execute_horizon}"
        f" | best_of_n={agent.best_of_n}"
        f" | adv_strategy={agent.adv_strategy}"
        f" | device={device}",
        flush=True,
    )

    next_log = args.log_every
    next_save = args.save_every
    next_eval = args.eval_every
    recent_successes = deque(maxlen=20)
    start_t = time.perf_counter()

    try:
        seed_stats = seed_replay(
            env, buffer, success_buffer, agent, obs_keys, checkpoint, args, device
        )
        print(
            f"seeded replay: transitions={len(buffer)}"
            f" | success_buffer={len(success_buffer) if success_buffer is not None else 0}"
            f" | agent_steps={int(seed_stats['steps'])}"
            f" | episodes={int(seed_stats['episodes'])}"
            f" | success_rate={seed_stats['success_rate']:.3f}"
            f" | mean_return={seed_stats['mean_return']:.3f}",
            flush=True,
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "agent_steps": int(seed_stats["steps"]),
                    "seed/episodes": seed_stats["episodes"],
                    "seed/agent_steps": seed_stats["steps"],
                    "seed/success_rate": seed_stats["success_rate"],
                    "seed/mean_return": seed_stats["mean_return"],
                    "replay/size": len(buffer),
                    "replay/success_buffer_size": len(success_buffer) if success_buffer is not None else 0,
                },
                step=int(seed_stats["steps"]),
            )

        obs = env.reset()
        episode_steps = 0
        agent_steps = int(seed_stats["steps"])
        chunks = 0
        last_metrics = {}
        episode_transitions = []

        while agent_steps < args.total_env_steps:
            transition = collect_chunk(
                env, obs, episode_steps, agent, obs_keys, checkpoint, args, device
            )
            add_to_buffer(buffer, transition)
            episode_transitions.append(transition)
            chunks += 1
            agent_steps += transition["steps"]
            episode_steps += transition["steps"]

            for _ in range(args.updates_per_chunk):
                if len(buffer) >= args.batch_size:
                    last_metrics = agent.update(buffer, success_buffer)

            if transition["done"]:
                if transition["success"] and success_buffer is not None:
                    success_buffer.add_episode(episode_transitions)
                recent_successes.append(transition["success"])
                obs = env.reset()
                episode_steps = 0
                episode_transitions = []
            else:
                obs = transition["next_env_obs"]

            if agent_steps >= next_log:
                recent_success = float(np.mean(recent_successes)) if recent_successes else 0.0
                metrics = " ".join(
                    f"| {key} {value:.4f}" for key, value in last_metrics.items()
                )
                if metrics:
                    metrics = f" {metrics}"
                print(
                    f"agent_steps {agent_steps}"
                    f" | chunks {chunks}"
                    f" | buffer {len(buffer)}"
                    f" | recent_success {recent_success:.3f}"
                    f"{metrics}",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_log = {
                        "agent_steps": agent_steps,
                        "train/agent_steps": agent_steps,
                        "train/chunks": chunks,
                        "train/elapsed_s": time.perf_counter() - start_t,
                        "rollout/recent_success": recent_success,
                        "rollout/chunk_reward": transition["raw_reward"],
                        "rollout/chunk_steps": transition["steps"],
                        "replay/size": len(buffer),
                        "replay/success_buffer_size": len(success_buffer) if success_buffer is not None else 0,
                    }
                    wandb_log.update({f"update/{key}": value for key, value in last_metrics.items()})
                    wandb_run.log(wandb_log, step=agent_steps)
                while agent_steps >= next_log:
                    next_log += args.log_every

            if args.eval_every is not None and agent_steps >= next_eval:
                metrics = run_eval(args, agent, checkpoint, device, agent_steps)
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "agent_steps": agent_steps,
                            "eval/success_rate": metrics["success_rate"],
                            "eval/mean_return": metrics["mean_return"],
                            "eval/mean_length": metrics["mean_length"],
                        },
                        step=agent_steps,
                    )
                while agent_steps >= next_eval:
                    next_eval += args.eval_every

            if args.save_every is not None and agent_steps >= next_save:
                save_path = Path(args.output)
                save_path = save_path.with_name(
                    f"{save_path.stem}_env_{agent_steps}{save_path.suffix}"
                )
                save_checkpoint(save_path, agent, args, agent_steps, chunks, len(buffer))
                print(f"saved checkpoint to {save_path}", flush=True)
                while agent_steps >= next_save:
                    next_save += args.save_every

        save_checkpoint(args.output, agent, args, agent_steps, chunks, len(buffer))
        print(f"saved final checkpoint to {args.output}", flush=True)
    finally:
        env.close()
        if wandb_run is not None:
            wandb_run.finish()

    return agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OGPO training template for Robomimic low-dim tasks.")
    parser.add_argument("--policy-checkpoint", default="outputs/flow_policy_lift.pt")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--dataset", default="datasets/robomimic/lift/ph/low_dim_v15.hdf5")
    parser.add_argument("--output", default="outputs/ogpo.pt")
    parser.add_argument("--total-env-steps", type=int, default=2_000_000)
    parser.add_argument("--start-transitions", type=int, default=10_000)
    parser.add_argument("--replay-size", type=int, default=2_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--updates-per-chunk", type=int, default=1)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--execute-horizon", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ppo-lr", type=float, default=4.5e-5)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--actor-weight-decay", type=float, default=0.0)
    parser.add_argument("--critic-weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1000.0)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--critic-hidden-size", type=int, default=None)
    parser.add_argument("--num-qs", type=int, default=10)
    parser.add_argument("--num-min", type=int, default=2)
    parser.add_argument("--q-agg", choices=["mean", "min", "subsample"], default="mean")
    parser.add_argument("--actor-samples", type=int, default=32)
    parser.add_argument("--best-of-n", type=int, default=8)
    parser.add_argument("--flow-noise-std", type=float, default=0.05)
    parser.add_argument("--no-debias-flow-noise", dest="debias_flow_noise", action="store_false")
    parser.add_argument("--deterministic-rollout", dest="stochastic_rollout", action="store_false")
    parser.add_argument("--ppo-clip", type=float, default=0.01)
    parser.add_argument("--actor-update-epochs", type=int, default=2)
    parser.add_argument("--adv-strategy", choices=["vanilla", "conservative"], default="conservative")
    parser.add_argument("--normalize-group", action="store_true")
    parser.add_argument("--success-buffer-size", type=int, default=200_000)
    parser.add_argument("--no-success-buffer", dest="use_success_buffer", action="store_false")
    parser.add_argument("--bc-coeff", type=float, default=1.0)
    parser.add_argument("--no-bc-regularization", dest="use_bc_regularization", action="store_false")
    parser.add_argument("--actor-scheduler", choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--actor-warmup-steps", type=int, default=2000)
    parser.add_argument("--actor-decay-steps", type=int, default=50000)
    parser.add_argument("--actor-end-value", type=float, default=2e-5)
    parser.add_argument("--critic-scheduler", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--critic-warmup-steps", type=int, default=500)
    parser.add_argument("--critic-decay-steps", type=int, default=5000)
    parser.add_argument("--critic-end-value", type=float, default=1e-8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=10_000)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--eval-horizon", type=int, default=400)
    parser.add_argument("--eval-execute-horizon", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="robomimic-flow-policy")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--ignore-done", action="store_true")
    parser.add_argument("--no-clip-actions", dest="clip_actions", action="store_false")
    parser.set_defaults(
        clip_actions=True,
        debias_flow_noise=True,
        stochastic_rollout=True,
        use_success_buffer=True,
        use_bc_regularization=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

# python3 -m architectures.flow_policy.ogpo
