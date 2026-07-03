import argparse
import copy
from collections import deque
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from architectures.flow_policy.critic import REDQCritic
from architectures.flow_policy.eval_rollout import (
    is_success,
    make_env,
    obs_to_tensor,
    unnormalize_action,
)
from architectures.flow_policy.flow_policy import FlowPolicy
from architectures.flow_policy.train import get_device, to_cpu
from common.replay_buffer import ReplayBuffer


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
        gamma: float = 0.99,
        tau: float = 0.05,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        grad_clip: float | None = 10.0,
        batch_size: int = 256,
        sample_steps: int = 10,
        discount_steps: int | None = None,
        actor_samples: int = 4,
        flow_noise_std: float = 0.1,
        ppo_clip: float = 0.2,
        actor_update_epochs: int = 2,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.flat_action_dim = action_horizon * action_dim
        self.num_min = num_min
        self.gamma = gamma
        self.discount = gamma ** (discount_steps or action_horizon)
        self.tau = tau
        self.grad_clip = grad_clip
        self.batch_size = batch_size
        self.sample_steps = sample_steps
        self.actor_samples = actor_samples
        self.flow_noise_std = flow_noise_std
        self.ppo_clip = ppo_clip
        self.actor_update_epochs = actor_update_epochs

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
            hidden_size=hidden_size,
        )
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.critic_opt = torch.optim.AdamW(
            self.critic.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor.sample(
            obs,
            num_steps=self.sample_steps,
            return_chain=False,
        )

    def update(self, buffer: ReplayBuffer) -> dict[str, float]:
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
            target_q = self.critic_target.min(next_obs, next_action, self.num_min)
            target_q = reward + (1.0 - done) * self.discount * target_q

        current_q = self.critic(obs, action)
        critic_loss = F.mse_loss(current_q, target_q.expand_as(current_q))

        self.critic_opt.zero_grad()
        critic_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_opt.step()
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
            ) # sample actor_samples chains per obs and freeze their old log probs (bc torch.no_grad())

        # sampled_actions: (B * actor_samples, action_horizon, action_dim)
        # chain:           (B * actor_samples, sample_steps + 1, action_horizon, action_dim)
        # old_log_probs:   (B * actor_samples, sample_steps)
        sampled_actions = sampled_actions.flatten(start_dim=1)
        old_log_prob = old_log_probs.sum(dim=1)     # sum log probs across each action chain
        # sampled_actions: (B * actor_samples, action_horizon * action_dim)
        # old_log_prob:    (B * actor_samples,)

        with torch.no_grad():
            q = self.critic.min(obs_rep, sampled_actions, self.num_min)    # we repeated obs bc same obs used for each actor_sample we do
            q = q.view(self.batch_size, self.actor_samples)   # (B * actor_samples) -> (B, actor_samples)
            advantage = q - q.mean(dim=1, keepdim=True)       # each value of q gets subtracted by the mean of all q's to get A for each action
            advantage = advantage / (q.std(dim=1, keepdim=True, unbiased=False) + 1e-6) 
            advantage = advantage.flatten()
            # q:         (B, actor_samples)
            # advantage: (B * actor_samples,)

        # here's the actual online update in which we do self.actor_update_epochs per update()
        # for each higher level update() we take output chains from our flow model (B * actor_sample times)
        # for each actor_update, we want to see how the new actor's log probs compare to the higher level chains 
        # as more epochs happen we want to push actor to give higher advantage actions more likelihood of happening
        actor_loss = torch.tensor(0.0, device=obs.device)
        for _ in range(self.actor_update_epochs):
            _, log_probs = self.actor.sample(
                obs_rep,
                num_steps=self.sample_steps,
                return_chain=False,
                noise_std=self.flow_noise_std,
                return_log_probs=True,
                chain=chain,
            ) 
            log_prob = log_probs.sum(dim=1)

            ratio = torch.exp(log_prob - old_log_prob)
            clipped_ratio = ratio.clamp(1.0 - self.ppo_clip, 1.0 + self.ppo_clip)
            actor_loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.actor_opt.step()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "q": current_q.mean().item(),
            "target_q": target_q.mean().item(),
            "reward": reward.mean().item(),
            "actor_q": q.mean().item(),
            "advantage_abs": advantage.abs().mean().item(),
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
        gamma=args.gamma,
        tau=args.tau,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        batch_size=args.batch_size,
        sample_steps=args.sample_steps,
        discount_steps=args.execute_horizon,
        actor_samples=args.actor_samples,
        flow_noise_std=args.flow_noise_std,
        ppo_clip=args.ppo_clip,
        actor_update_epochs=args.actor_update_epochs,
    ).to(device)
    agent.actor.load_state_dict(checkpoint["model_state_dict"])
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
    action_norm = agent.act(obs_tensor)[0]
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

    while len(buffer) < args.start_transitions:
        transition = collect_chunk(
            env, obs, episode_steps, agent, obs_keys, checkpoint, args, device
        )
        add_to_buffer(buffer, transition)
        episode_steps += transition["steps"]
        episode_return += transition["raw_reward"]

        if transition["done"]:
            episodes += 1
            successes.append(transition["success"])
            returns.append(episode_return)
            obs = env.reset()
            episode_steps = 0
            episode_return = 0.0
        else:
            obs = transition["next_env_obs"]

    return {
        "episodes": float(episodes),
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

    return wandb.init(
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
    env = make_env(args.dataset, record_video=False)
    wandb_run = init_wandb(args, checkpoint, agent, device)

    print(
        f"OGPO: start_transitions={args.start_transitions}"
        f" | total_env_steps={args.total_env_steps}"
        f" | action_horizon={agent.action_horizon}"
        f" | execute_horizon={args.execute_horizon}"
        f" | device={device}",
        flush=True,
    )

    next_log = args.log_every
    next_save = args.save_every
    recent_successes = deque(maxlen=20)
    start_t = time.perf_counter()

    try:
        seed_stats = seed_replay(env, buffer, agent, obs_keys, checkpoint, args, device)
        print(
            f"seeded replay: transitions={len(buffer)}"
            f" | episodes={int(seed_stats['episodes'])}"
            f" | success_rate={seed_stats['success_rate']:.3f}"
            f" | mean_return={seed_stats['mean_return']:.3f}",
            flush=True,
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "seed/episodes": seed_stats["episodes"],
                    "seed/success_rate": seed_stats["success_rate"],
                    "seed/mean_return": seed_stats["mean_return"],
                    "replay/size": len(buffer),
                },
                step=0,
            )

        obs = env.reset()
        episode_steps = 0
        env_steps = 0
        chunks = 0
        last_metrics = {}

        while env_steps < args.total_env_steps:
            transition = collect_chunk(
                env, obs, episode_steps, agent, obs_keys, checkpoint, args, device
            )
            add_to_buffer(buffer, transition)
            chunks += 1
            env_steps += transition["steps"]
            episode_steps += transition["steps"]

            for _ in range(args.updates_per_chunk):
                if len(buffer) >= args.batch_size:
                    last_metrics = agent.update(buffer)

            if transition["done"]:
                recent_successes.append(transition["success"])
                obs = env.reset()
                episode_steps = 0
            else:
                obs = transition["next_env_obs"]

            if env_steps >= next_log:
                recent_success = float(np.mean(recent_successes)) if recent_successes else 0.0
                metrics = " ".join(
                    f"| {key} {value:.4f}" for key, value in last_metrics.items()
                )
                if metrics:
                    metrics = f" {metrics}"
                print(
                    f"env_steps {env_steps}"
                    f" | chunks {chunks}"
                    f" | buffer {len(buffer)}"
                    f" | recent_success {recent_success:.3f}"
                    f"{metrics}",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_log = {
                        "train/env_steps": env_steps,
                        "train/chunks": chunks,
                        "train/elapsed_s": time.perf_counter() - start_t,
                        "rollout/recent_success": recent_success,
                        "rollout/chunk_reward": transition["raw_reward"],
                        "rollout/chunk_steps": transition["steps"],
                        "replay/size": len(buffer),
                    }
                    wandb_log.update({f"update/{key}": value for key, value in last_metrics.items()})
                    wandb_run.log(wandb_log, step=env_steps)
                while env_steps >= next_log:
                    next_log += args.log_every

            if args.save_every is not None and env_steps >= next_save:
                save_path = Path(args.output)
                save_path = save_path.with_name(
                    f"{save_path.stem}_env_{env_steps}{save_path.suffix}"
                )
                save_checkpoint(save_path, agent, args, env_steps, chunks, len(buffer))
                print(f"saved checkpoint to {save_path}", flush=True)
                while env_steps >= next_save:
                    next_save += args.save_every

        save_checkpoint(args.output, agent, args, env_steps, chunks, len(buffer))
        print(f"saved final checkpoint to {args.output}", flush=True)
    finally:
        env.close()
        if wandb_run is not None:
            wandb_run.finish()

    return agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OGPO training template for Robomimic low-dim tasks.")
    parser.add_argument("--policy-checkpoint", default="outputs/flow_policy_lift.pt")
    parser.add_argument("--dataset", default="datasets/robomimic/lift/ph/low_dim_v15.hdf5")
    parser.add_argument("--output", default="outputs/ogpo.pt")
    parser.add_argument("--total-env-steps", type=int, default=10_000)
    parser.add_argument("--start-transitions", type=int, default=100)
    parser.add_argument("--replay-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--updates-per-chunk", type=int, default=1)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--execute-horizon", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-qs", type=int, default=10)
    parser.add_argument("--num-min", type=int, default=2)
    parser.add_argument("--actor-samples", type=int, default=4)
    parser.add_argument("--flow-noise-std", type=float, default=0.1)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--actor-update-epochs", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=10_000)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="robomimic-flow-policy")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--ignore-done", action="store_true")
    parser.add_argument("--no-clip-actions", dest="clip_actions", action="store_false")
    parser.set_defaults(clip_actions=True)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

# python3 -m architectures.flow_policy.ogpo
