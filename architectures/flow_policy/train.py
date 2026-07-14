import argparse
from itertools import cycle
from pathlib import Path
import time

import numpy as np
import torch

from architectures.flow_policy.flow_policy import FlowPolicy
from architectures.flow_policy.data_loader import make_dataloader


def get_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        print("Using: cuda")
        return "cuda"
    if torch.backends.mps.is_available():
        print("Using: mps")
        return "mps"
    print("Using: cpu")
    return "cpu"


def to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj).cpu()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_cpu(v) for v in obj)
    return obj


def save_checkpoint(
    save_path: Path,
    model: FlowPolicy,
    optimizer: torch.optim.Optimizer,
    train_config: dict,
    dataset,
    global_step: int,
):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": to_cpu(model.state_dict()),
            "optimizer_state_dict": to_cpu(optimizer.state_dict()),
            "model_config": {
                "obs_dim": model.obs_dim,
                "action_dim": model.action_dim,
                "action_horizon": model.action_horizon,
                "hidden_size": model.hidden_size,
                "num_layers": model.num_layers,
                "time_embed_dim": model.time_embed_dim,
            },
            "train_config": {**train_config, "global_step": global_step},
            "dataset": {
                "obs_keys": dataset.obs_keys,
                "obs_key_shapes": dataset.obs_key_shapes,
                "stats": to_cpu(dataset.stats),
            },
        },
        save_path,
    )


def eval_checkpoint(model: FlowPolicy, dataset) -> dict:
    return {
        "model_config": {
            "action_horizon": model.action_horizon,
        },
        "dataset": {
            "obs_keys": dataset.obs_keys,
            "stats": to_cpu(dataset.stats),
        },
    }


def init_wandb(args: argparse.Namespace, dataset, model: FlowPolicy, device: str):
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
            "obs_dim": dataset.obs_dim,
            "action_dim": dataset.action_dim,
            "obs_keys": list(dataset.obs_keys),
            "model_config": {
                "obs_dim": model.obs_dim,
                "action_dim": model.action_dim,
                "action_horizon": model.action_horizon,
                "hidden_size": model.hidden_size,
                "num_layers": model.num_layers,
                "time_embed_dim": model.time_embed_dim,
            },
        },
    )
    wandb.define_metric("agent_steps")
    wandb.define_metric("train/*", step_metric="agent_steps")
    wandb.define_metric("eval/*", step_metric="agent_steps")
    return run


def run_eval(args: argparse.Namespace, model: FlowPolicy, dataset, device: str, global_step: int):
    from architectures.flow_policy.eval_rollout import eval_action_fn

    was_training = model.training
    model.eval()
    try:
        metrics = eval_action_fn(
            lambda obs: model.sample(
                obs,
                num_steps=args.eval_sample_steps,
                return_chain=False,
            )[0],
            eval_checkpoint(model, dataset),
            args.dataset,
            device,
            episodes=args.eval_episodes,
            horizon=args.eval_horizon,
            execute_horizon=args.eval_execute_horizon,
            clip_actions=args.clip_actions,
            ignore_done=args.ignore_done,
            seed=args.eval_seed,
        )
    finally:
        if was_training:
            model.train()

    agent_steps = global_step * args.batch_size
    print(
        f"eval step {global_step}"
        f" | agent_steps {agent_steps}"
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
    obs_keys = tuple(args.obs_keys) if args.obs_keys else None

    loader = make_dataloader(
        hdf5_path=args.dataset,
        obs_keys=obs_keys,
        action_horizon=args.action_horizon,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pad_action_chunks=not args.no_pad_action_chunks,
        demo_limit=args.demo_limit,
    )
    dataset = loader.dataset

    print(
        "Training data:"
        f" {len(dataset)} windows | {len(dataset.episodes)} demos |"
        f" obs_dim={dataset.obs_dim} | action_dim={dataset.action_dim} |"
        f" obs_keys={list(dataset.obs_keys)}",
        flush=True,
    )

    model = FlowPolicy(
        obs_dim=dataset.obs_dim,
        action_dim=dataset.action_dim,
        action_horizon=args.action_horizon,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        time_embed_dim=args.time_embed_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint.get("train_config", {}).get("global_step", 0))
        print(f"resumed checkpoint {args.resume} at step {start_step}", flush=True)

    wandb_run = init_wandb(args, dataset, model, device)

    train_config = vars(args).copy()
    global_step = start_step
    running_loss = 0.0
    loss_count = 0
    start_t = time.perf_counter()
    last_log_t = start_t

    model.train()
    for obs, actions in cycle(loader):
        obs = obs.to(device)
        actions = actions.to(device)

        optimizer.zero_grad()
        loss = model.flow_matching_loss(obs, actions)
        loss.backward()

        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        global_step += 1
        running_loss += loss.item()
        loss_count += 1

        if global_step == 1 or global_step % args.log_every == 0:
            now = time.perf_counter()
            log_steps = 1 if global_step == 1 else args.log_every
            steps_per_s = log_steps / max(now - last_log_t, 1e-9)
            avg_loss = running_loss / max(loss_count, 1)
            print(
                f"step {global_step}"
                f" | agent_steps {global_step * args.batch_size}"
                f" | loss {loss.item():.5f}"
                f" | avg_loss {avg_loss:.5f}"
                f" | {steps_per_s:.2f} steps/s",
                flush=True,
            )
            if wandb_run is not None:
                agent_steps = global_step * args.batch_size
                wandb_run.log(
                    {
                        "agent_steps": agent_steps,
                        "train/global_step": global_step,
                        "train/loss": loss.item(),
                        "train/avg_loss": avg_loss,
                        "train/steps_per_s": steps_per_s,
                    },
                    step=agent_steps,
                )
            last_log_t = now

        if args.eval_every is not None and global_step % args.eval_every == 0:
            metrics = run_eval(args, model, dataset, device, global_step)
            if wandb_run is not None:
                agent_steps = global_step * args.batch_size
                wandb_run.log(
                    {
                        "agent_steps": agent_steps,
                        "eval/success_rate": metrics["success_rate"],
                        "eval/mean_return": metrics["mean_return"],
                        "eval/mean_length": metrics["mean_length"],
                    },
                    step=agent_steps,
                )
            if (
                args.target_success_rate is not None
                and metrics["success_rate"] >= args.target_success_rate
            ):
                print(
                    f"target success_rate {args.target_success_rate:.3f} reached"
                    f" at step {global_step}",
                    flush=True,
                )
                break

        if args.save_every is not None and global_step % args.save_every == 0:
            checkpoint_path = Path(args.output)
            checkpoint_path = checkpoint_path.with_name(
                f"{checkpoint_path.stem}_step_{global_step}{checkpoint_path.suffix}"
            )
            save_checkpoint(checkpoint_path, model, optimizer, train_config, dataset, global_step)
            print(f"saved checkpoint to {checkpoint_path}", flush=True)

        if global_step >= args.total_steps:
            break

    save_checkpoint(Path(args.output), model, optimizer, train_config, dataset, global_step)
    elapsed = time.perf_counter() - start_t
    print(f"saved final checkpoint to {args.output}", flush=True)
    print(f"finished {global_step} steps in {elapsed:.1f}s", flush=True)
    if wandb_run is not None:
        wandb_run.finish()

    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small flow policy on Robomimic low-dim data.")
    parser.add_argument("--dataset", required=True, help="Path to a Robomimic .hdf5 dataset.")
    parser.add_argument(
        "--obs-keys",
        nargs="*",
        default=None,
        help="Low-dim obs keys. Defaults to Robomimic Lift keys when present.",
    )
    parser.add_argument("--output", default="outputs/flow_policy_lift.pt")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--action-horizon", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--time-embed-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--total-steps", type=int, default=20_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--eval-horizon", type=int, default=400)
    parser.add_argument("--eval-execute-horizon", type=int, default=None)
    parser.add_argument("--eval-sample-steps", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--target-success-rate", type=float, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="robomimic-flow-policy")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--ignore-done", action="store_true")
    parser.add_argument("--no-clip-actions", dest="clip_actions", action="store_false")
    parser.add_argument("--demo-limit", type=int, default=None)
    parser.add_argument("--no-pad-action-chunks", action="store_true")
    parser.set_defaults(clip_actions=True)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
