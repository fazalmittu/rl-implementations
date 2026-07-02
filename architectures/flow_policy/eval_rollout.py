import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from architectures.flow_policy.flow_policy import FlowPolicy
from architectures.flow_policy.train import get_device


LIVE_OBS_KEY_MAP = {
    "object": "object-state",
}


def load_env_args(dataset_path: str | Path) -> dict:
    with h5py.File(dataset_path, "r") as h5:
        return json.loads(h5["data"].attrs["env_args"])


def make_env(dataset_path: str | Path, record_video: bool = False):
    import robosuite as suite

    env_args = load_env_args(dataset_path)
    env_kwargs = dict(env_args["env_kwargs"])
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = record_video
    env_kwargs["use_camera_obs"] = False
    return suite.make(env_args["env_name"], **env_kwargs)


def video_prefix(dataset_path: str | Path) -> str:
    env_name = load_env_args(dataset_path)["env_name"]
    chars = [ch.lower() if ch.isalnum() else "_" for ch in env_name]
    prefix = "".join(chars).strip("_")
    return prefix or "rollout"


def checkpoint_stats_tensor(checkpoint: dict, key: str, stat: str, device: str) -> torch.Tensor:
    value = checkpoint["dataset"]["stats"][key][stat]
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.float32)
    return torch.tensor(value, device=device, dtype=torch.float32)


def obs_to_tensor(obs: dict, obs_keys: tuple[str, ...], checkpoint: dict, device: str) -> torch.Tensor:
    parts = []
    for key in obs_keys:
        live_key = LIVE_OBS_KEY_MAP.get(key, key)
        if live_key not in obs:
            available = ", ".join(sorted(obs.keys()))
            raise KeyError(f"missing live obs key '{live_key}' for checkpoint key '{key}'; available: {available}")
        parts.append(np.asarray(obs[live_key], dtype=np.float32).reshape(-1))

    obs_vec = torch.tensor(np.concatenate(parts), device=device, dtype=torch.float32).unsqueeze(0)
    mean = checkpoint_stats_tensor(checkpoint, "obs", "mean", device)
    std = checkpoint_stats_tensor(checkpoint, "obs", "std", device).clamp_min(1e-6)
    return (obs_vec - mean) / std


def unnormalize_action(action: torch.Tensor, checkpoint: dict) -> torch.Tensor:
    mean = checkpoint["dataset"]["stats"]["action"]["mean"]
    std = checkpoint["dataset"]["stats"]["action"]["std"]
    if not torch.is_tensor(mean):
        mean = torch.tensor(mean, dtype=action.dtype, device=action.device)
    else:
        mean = mean.to(device=action.device, dtype=action.dtype)
    if not torch.is_tensor(std):
        std = torch.tensor(std, dtype=action.dtype, device=action.device)
    else:
        std = std.to(device=action.device, dtype=action.dtype)
    return action * std + mean


def is_success(env) -> bool:
    if hasattr(env, "_check_success"):
        return bool(env._check_success())
    return False


def render_frame(env, args: argparse.Namespace) -> np.ndarray:
    frame = env.sim.render(
        camera_name=args.camera_name,
        width=args.video_width,
        height=args.video_height,
    )
    if not args.no_flip_video:
        frame = np.flipud(frame)
    return frame


@torch.inference_mode()
def rollout(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = FlowPolicy(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    obs_keys = tuple(checkpoint["dataset"]["obs_keys"])
    execute_horizon = args.execute_horizon or model.action_horizon
    if execute_horizon <= 0 or execute_horizon > model.action_horizon:
        raise ValueError(f"execute_horizon must be in [1, {model.action_horizon}]")

    record_video = args.video_dir is not None
    env = make_env(args.dataset, record_video=record_video)
    prefix = video_prefix(args.dataset)
    video_dir = Path(args.video_dir) if args.video_dir is not None else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Rollout eval: episodes={args.episodes} | horizon={args.horizon} | "
        f"action_horizon={model.action_horizon} | execute_horizon={execute_horizon} | "
        f"sample_steps={args.sample_steps} | device={device}",
        flush=True,
    )

    successes = []
    lengths = []
    returns = []
    start_t = time.perf_counter()

    try:
        for episode in range(args.episodes):
            obs = env.reset()
            total_reward = 0.0
            success = False
            steps = 0
            video_writer = None
            video_path = None

            should_record = video_dir is not None and episode % args.record_every == 0
            if should_record:
                import imageio.v2 as imageio

                video_path = video_dir / f"{prefix}_episode_{episode + 1:03d}.mp4"
                video_writer = imageio.get_writer(str(video_path), fps=args.video_fps)
                video_writer.append_data(render_frame(env, args))

            while steps < args.horizon:
                obs_tensor = obs_to_tensor(obs, obs_keys, checkpoint, device)
                action_chunk_norm = model.sample(
                    obs_tensor,
                    num_steps=args.sample_steps,
                    return_chain=False,
                )[0]
                action_chunk = unnormalize_action(action_chunk_norm, checkpoint).detach().cpu().numpy()

                if args.clip_actions:
                    action_chunk = np.clip(action_chunk, -1.0, 1.0)

                for action in action_chunk[:execute_horizon]:
                    obs, reward, done, _ = env.step(action)
                    total_reward += float(reward)
                    steps += 1
                    success = is_success(env)
                    if video_writer is not None:
                        video_writer.append_data(render_frame(env, args))

                    if success or (done and not args.ignore_done) or steps >= args.horizon:
                        break

                if success or (done and not args.ignore_done):
                    break

            if video_writer is not None:
                video_writer.close()

            successes.append(float(success))
            lengths.append(steps)
            returns.append(total_reward)
            print(
                f"episode {episode + 1}/{args.episodes}"
                f" | success={int(success)}"
                f" | steps={steps}"
                f" | return={total_reward:.3f}"
                f"{f' | video={video_path}' if video_path is not None else ''}",
                flush=True,
            )
    finally:
        env.close()

    elapsed = time.perf_counter() - start_t
    print("\nRollout metrics")
    print(f"  success_rate: {np.mean(successes):.3f}")
    print(f"  mean_length:  {np.mean(lengths):.1f}")
    print(f"  mean_return:  {np.mean(returns):.3f}")
    print(f"  elapsed:      {elapsed:.1f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robosuite rollout eval for a low-dim flow policy.")
    parser.add_argument("--checkpoint", default="outputs/flow_policy_lift.pt")
    parser.add_argument("--dataset", default="datasets/robomimic/lift/ph/low_dim_v15.hdf5")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--execute-horizon", type=int, default=None)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--ignore-done", action="store_true")
    parser.add_argument("--no-clip-actions", dest="clip_actions", action="store_false")
    parser.add_argument("--video-dir", default=None, help="Directory to write rollout mp4 videos.")
    parser.add_argument("--record-every", type=int, default=1, help="Record every Nth episode when --video-dir is set.")
    parser.add_argument("--camera-name", default="agentview")
    parser.add_argument("--video-width", type=int, default=320)
    parser.add_argument("--video-height", type=int, default=240)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument(
        "--no-flip-video",
        action="store_true",
        help="Disable vertical flipping of MuJoCo offscreen frames.",
    )
    parser.set_defaults(clip_actions=True)
    return parser.parse_args()


if __name__ == "__main__":
    rollout(parse_args())
