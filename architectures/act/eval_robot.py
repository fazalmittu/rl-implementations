import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from architectures.act.act import ACT, get_device
from architectures.act.data_loader import DATASET_ROOT

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower


DEFAULT_CHECKPOINT = "outputs/act_so101_pickup_env_30_clean.pt"
DEFAULT_PORT = "/dev/cu.usbmodem5C4C1257251"
CAMERA_KEY = "observation.images.environment"


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def stats_tensor(stats, key: str, stat: str, device: str):
    return torch.tensor(stats[key][stat], dtype=torch.float32, device=device)


def normalize_state(state: torch.Tensor, stats, device: str) -> torch.Tensor:
    mean = stats_tensor(stats, "observation.state", "mean", device)
    std = stats_tensor(stats, "observation.state", "std", device).clamp_min(1e-6)
    return (state - mean) / std


def unnormalize_action(action: torch.Tensor, stats, device: str) -> torch.Tensor:
    mean = stats_tensor(stats, "action", "mean", device)
    std = stats_tensor(stats, "action", "std", device).clamp_min(1e-6)
    return action * std + mean


def preprocess_image(frame, image_size: tuple[int, int], device: str) -> torch.Tensor:
    h, w = image_size
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if frame.shape[:2] != (h, w):
        frame = cv2.resize(frame, (w, h))

    image = torch.from_numpy(frame).float().to(device) / 255.0
    image = image.permute(2, 0, 1).contiguous()

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(3, 1, 1)
    image = (image - mean) / std

    return image.unsqueeze(0).unsqueeze(0)


def make_state(obs: dict, action_names: list[str], device: str) -> torch.Tensor:
    values = [obs[name] for name in action_names]
    return torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0)


def make_action_dict(action: torch.Tensor, action_names: list[str]) -> dict[str, float]:
    return {name: float(value) for name, value in zip(action_names, action.cpu().tolist())}


def load_model(checkpoint_path: str, device: str) -> ACT:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]

    model = ACT(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def connect_camera(index: int, width: int, height: int, fps: int):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera index {index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def run_episode(args):
    device = args.device or get_device()
    dataset_root = Path(args.dataset_root)
    info = load_json(dataset_root / "meta" / "info.json")
    stats = load_json(dataset_root / "meta" / "stats.json")

    action_names = info["features"]["action"]["names"]
    image_shape = info["features"][CAMERA_KEY]["shape"]
    image_size = (image_shape[0], image_shape[1])
    height, width = image_size

    model = load_model(args.checkpoint, device)

    robot_cfg = SOFollowerRobotConfig(
        id=args.robot_id,
        port=args.port,
        max_relative_target=args.max_relative_target,
        disable_torque_on_disconnect=False,
        cameras={},
    )
    robot = SOFollower(robot_cfg)
    cap = None

    if not args.yes and not args.dry_run:
        input("Clear the robot workspace, place the object/target, then press ENTER to start.")

    try:
        robot.connect(calibrate=False)
        cap = connect_camera(args.camera_index, width, height, args.fps)

        next_tick = time.perf_counter()
        end_t = time.perf_counter() + args.episode_time_s
        actions_sent = 0

        while time.perf_counter() < end_t:
            obs = robot.get_observation()
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("could not read camera frame")

            state = make_state(obs, action_names, device)
            state_norm = normalize_state(state, stats, device)
            image = preprocess_image(frame, image_size, device)

            action_chunk_norm = model.predict_action_chunk(image, state_norm)
            action_chunk = unnormalize_action(action_chunk_norm.squeeze(0), stats, device)

            if args.print_actions:
                print(
                    "chunk",
                    {
                        "min": action_chunk.min(dim=0).values.detach().cpu().numpy().round(2).tolist(),
                        "max": action_chunk.max(dim=0).values.detach().cpu().numpy().round(2).tolist(),
                    },
                )

            for action in action_chunk[: args.n_action_steps]:
                if time.perf_counter() >= end_t:
                    break

                action_dict = make_action_dict(action, action_names)
                if args.dry_run:
                    print(action_dict)
                else:
                    robot.send_action(action_dict)

                actions_sent += 1
                next_tick += 1.0 / args.fps
                sleep_s = next_tick - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)

        print(f"Finished episode. actions_sent={actions_sent}")

    finally:
        if cap is not None:
            cap.release()

        if robot.is_connected:
            try:
                robot.bus.disable_torque(num_retry=20)
                print("torque_off_ok")
            except Exception as exc:
                print(f"torque_off_failed: {exc}")
            robot.disconnect()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", default=DATASET_ROOT)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--robot-id", default="so101_follower_1")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episode-time-s", type=float, default=10.0)
    parser.add_argument("--n-action-steps", type=int, default=100)
    parser.add_argument("--max-relative-target", type=float, default=5.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-actions", action="store_true")
    parser.add_argument("--yes", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_episode(parse_args())

# python3 -m architectures.act.eval_robot --dry-run
