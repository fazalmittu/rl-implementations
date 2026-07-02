from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


DEFAULT_LIFT_OBS_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "object",
)


def _demo_sort_key(name: str) -> tuple[int, str]:
    if name.startswith("demo_"):
        suffix = name.split("_", 1)[1]
        if suffix.isdigit():
            return int(suffix), name
    return 10**12, name


class RobomimicLowdimDataset(Dataset):
    """Robomimic HDF5 dataset for low-dimensional action-chunk BC.

    Expected layout:
        data/demo_*/actions
        data/demo_*/obs/<obs_key>
    """

    def __init__(
        self,
        hdf5_path: str | Path,
        obs_keys: list[str] | tuple[str, ...] | None = None,
        action_horizon: int = 4,
        normalize_obs: bool = True,
        normalize_action: bool = True,
        pad_action_chunks: bool = True,
        demo_limit: int | None = None,
    ):
        self.hdf5_path = Path(hdf5_path)
        self.action_horizon = action_horizon
        self.normalize_obs = normalize_obs
        self.normalize_action = normalize_action
        self.pad_action_chunks = pad_action_chunks

        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"dataset not found: {self.hdf5_path}")

        self.obs_keys = self._resolve_obs_keys(obs_keys)
        self.episodes = self._load_episodes(demo_limit)
        self.starts = self._make_starts()
        self.stats = self._compute_stats()

        if not self.starts:
            raise ValueError(
                f"no valid action chunks found in {self.hdf5_path}; "
                f"action_horizon={self.action_horizon}"
            )

        self.obs_dim = int(self.episodes[0]["obs"].shape[-1])
        self.action_dim = int(self.episodes[0]["actions"].shape[-1])
        self.obs_key_shapes = self._load_obs_key_shapes()

    def _demo_names(self, h5: h5py.File) -> list[str]:
        if "data" not in h5:
            raise KeyError(f"{self.hdf5_path} does not contain a top-level 'data' group")
        return sorted(h5["data"].keys(), key=_demo_sort_key)

    def _resolve_obs_keys(self, obs_keys: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
        with h5py.File(self.hdf5_path, "r") as h5:
            demo_names = self._demo_names(h5)
            if not demo_names:
                raise ValueError(f"no demos found under {self.hdf5_path}/data")
            obs_group = h5["data"][demo_names[0]]["obs"]

            if obs_keys:
                missing = [key for key in obs_keys if key not in obs_group]
                if missing:
                    available = ", ".join(sorted(obs_group.keys()))
                    raise KeyError(f"missing obs keys {missing}; available keys: {available}")
                return tuple(obs_keys)

            default_keys = tuple(key for key in DEFAULT_LIFT_OBS_KEYS if key in obs_group)
            if default_keys:
                return default_keys

            lowdim_keys = []
            for key in sorted(obs_group.keys()):
                value = obs_group[key]
                if value.ndim == 2 and np.issubdtype(value.dtype, np.number):
                    lowdim_keys.append(key)
            if not lowdim_keys:
                available = ", ".join(sorted(obs_group.keys()))
                raise ValueError(f"could not infer low-dim obs keys; available keys: {available}")
            return tuple(lowdim_keys)

    def _load_episodes(self, demo_limit: int | None) -> list[dict[str, np.ndarray]]:
        episodes = []
        with h5py.File(self.hdf5_path, "r") as h5:
            demo_names = self._demo_names(h5)
            if demo_limit is not None:
                demo_names = demo_names[:demo_limit]

            for demo_name in demo_names:
                demo = h5["data"][demo_name]
                actions = np.asarray(demo["actions"], dtype=np.float32)
                obs_parts = [
                    np.asarray(demo["obs"][key], dtype=np.float32).reshape(actions.shape[0], -1)
                    for key in self.obs_keys
                ]
                obs = np.concatenate(obs_parts, axis=-1).astype(np.float32)

                if len(actions) != len(obs):
                    raise ValueError(
                        f"{demo_name} has mismatched obs/actions lengths: "
                        f"{len(obs)} vs {len(actions)}"
                    )

                episodes.append({"name": demo_name, "obs": obs, "actions": actions})

        if not episodes:
            raise ValueError(f"no episodes loaded from {self.hdf5_path}")
        return episodes

    def _load_obs_key_shapes(self) -> dict[str, tuple[int, ...]]:
        shapes = {}
        with h5py.File(self.hdf5_path, "r") as h5:
            demo_name = self._demo_names(h5)[0]
            for key in self.obs_keys:
                shapes[key] = tuple(h5["data"][demo_name]["obs"][key].shape[1:])
        return shapes

    def _make_starts(self) -> list[tuple[int, int]]:
        starts = []
        for episode_idx, episode in enumerate(self.episodes):
            episode_len = len(episode["actions"])
            if self.pad_action_chunks:
                valid_starts = episode_len
            else:
                valid_starts = max(episode_len - self.action_horizon + 1, 0)

            for start in range(valid_starts):
                starts.append((episode_idx, start))
        return starts

    def _compute_stats(self) -> dict[str, dict[str, np.ndarray]]:
        obs = np.concatenate([episode["obs"] for episode in self.episodes], axis=0)
        actions = np.concatenate([episode["actions"] for episode in self.episodes], axis=0)
        return {
            "obs": {
                "mean": obs.mean(axis=0).astype(np.float32),
                "std": np.maximum(obs.std(axis=0), 1e-6).astype(np.float32),
            },
            "action": {
                "mean": actions.mean(axis=0).astype(np.float32),
                "std": np.maximum(actions.std(axis=0), 1e-6).astype(np.float32),
            },
        }

    def _normalize(self, x: np.ndarray, key: str) -> np.ndarray:
        return (x - self.stats[key]["mean"]) / self.stats[key]["std"]

    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.stats["action"]["mean"], device=action.device, dtype=action.dtype)
        std = torch.tensor(self.stats["action"]["std"], device=action.device, dtype=action.dtype)
        return action * std + mean

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        episode_idx, start = self.starts[index]
        episode = self.episodes[episode_idx]

        obs = episode["obs"][start]
        actions = episode["actions"][start : start + self.action_horizon]

        if len(actions) < self.action_horizon:
            pad = np.repeat(actions[-1:], self.action_horizon - len(actions), axis=0)
            actions = np.concatenate([actions, pad], axis=0)

        if self.normalize_obs:
            obs = self._normalize(obs, "obs")
        if self.normalize_action:
            actions = self._normalize(actions, "action")

        return torch.from_numpy(obs.astype(np.float32)), torch.from_numpy(actions.astype(np.float32))

    def __len__(self) -> int:
        return len(self.starts)


def make_dataloader(
    hdf5_path: str | Path,
    obs_keys: list[str] | tuple[str, ...] | None = None,
    action_horizon: int = 4,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    pad_action_chunks: bool = True,
    demo_limit: int | None = None,
) -> DataLoader:
    dataset = RobomimicLowdimDataset(
        hdf5_path=hdf5_path,
        obs_keys=obs_keys,
        action_horizon=action_horizon,
        pad_action_chunks=pad_action_chunks,
        demo_limit=demo_limit,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
