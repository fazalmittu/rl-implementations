import json
from bisect import bisect_right
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


DATASET_ROOT = "/Users/fazal/Desktop/lerobot-test/datasets/so101_pickup_env_30_clean"


class LeRobotACTDataset(Dataset):
    def __init__(
        self,
        root: str = DATASET_ROOT,
        camera_key: str = "observation.images.environment",
        chunk_size: int = 100,
        image_size: tuple[int, int] = (480, 640),
        normalize_state: bool = True,
        normalize_action: bool = True,
        imagenet_norm: bool = True,
        episodes: list[int] | None = None,
    ):
        self.root = Path(root)
        self.camera_key = camera_key
        self.chunk_size = chunk_size
        self.image_size = image_size
        self.normalize_state = normalize_state
        self.normalize_action = normalize_action
        self.imagenet_norm = imagenet_norm

        self.data = self._load_data(episodes)
        self.stats = self._load_stats()
        self.starts = self._make_starts()

        self.video_paths = sorted((self.root / "videos" / self.camera_key).glob("chunk-*/file-*.mp4"))
        if len(self.video_paths) == 0:
            raise FileNotFoundError(f"no videos found for camera_key={self.camera_key}")

        self.video_frame_counts = self._get_video_frame_counts()
        self.video_offsets = np.cumsum(self.video_frame_counts).tolist()
        self.captures = {}

        self.img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def _load_data(self, episodes):
        parquet_paths = sorted((self.root / "data").glob("chunk-*/file-*.parquet"))
        if len(parquet_paths) == 0:
            raise FileNotFoundError(f"no parquet files found under {self.root / 'data'}")

        dfs = [pd.read_parquet(path) for path in parquet_paths]
        data = pd.concat(dfs, ignore_index=True).sort_values("index").reset_index(drop=True)

        if episodes is not None:
            data = data[data["episode_index"].isin(episodes)].reset_index(drop=True)

        return data

    def _load_stats(self):
        with open(self.root / "meta" / "stats.json", "r") as f:
            return json.load(f)

    def _stats_tensor(self, key: str, stat: str) -> torch.Tensor:
        return torch.tensor(self.stats[key][stat], dtype=torch.float32)

    def _normalize(self, x: torch.Tensor, key: str) -> torch.Tensor:
        mean = self._stats_tensor(key, "mean")
        std = self._stats_tensor(key, "std").clamp_min(1e-6)
        return (x - mean) / std

    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        mean = self._stats_tensor("action", "mean").to(action.device)
        std = self._stats_tensor("action", "std").clamp_min(1e-6).to(action.device)
        return action * std + mean

    def _make_starts(self):
        starts = []
        for _, ep_data in self.data.groupby("episode_index", sort=True):
            idxs = ep_data.index.to_numpy()
            if len(idxs) < self.chunk_size:
                continue

            for i in range(0, len(idxs) - self.chunk_size + 1):
                starts.append(int(idxs[i]))

        return starts

    def _get_video_frame_counts(self):
        counts = []
        for path in self.video_paths:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"could not open video {path}")
            counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            cap.release()
        return counts

    def _capture(self, video_i: int):
        if video_i not in self.captures:
            cap = cv2.VideoCapture(str(self.video_paths[video_i]))
            if not cap.isOpened():
                raise RuntimeError(f"could not open video {self.video_paths[video_i]}")
            self.captures[video_i] = cap

        return self.captures[video_i]

    def _read_image(self, global_frame_i: int) -> torch.Tensor:
        video_i = bisect_right(self.video_offsets, global_frame_i)
        prev_offset = 0 if video_i == 0 else self.video_offsets[video_i - 1]
        local_frame_i = global_frame_i - prev_offset

        cap = self._capture(video_i)
        cap.set(cv2.CAP_PROP_POS_FRAMES, local_frame_i)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"could not read frame {global_frame_i}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w = self.image_size
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))

        img = torch.from_numpy(frame).float() / 255.0
        img = img.permute(2, 0, 1).contiguous()

        if self.imagenet_norm:
            img = (img - self.img_mean) / self.img_std

        return img.unsqueeze(0)

    def __getitem__(self, i):
        start = self.starts[i]
        row = self.data.iloc[start]
        action_rows = self.data.iloc[start : start + self.chunk_size]

        global_frame_i = int(row["index"])
        image = self._read_image(global_frame_i)

        state = torch.tensor(row["observation.state"], dtype=torch.float32)
        actions = torch.tensor(np.stack(action_rows["action"].to_numpy()), dtype=torch.float32)

        if self.normalize_state:
            state = self._normalize(state, "observation.state")
        if self.normalize_action:
            actions = self._normalize(actions, "action")

        return image, state, actions

    def __len__(self):
        return len(self.starts)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["captures"] = {}
        return state

    def close(self):
        for cap in self.captures.values():
            cap.release()
        self.captures = {}


def make_dataloader(
    root: str = DATASET_ROOT,
    batch_size: int = 4,
    chunk_size: int = 100,
    shuffle: bool = True,
    num_workers: int = 0,
):
    dataset = LeRobotACTDataset(root=root, chunk_size=chunk_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )


if __name__ == "__main__":
    loader = make_dataloader(batch_size=4, chunk_size=100, shuffle=True)
    images, state, actions = next(iter(loader))

    print(images.shape)
    print(state.shape)
    print(actions.shape)

# python3 -m architectures.act.data_loader
