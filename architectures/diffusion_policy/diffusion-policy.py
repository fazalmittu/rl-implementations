from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from architectures.act.data_loader import make_dataloader
from architectures.diffusion_policy.obs_encoder import ObsEncoder
from architectures.diffusion_policy.timestep_encoder import TimeStepEncoder
from architectures.diffusion_policy.unet import UNet

class DiffusionPolicy(nn.Module):

    def __init__(
        self, 
        hidden_size: int = 128, 
        state_dim: int = 6,
        action_dim: int = 6,
        action_pred_horizon: int = 16,
        obs_horizon: int = 1,
        action_horizon: int = 8,
        batch_size: int = 4,
        diffusion_timesteps: int = 10,
        beta_start: float = 1e-4,
        beta_end: float = 0.02
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_pred_horizon = action_pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.batch_size = batch_size
        self.diffusion_timesteps = diffusion_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.obs_encoder = ObsEncoder(hidden_size=self.hidden_size, state_dim=self.state_dim)
        self.timestep_encoder = TimeStepEncoder(hidden_size=self.hidden_size)

        self.unet = UNet(action_dim=self.action_dim, hidden_size=self.hidden_size)

        betas = torch.linspace(self.beta_start, self.beta_end, self.diffusion_timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

    def loss(self, noise_pred: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        noise_pred: (B, action_pred_horizon, action_dim)
        noise:      (B, action_pred_horizon, action_dim)
        """

        return F.mse_loss(noise_pred, noise, reduction="mean")

    def add_noise(self, actions: torch.Tensor, noise: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_bars = self.sqrt_alpha_bars[time].view(-1, 1, 1)
        sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars[time].view(-1, 1, 1)

        return sqrt_alpha_bars * actions + sqrt_one_minus_alpha_bars * noise

    def forward(self, images: torch.Tensor, state: torch.Tensor, time: torch.Tensor, a_noisy: torch.Tensor) -> torch.Tensor:
        """
        images:  (B, num_cams, C, H, W) or (B, obs_horizon, num_images, C, H, W)
        state:   (B, state_dim) or (B, obs_horizon, state_dim)
        time:    (B)
        a_noisy: (B, action_pred_horizon, action_dim)
        """

        if images.dim() == 5:
            images = images.unsqueeze(1)
        if state.dim() == 2:
            state = state.unsqueeze(1)

        obs_emb = self.obs_encoder(images, state)
        time_embs = self.timestep_encoder(time)

        noise_pred = self.unet(a_noisy, obs_emb, time_embs)

        return noise_pred


def get_device() -> str:
    if torch.backends.mps.is_available():
        print("Using: mps")
        return "mps"
    if torch.cuda.is_available():
        print("Using: cuda")
        return "cuda"
    print("Using: cpu")
    return "cpu"

def to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_cpu(v) for v in obj)
    return obj

def train(
    model: DiffusionPolicy | None = None,
    epochs: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    grad_clip: float | None = None,
    batch_size: int = 4,
    chunk_size: int = 16,
    device: str = "cpu",
    max_steps_per_epoch: int | None = None,
    total_steps: int | None = None,
    save_path: str | None = None,
    log_every: int = 10,
    save_every: int | None = None,
    num_workers: int = 0,
):

    if model is None:
        model = DiffusionPolicy(
            hidden_size=128,
            state_dim=6,
            action_dim=6,
            action_pred_horizon=chunk_size,
            obs_horizon=1,
            action_horizon=8,
            batch_size=batch_size,
            diffusion_timesteps=10,
        )

    if model.action_pred_horizon % 8 != 0:
        raise ValueError("action_pred_horizon must be divisible by 8 for the current U-Net")

    loader = make_dataloader(
        batch_size=batch_size,
        chunk_size=model.action_pred_horizon,
        num_workers=num_workers,
    )
    print(
        f"Training data: {len(loader.dataset)} samples | {len(loader)} steps/epoch | batch_size={batch_size}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = model.loss
    model = model.to(device)

    if total_steps is not None:
        steps_per_epoch = max_steps_per_epoch if max_steps_per_epoch is not None else len(loader)
        epochs = (total_steps + steps_per_epoch - 1) // steps_per_epoch

    global_step = 0
    train_start_t = time.perf_counter()
    last_log_t = train_start_t

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0
        num_steps = 0

        for step, (images, state, actions) in enumerate(loader):
            images = images.to(device)
            state = state.to(device)
            actions = actions.to(device)

            optimizer.zero_grad()

            timesteps = torch.randint(
                0,
                model.diffusion_timesteps,
                (actions.shape[0],),
                device=actions.device,
            )
            noise = torch.randn_like(actions)
            a_noisy = model.add_noise(actions, noise, timesteps)
            noise_pred = model(images, state, timesteps, a_noisy)

            loss = loss_fn(noise_pred, noise)

            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            running_loss += loss.item()
            num_steps += 1
            global_step += 1

            if log_every is not None and (global_step == 1 or global_step % log_every == 0):
                now = time.perf_counter()
                dt = now - last_log_t
                steps_per_s = log_every / dt if global_step != 1 and dt > 0 else 0.0
                avg_loss = running_loss / num_steps
                print(
                    f"Step {global_step}"
                    f" | epoch {epoch+1}/{epochs}"
                    f" | loss {loss.item():.4f}"
                    f" | avg_loss {avg_loss:.4f}"
                    f" | {steps_per_s:.2f} steps/s",
                    flush=True,
                )
                last_log_t = now

            if save_path is not None and save_every is not None and global_step % save_every == 0:
                checkpoint_path = Path(save_path)
                checkpoint_path = checkpoint_path.with_name(
                    f"{checkpoint_path.stem}_step_{global_step}{checkpoint_path.suffix}"
                )
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    epochs,
                    lr,
                    weight_decay,
                    grad_clip,
                    batch_size,
                    device,
                    max_steps_per_epoch,
                    total_steps,
                    global_step,
                    num_workers,
                )
                print(f"Saved checkpoint to {checkpoint_path}", flush=True)

            if total_steps is not None and global_step >= total_steps:
                break

            if max_steps_per_epoch is not None and step + 1 >= max_steps_per_epoch:
                break

        print(f"Epoch {epoch+1}/{epochs} - Step {global_step} - Loss: {running_loss / num_steps}", flush=True)

        if total_steps is not None and global_step >= total_steps:
            break

    if save_path is not None:
        save_checkpoint(
            Path(save_path),
            model,
            optimizer,
            epochs,
            lr,
            weight_decay,
            grad_clip,
            batch_size,
            device,
            max_steps_per_epoch,
            total_steps,
            global_step,
            num_workers,
        )
        print(f"Saved model to {save_path}", flush=True)

    return model


def save_checkpoint(
    save_path: Path,
    model: DiffusionPolicy,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float | None,
    batch_size: int,
    device: str,
    max_steps_per_epoch: int | None,
    total_steps: int | None,
    global_step: int,
    num_workers: int = 0,
):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": to_cpu(model.state_dict()),
            "optimizer_state_dict": to_cpu(optimizer.state_dict()),
            "model_config": {
                "hidden_size": model.hidden_size,
                "state_dim": model.state_dim,
                "action_dim": model.action_dim,
                "action_pred_horizon": model.action_pred_horizon,
                "obs_horizon": model.obs_horizon,
                "action_horizon": model.action_horizon,
                "batch_size": model.batch_size,
                "diffusion_timesteps": model.diffusion_timesteps,
                "beta_start": model.beta_start,
                "beta_end": model.beta_end,
            },
            "train_config": {
                "epochs": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "grad_clip": grad_clip,
                "batch_size": batch_size,
                "chunk_size": model.action_pred_horizon,
                "device": device,
                "max_steps_per_epoch": max_steps_per_epoch,
                "total_steps": total_steps,
                "global_step": global_step,
                "num_workers": num_workers,
            },
        },
        save_path,
    )


if __name__ == "__main__":

    torch.manual_seed(1000)

    batch_size = 8
    chunk_size = 16

    model = DiffusionPolicy(
        hidden_size=512,
        state_dim=6,
        action_dim=6,
        action_pred_horizon=chunk_size,
        obs_horizon=1,
        action_horizon=8,
        batch_size=batch_size,
        diffusion_timesteps=100,
    )

    train(
        model=model,
        total_steps=20_000,
        lr=1e-4,
        weight_decay=1e-6,
        grad_clip=10.0,
        batch_size=batch_size,
        chunk_size=chunk_size,
        device=get_device(),
        save_path="outputs/diffusion_policy_so101_pickup_env_30_clean.pt",
        log_every=10,
        save_every=5000,
        num_workers=2,
    )

# python3 architectures/diffusion_policy/diffusion-policy.py
