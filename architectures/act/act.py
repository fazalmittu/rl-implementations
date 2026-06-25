from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from architectures.act.img_processor import CameraProcessor
from architectures.act.obs_encoder import ObsEncoder
from architectures.act.proprio_state_processor import StateProcessor
from architectures.act.vae import VAE

from architectures.act.data_loader import make_dataloader

class ACT(nn.Module):

    def __init__(
        self, 
        num_cams: int = 1,
        cam_width: int = 640,
        cam_height: int = 480,
        state_dim: int = 6,
        hidden_size: int = 512,
        latent_dim: int = 32,
        chunk_size: int = 100,
        beta: float = 10.0,
        batch_size: int = 4,
        nhead: int = 8,
        dim_feedforward: int = 3200,
        dropout: float = 0.1,
        obs_encoder_layers: int = 4,
        vae_encoder_layers: int = 4,
        vae_decoder_layers: int = 1,
    ):
        super().__init__()

        self.num_cams = num_cams
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.chunk_size = chunk_size
        self.beta = beta
        self.batch_size = batch_size
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.obs_encoder_layers = obs_encoder_layers
        self.vae_encoder_layers = vae_encoder_layers
        self.vae_decoder_layers = vae_decoder_layers

        self.camera_processor = CameraProcessor(hidden_size=self.hidden_size)
        self.state_processor = StateProcessor(hidden_size=self.hidden_size, state_dim=self.state_dim)
        self.obs_encoder = ObsEncoder(
            hidden_size=self.hidden_size,
            num_layers=self.obs_encoder_layers,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )
        self.vae = VAE(
            hidden_size=self.hidden_size,
            state_dim=self.state_dim,
            encoder_layers=self.vae_encoder_layers,
            decoder_layers=self.vae_decoder_layers,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            latent_dim=self.latent_dim,
            action_chunk_size=self.chunk_size,
            batch_size=self.batch_size
        )

    def loss(
        self, 
        actions: torch.Tensor, 
        action_preds: torch.Tensor, 
        mean: torch.Tensor, 
        logvar: torch.Tensor
    ):
        l1_loss = F.l1_loss(action_preds, actions, reduction='mean')
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        total_loss = l1_loss + self.beta * kl_loss

        return total_loss

    def forward(
        self, 
        images: torch.Tensor,   # (B, num_cams, C, H, W)
        state: torch.Tensor,    # (B, state_dim)
        actions: torch.Tensor,  # (B, action_chunk_size, state_dim)
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        h_obs = self.encode_obs(images, state)

        action_preds, mean, logvar = self.vae(actions, h_obs)

        return action_preds, mean, logvar

    def encode_obs(self, images: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        img_tokens = self.camera_processor(images)
        state_token = self.state_processor(state)
        return self.obs_encoder(img_tokens, state_token)

    @torch.no_grad()
    def predict_action_chunk(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict a full action chunk from observation only.

        During inference ACT uses the mean of the latent prior, z=0, instead
        of encoding a ground-truth action chunk.
        """
        was_training = self.training
        self.eval()

        try:
            device = next(self.parameters()).device
            images = images.to(device)
            state = state.to(device)

            h_obs = self.encode_obs(images, state)

            if z is None:
                z = torch.zeros(
                    state.shape[0],
                    self.latent_dim,
                    device=device,
                    dtype=state.dtype,
                )
            else:
                z = z.to(device=device, dtype=state.dtype)

            return self.vae.decode(z, h_obs)
        finally:
            if was_training:
                self.train()

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
    model: ACT | None = None,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    grad_clip: float | None = None,
    batch_size: int = 4,
    chunk_size: int = 100,
    device: str = "cpu",
    max_steps_per_epoch: int | None = None,
    total_steps: int | None = None,
    save_path: str | None = None,
):

    if model is None:
        model = ACT(
            num_cams=1,
            cam_width=640,
            cam_height=480,
            state_dim=6,
            hidden_size=512,
            latent_dim=32,
            chunk_size=chunk_size,
            beta=10.0,
            batch_size=batch_size,
        )

    loader = make_dataloader(batch_size=batch_size, chunk_size=chunk_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = model.loss
    model = model.to(device)

    if total_steps is not None:
        steps_per_epoch = max_steps_per_epoch if max_steps_per_epoch is not None else len(loader)
        epochs = (total_steps + steps_per_epoch - 1) // steps_per_epoch

    global_step = 0

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0
        num_steps = 0

        for step, (images, state, actions) in enumerate(loader):
            images = images.to(device)
            state = state.to(device)
            actions = actions.to(device)

            optimizer.zero_grad()

            action_preds, mean, logvar = model(images, state, actions)

            loss = loss_fn(actions, action_preds, mean, logvar)

            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            running_loss += loss.item()
            num_steps += 1
            global_step += 1

            if total_steps is not None and global_step >= total_steps:
                break

            if max_steps_per_epoch is not None and step + 1 >= max_steps_per_epoch:
                break

        print(f"Epoch {epoch+1}/{epochs} - Step {global_step} - Loss: {running_loss / num_steps}")

        if total_steps is not None and global_step >= total_steps:
            break

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": to_cpu(model.state_dict()),
                "optimizer_state_dict": to_cpu(optimizer.state_dict()),
                "model_config": {
                    "num_cams": model.num_cams,
                    "cam_width": model.cam_width,
                    "cam_height": model.cam_height,
                    "state_dim": model.state_dim,
                    "hidden_size": model.hidden_size,
                    "latent_dim": model.latent_dim,
                    "chunk_size": model.chunk_size,
                    "beta": model.beta,
                    "batch_size": model.batch_size,
                    "nhead": model.nhead,
                    "dim_feedforward": model.dim_feedforward,
                    "dropout": model.dropout,
                    "obs_encoder_layers": model.obs_encoder_layers,
                    "vae_encoder_layers": model.vae_encoder_layers,
                    "vae_decoder_layers": model.vae_decoder_layers,
                },
                "train_config": {
                    "epochs": epochs,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "grad_clip": grad_clip,
                    "batch_size": batch_size,
                    "chunk_size": chunk_size,
                    "device": device,
                    "max_steps_per_epoch": max_steps_per_epoch,
                    "total_steps": total_steps,
                    "global_step": global_step,
                },
            },
            save_path,
        )
        print(f"Saved model to {save_path}")

    return model

if __name__ == "__main__":

    torch.manual_seed(1000)

    batch_size = 8
    chunk_size = 100

    model = ACT(
        num_cams=1,
        cam_width=640,
        cam_height=480,
        state_dim=6,
        hidden_size=512,
        latent_dim=32,
        chunk_size=chunk_size,
        beta=10.0,
        batch_size=batch_size,
        nhead=8,
        dim_feedforward=3200,
        dropout=0.1,
        obs_encoder_layers=4,
        vae_encoder_layers=4,
        vae_decoder_layers=1,
    )

    train(
        model=model,
        total_steps=20_000,
        lr=1e-5,
        weight_decay=1e-4,
        grad_clip=10.0,
        batch_size=batch_size,
        chunk_size=chunk_size,
        device=get_device(),
        save_path="outputs/act_so101_pickup_env_30_clean.pt",
    )
    

# python3 -m architectures.act.act

        
    
