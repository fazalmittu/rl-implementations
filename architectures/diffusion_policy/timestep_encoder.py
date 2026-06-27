import torch
import torch.nn as nn

from architectures.utils.pos_encodings import ddpm_timestep_embedding

class TimeStepEncoder(nn.Module):

    def __init__(self, hidden_size: int = 128):
        super().__init__()

        self.hidden_size = hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.SiLU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size)
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        timesteps: (B) -- will always be values in (0, diffusion_timesteps)
        """

        sin_embs = ddpm_timestep_embedding(timesteps, self.hidden_size)
        sin_embs = self.mlp(sin_embs)

        return sin_embs

        # print(sin_embs.shape)

if __name__ == "__main__":

    enc = TimeStepEncoder()

    enc(torch.randn((4)))

# python3 -m architectures.diffusion_policy.timestep_encoder