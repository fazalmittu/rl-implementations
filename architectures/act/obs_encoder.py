import torch
import torch.nn as nn

class ObsEncoder(nn.Module):

    def __init__(
        self,
        hidden_size: int = 512,
        num_layers: int = 4,
        nhead: int = 8,
        dim_feedforward: int = 3200,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nhead = nhead

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)

    def forward(self, img_tokens: torch.Tensor, state_token: torch.Tensor) -> torch.Tensor:
        out = self.encoder(torch.cat((img_tokens, state_token), dim=1))
        # print(out.shape)
        return out

if __name__ == "__main__":
    obs_encoder = ObsEncoder()

    print(obs_encoder(torch.zeros((4, 900, 512)), torch.zeros((4, 1, 512))))

# python3 -m architectures.act.obs_encoder
