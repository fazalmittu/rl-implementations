import torch
import torch.nn as nn

class ObsEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

    def forward(self, img_tokens: torch.Tensor, state_token: torch.Tensor) -> torch.Tensor:
        out = self.encoder(torch.cat((img_tokens, state_token), dim=1))
        print(out.shape)
        return out

if __name__ == "__main__":
    obs_encoder = ObsEncoder()

    print(obs_encoder(torch.zeros((4, 900, 256)), torch.zeros((4, 1, 256))))

# python3 -m architectures.act.obs_encoder