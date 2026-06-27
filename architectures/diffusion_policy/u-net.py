import torch
import torch.nn as nn

class ConditionalSequential(nn.Module):

    def __init__(self, *args, cond_dim: int, cond_channels: int):
        super().__init__()

        self.net = nn.Sequential(*args)
        self.cond_proj = nn.Linear(cond_dim, cond_channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i == 0:
                x = x + self.cond_proj(cond).unsqueeze(-1)

        return x


class UNet(nn.Module):

    def __init__(self, action_dim: int = 6, hidden_size: int = 128):
        super().__init__()

        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.cond_dim = self.hidden_size * 2

        self.action_proj = nn.Linear(self.action_dim, self.hidden_size)

        # encoder levels
        self.encoder_l1 = ConditionalSequential(
            nn.Conv1d(self.hidden_size, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, 1, 1),
            cond_dim=self.cond_dim,
            cond_channels=256
        )
        
        self.max_pool_l1 = nn.MaxPool1d(2)

        self.encoder_l2 = ConditionalSequential(
            nn.Conv1d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 3, 1, 1),
            cond_dim=self.cond_dim,
            cond_channels=512
        )
        
        self.max_pool_l2 = nn.MaxPool1d(2)

        self.encoder_l3 = ConditionalSequential(
            nn.Conv1d(512, 1024, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 3, 1, 1),
            cond_dim=self.cond_dim,
            cond_channels=1024
        )
        
        self.max_pool_l3 = nn.MaxPool1d(2)

        self.bottleneck = ConditionalSequential(
            nn.Conv1d(1024, 2048, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(2048, 2048, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(2048, 2048, 3, 1, 1),
            cond_dim=self.cond_dim,
            cond_channels=2048
        )

        self.conv_transpose_l3 = nn.ConvTranspose1d(2048, 1024, 4, 2, 1)
        
        self.decoder_l3 = ConditionalSequential(
            nn.Conv1d(2048, 1024, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 3, 1, 1),
            cond_dim=self.cond_dim,
            cond_channels=1024
        )

        self.conv_transpose_l2 = nn.ConvTranspose1d(1024, 512, 4, 2, 1)

        self.decoder_l2 = ConditionalSequential(
            nn.Conv1d(1024, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 3, 1, 1),
            cond_dim=self.cond_dim,
            cond_channels=512
        )

        self.conv_transpose_l1 = nn.ConvTranspose1d(512, 256, 4, 2, 1)

        self.decoder_l1 = ConditionalSequential(
            nn.Conv1d(512, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, 1, 1),
            nn.Conv1d(256, 128, 1),
            cond_dim=self.cond_dim,
            cond_channels=256
        )

        self.out_proj = nn.Conv1d(128, self.action_dim, 1)


    def forward(self, actions: torch.Tensor, obs_emb: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        actions: (B, action_pred_horizon, action_dim)
        obs_emb: (B, hidden_state)
        t: (B, hidden_state)
        """

        if obs_emb.dim() == 3:
            obs_emb = obs_emb.mean(dim=1)

        cond = torch.cat((obs_emb, t), dim=1)

        action_embs = self.action_proj(actions)
        action_embs = action_embs.permute(0, 2, 1)   # torch.conv1d expects (N, C, L)

        encoder_l1_out = self.encoder_l1(action_embs, cond)
        # print(encoder_l1_out.shape)

        encoder_l2_in = self.max_pool_l1(encoder_l1_out)
        # print(encoder_l2_in.shape)

        encoder_l2_out = self.encoder_l2(encoder_l2_in, cond)
        # print(encoder_l2_out.shape)

        encoder_l3_in = self.max_pool_l2(encoder_l2_out)
        # print(encoder_l3_in.shape)

        encoder_l3_out = self.encoder_l3(encoder_l3_in, cond)
        # print(encoder_l3_out.shape)

        bottleneck_in = self.max_pool_l3(encoder_l3_out)
        # print(bottleneck_in.shape)

        bottleneck_out = self.bottleneck(bottleneck_in, cond)
        # print(bottleneck_out.shape)

        decoder_l3_in = self.conv_transpose_l3(bottleneck_out)
        # print(decoder_l3_in.shape)

        decoder_l3_out = self.decoder_l3(torch.cat((encoder_l3_out, decoder_l3_in), dim=1), cond)
        # print(decoder_l3_out.shape)
        
        decoder_l2_in = self.conv_transpose_l2(decoder_l3_out)
        # print(decoder_l2_in.shape)

        decoder_l2_out = self.decoder_l2(torch.cat((encoder_l2_out, decoder_l2_in), dim=1), cond) 
        # print(decoder_l2_out.shape)

        decoder_l1_in = self.conv_transpose_l1(decoder_l2_out)
        # print(decoder_l1_in.shape)

        decoder_l1_out = self.decoder_l1(torch.cat((encoder_l1_out, decoder_l1_in), dim=1), cond)
        # print(decoder_l1_out.shape)

        out = self.out_proj(decoder_l1_out)
        out = out.permute(0, 2, 1)

        return out


if __name__ == "__main__":

    unet = UNet()

    print(unet(torch.randn((4, 16, 6)), torch.randn((4, 128)), torch.randn((4, 128))).shape)

# python3 -m architectures.diffusion_policy.u-net
