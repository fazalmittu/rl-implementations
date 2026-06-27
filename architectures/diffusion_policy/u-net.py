import torch
import torch.nn as nn

class UNet(nn.Module):

    def __init__(self, action_dim: int = 6, hidden_size: int = 128):
        super().__init__()

        self.action_dim = action_dim
        self.hidden_size = hidden_size

        self.action_proj = nn.Linear(self.action_dim, self.hidden_size)

        # encoder levels
        self.encoder_l1 = nn.Sequential(
            nn.Conv1d(self.hidden_size, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, 1, 1),
        )
        
        self.max_pool_l1 = nn.MaxPool1d(2)

        self.encoder_l2 = nn.Sequential(
            nn.Conv1d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 3, 1, 1),
        )
        
        self.max_pool_l2 = nn.MaxPool1d(2)

        self.encoder_l3 = nn.Sequential(
            nn.Conv1d(512, 1024, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 3, 1, 1),
        )
        
        self.max_pool_l3 = nn.MaxPool1d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv1d(1024, 1024, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 3, 1, 1),
        )

        self.conv_transpose_l3 = nn.ConvTranspose1d(1024, 512, 4, 2, 1)
        
        self.decoder_l3 = nn.Sequential(
            nn.Conv1d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 3, 1, 1),
        )

        self.conv_transpose_l2 = nn.ConvTranspose1d(512, 256, 4, 2, 1)

        self.decoder_l2 = nn.Sequential(
            nn.Conv1d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, 1, 1),
        )

        self.conv_transpose_l1 = nn.ConvTranspose1d(256, 128, 4, 2, 1)

        self.decoder_l1 = nn.Sequential(
            nn.Conv1d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, 1, 1),
        )



    def forward(self, actions: torch.Tensor, obs_emb: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        actions: (B, action_pred_horizon, action_dim)
        obs_emb: (B, hidden_state)
        t: (B, hidden_state)
        """

        action_embs = self.action_proj(actions)
        action_embs = action_embs.permute(0, 2, 1)   # torch.conv1d expects (N, C, L)
        # print(action_embs.shape)

        encoder_l1_out = self.encoder_l1(action_embs)
        # print(encoder_l1_out.shape)

        encoder_l2_in = self.max_pool_l1(encoder_l1_out)
        # print(encoder_l2_in.shape)

        encoder_l2_out = self.encoder_l2(encoder_l2_in)
        # print(encoder_l2_out.shape)

        encoder_l3_in = self.max_pool_l2(encoder_l2_out)
        # print(encoder_l3_in.shape)

        encoder_l3_out = self.encoder_l3(encoder_l3_in)
        # print(encoder_l2_out.shape)

        bottleneck_in = self.max_pool_l3(encoder_l3_out)
        # print(bottleneck_in.shape)

        bottleneck_out = self.bottleneck(bottleneck_in)
        # print(bottleneck_out.shape)

        decoder_l3_in = self.conv_transpose_l3(bottleneck_out)
        # print(decoder_l3_in.shape)

        decoder_l3_out = self.decoder_l3(decoder_l3_in) + encoder_l3_out
        # print(decoder_l3_out.shape)
        
        decoder_l2_in = self.conv_transpose_l2(decoder_l3_out)
        # print(decoder_l2_in.shape)

        decoder_l2_out = self.decoder_l2(decoder_l2_in) + encoder_l2_out
        # print(decoder_l2_out.shape)

        decoder_l1_in = self.conv_transpose_l1(decoder_l2_out)
        # print(decoder_l1_in.shape)

        decoder_l1_out = self.decoder_l1(decoder_l1_in) + encoder_l1_out
        # print(decoder_l1_out.shape)


if __name__ == "__main__":

    unet = UNet()

    unet(torch.randn((4, 16, 6)), torch.randn((4, 512)), torch.randn((4, 512)))

# python3 -m architectures.diffusion_policy.u-net