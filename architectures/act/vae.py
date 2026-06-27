import torch
import torch.nn as nn
from architectures.utils.pos_encodings import positionalencoding1d

class VAE(nn.Module):

    def __init__(
        self, 
        hidden_size: int = 512, 
        state_dim: int = 6, 
        encoder_layers: int = 4,
        decoder_layers: int = 1,
        nhead: int = 8,
        dim_feedforward: int = 3200,
        dropout: float = 0.1,
        latent_dim: int = 32,
        action_chunk_size: int = 100,
        batch_size: int = 4,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.state_dim = state_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.nhead = nhead
        self.latent_dim = latent_dim
        self.action_chunk_size = action_chunk_size
        self.batch_size = batch_size

        self.cls_token = nn.Parameter(torch.randn((1, 1, self.hidden_size)))

        self.action_projection = nn.Linear(self.state_dim, self.hidden_size)
        self.latent_projection = nn.Linear(self.latent_dim, self.hidden_size)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.encoder_layers)
        self.ffn_mean = nn.Linear(self.hidden_size, self.latent_dim)
        self.ffn_logvar = nn.Linear(self.hidden_size, self.latent_dim)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_size,
            nhead=self.nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.decoder_layers)
        self.ffn_actions = nn.Linear(self.hidden_size, self.state_dim)

        self.query_embs = nn.Embedding(num_embeddings=self.action_chunk_size, embedding_dim=self.hidden_size)

    def encode(self, actions: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """ actions: (B, action_chunk_size, state_dim)"""
        B, action_chunk_size, _ = actions.shape
        action_embs = self.action_projection(actions)
        pos_enc = positionalencoding1d(self.hidden_size, action_chunk_size)
        pos_enc = pos_enc.to(device=actions.device, dtype=actions.dtype)

        action_embs = action_embs + pos_enc

        encoder_in = torch.cat((self.cls_token.expand(B, -1, -1), action_embs), dim=1)
        encoder_out = self.encoder(encoder_in)

        cls_out = encoder_out[:, 0, :]

        mean = self.ffn_mean(cls_out)
        logvar = self.ffn_logvar(cls_out)

        return mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z: torch.Tensor, h_obs: torch.Tensor) -> torch.Tensor:
        """ z: latent vector, h_obs: obs_encoder output """
        B = z.shape[0]
        latent_emb = self.latent_projection(z)   # (B, hidden_size)
        latent_emb = latent_emb.unsqueeze(1)     # (B, 1, hidden_size)

        query_emb_tens = self.query_embs.weight  # (action_chunk_size, hidden_size)
        query_emb_tens = query_emb_tens.unsqueeze(0).expand(B, -1, -1)

        decoder_tgt = torch.cat((latent_emb, query_emb_tens), dim=1)
        decoder_out = self.decoder(decoder_tgt, h_obs)
        query_outs = decoder_out[:, 1:, :]

        actions = self.ffn_actions(query_outs)

        return actions

    def forward(self, actions: torch.Tensor, h_obs: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        mean, logvar = self.encode(actions)
        z = self.reparameterize(mean, logvar)
        actions = self.decode(z, h_obs)

        return actions, mean, logvar 
        

if __name__ == "__main__":
    vae = VAE(hidden_size=512, state_dim=6, nhead=8, latent_dim=32, action_chunk_size=100)

    # vae.encode(torch.randn((4, 100, 6)))

    out = vae.forward(torch.zeros((4, 100, 6)), torch.zeros(4, 901, 512))
    
    shapes = [t.shape for t in out]
    print(shapes)

# python3 -m architectures.act.vae


