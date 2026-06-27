import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ObsEncoder(nn.Module):
    """ 
    takes in observations and turns into embeddings 

    images -> ResNet
    pro-prio state -> nn.Linear projection
    
    """

    def __init__(self, hidden_size: int = 128, state_dim: int = 6):
        super().__init__()

        self.hidden_size = hidden_size
        self.state_dim = state_dim

        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet_feature_extractor = nn.Sequential(*list(self.resnet.children())[:-2])

        self.state_proj = nn.Linear(self.state_dim, self.hidden_size)

    def forward(self, images: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        images: (B, obs_horizon, num_images, C, H, W)
        state: (B, obs_horizon, state_dim)
        """

        B, obs_horizon, _, _, _, _ = images.shape

        img_embeddings = []

        for b in range(B):
            batch_imgs = []
            for o in range(obs_horizon):
                features = self.resnet_feature_extractor(images[b][o])
                features = features.reshape(-1, self.hidden_size)
                batch_imgs.append(features)
            
            batch_imgs = torch.cat(batch_imgs, dim=0)
            img_embeddings.append(batch_imgs)
        
        img_embeddings = torch.stack(img_embeddings, dim=0)

        state_embeddings = []
        for b in range(B):
            features = self.state_proj(state[b])
            state_embeddings.append(features)
        
        state_embeddings = torch.stack(state_embeddings, dim=0)

        final = torch.cat((img_embeddings, state_embeddings), dim=1)

        return final  # (B, num_tokens, hidden_size)


if __name__ == "__main__":

    encoder = ObsEncoder()

    encoder(torch.randn((4, 2, 3, 3, 640, 480)), torch.randn((4, 2, 6)))

# python3 -m architectures.diffusion_policy.obs_encoder
