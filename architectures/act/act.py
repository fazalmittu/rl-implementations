import torch
import torch.nn as nn

class ACT(nn.Module):

    def __init__(
        self, 
        num_cams: int = 1,
        cam_width: int = 480,
        cam_height: int = 640,
        state_dim: int = 6,
        hidden_size: int = 256,
        latent_dim: int = 32,
        chunk_size: int = 100,
        beta: float = 10.0,
        batch_size: int = 4,
    ):
        super.__init__()

        self.num_cams = num_cams
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.chunk_size = chunk_size
        self.beta = beta

        # need to handle the actual processing of the images (through ResNet) in a separate file/class

        # need to instantiate the observation encoder (just a few transformer blocks)

        # need to instantiate the VAE (will house the encoder and decoder)
        
    