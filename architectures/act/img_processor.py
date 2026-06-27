import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from architectures.utils.pos_encodings import positionalencoding2d

class CameraProcessor(nn.Module):
    
    def __init__(self, hidden_size: int = 512):
        super().__init__()

        self.hidden_size = hidden_size

        self.RESNET_HIDDEN_SIZE = 512  # num channels of layer of ResNet we use

        # need to get hidden representations, we go 2 layers deep backwards since 1 layer deep reduces spatial dims to 1x1
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet_feature_extractor = nn.Sequential(*list(self.resnet.children())[:-2])

        self.channel_reduction = nn.Conv2d(self.RESNET_HIDDEN_SIZE, self.hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x: (B, num_cams, C, H, W) """
        B, num_cams, C, H, W = x.shape

        x = x.reshape(B * num_cams, C, H, W)
        feat = self.resnet_feature_extractor(x)
        feat_down = self.channel_reduction(feat)
        _, _, h, w = feat_down.shape

        tokens = feat_down.permute(0, 2, 3, 1).reshape(B, num_cams * h * w, self.hidden_size)

        pos_enc = positionalencoding2d(self.hidden_size, h, w)
        pos_enc = pos_enc.to(device=x.device, dtype=x.dtype)
        pos_enc = pos_enc.permute(1, 2, 0).reshape(-1, self.hidden_size)
        pos_enc = pos_enc.repeat(num_cams, 1).unsqueeze(0)

        return tokens + pos_enc

if __name__ == "__main__":
    processor = CameraProcessor(hidden_size=512)

    print(processor(torch.zeros((4, 3, 3, 640, 480))).shape)

# python3 -m architectures.act.img_processor
