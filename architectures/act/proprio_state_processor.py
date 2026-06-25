import torch
import torch.nn as nn

class StateProcessor(nn.Module):

    def __init__(self, hidden_size: int = 256, state_dim: int = 6):
        super().__init__()

        self.hidden_size = hidden_size
        self.state_dim = state_dim

        self.projection = nn.Linear(state_dim, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x: (B, state_dim) """
        return self.projection(x).unsqueeze(1)

if __name__ == "__main__":
    processor = StateProcessor(256, 6)

    print(processor(torch.zeros((4, 6))).shape)

# python3 -m architectures.act.proprio_state_processor