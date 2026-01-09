import torch.nn as nn

class Projector(nn.Module):
    """Projects features from source_dim to llm_dim."""
    def __init__(self, source_dim, llm_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(source_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, x):
        return self.net(x)
