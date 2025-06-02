import torch.nn as nn
from Attention.LayerNormalisation import LayerNormalisation


class Encoder(nn.Module):
    def __init__(self, features: int, layer: nn.ModuleList) -> None:
        super().__init__()
        self.layer = layer
        self.norm = LayerNormalisation(features)

    def forward(self, x, mask):
        for layer in self.layer:
            x = layer(x, mask)
        return self.norm(x)
