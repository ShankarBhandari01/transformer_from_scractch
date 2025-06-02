import torch
import torch.nn as nn
from Attention.LayerNormalisation import LayerNormalisation


class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout=float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalisation(features)

    def forward(self, x, sublayer):
        normalized = self.norm(x)
        sublayer_output = sublayer(normalized)
        # Ensure sublayer_output is a tensor
        if not isinstance(sublayer_output, torch.Tensor):
            raise TypeError(f"Sublayer must return a torch.Tensor, got {type(sublayer_output)}")

        dropped = self.dropout(sublayer_output)
        return x + dropped
