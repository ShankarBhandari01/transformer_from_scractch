import torch.nn as nn
import torch


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocav_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, vocav_size)

    def forward(self, x):
        # (Batch seq_len,d_model )--> (Batch seq_len,vocab_size)
        return torch.log_softmax(self.projection(x), dim=-1)
