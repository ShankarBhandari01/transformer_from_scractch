import torch.nn as nn
from Attention.MultiHeadAttenionBlock import MultiheadAttentionBlock
from Attention.FeedForwardBlock import FeedForwardBlock
from Attention.ResidualConnection import ResidualConnection


class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiheadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # First residual connection with self-attention
        def self_attention_fn(x_input):
            return self.self_attention_block(x_input, x_input, x_input, src_mask)

        x = self.residual_connection[0](x, self_attention_fn)

        # Second residual connection with feed-forward
        def feed_forward_fn(x_input):
            return self.feed_forward_block(x_input)

        x = self.residual_connection[1](x, feed_forward_fn)
        return x

