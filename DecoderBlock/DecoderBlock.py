import torch.nn as nn
from Attention.MultiHeadAttenionBlock import MultiheadAttentionBlock
from Attention.FeedForwardBlock import FeedForwardBlock
from Attention.ResidualConnection import ResidualConnection


class DecoderBlock(nn.Module):
    def __init__(self, features:int, self_attention_block: MultiheadAttentionBlock, cross_attention_block: MultiheadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.cross_attention_block = cross_attention_block
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(features,dropout) for _ in range(3)])

    def forward(self, x, encoder_output, srv_mask, tgt_mask):
        x = self.residual_connection[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(
            x, encoder_output, encoder_output, srv_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x

