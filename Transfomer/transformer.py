import torch.nn as nn
import torch
from DecoderBlock.DecoderBlock import DecoderBlock
from EncoderBlock.EncoderBlock import EncoderBlock
from DecoderBlock.Decoder import Decoder
from EncoderBlock.Encoder import Encoder
from Attention.InputEmeddings import InputEmbedding
from Attention.PositionalEncoding import PostionalEncoding
from Attention.MultiHeadAttenionBlock import MultiheadAttentionBlock as MultiHeadAttentionBlock
from Attention.FeedForwardBlock import FeedForwardBlock


class Transformer(nn.Module):
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: InputEmbedding,
                 tgt_embed: InputEmbedding,
                 src_pos: PostionalEncoding,
                 tgt_pos: PostionalEncoding,
                 projection_layer: nn.Module
                 ) -> None:
        super().__init__()
        self.encoder_module = encoder
        self.decoder_module = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder_module(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder_module(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)

    @staticmethod
    def build_transformer(src_vocab_size: int, tgt_vocab_size: int,
                          src_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1,
                          d_ff: int = 2048):

        src_embed = InputEmbedding(d_model, src_vocab_size)
        tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

        # Positional encodings
        src_pos = PostionalEncoding(d_model, src_seq_len, dropout)
        tgt_pos = PostionalEncoding(d_model, src_seq_len, dropout)

        # Encoder
        encoder_blocks = []
        for _ in range(N):
            encoder_blocks.append(
                EncoderBlock(
                    d_model,
                    MultiHeadAttentionBlock(d_model, h, dropout),
                    FeedForwardBlock(d_model, d_ff, dropout),
                    dropout
                )
            )

        # Decoder
        decoder_blocks = []
        for _ in range(N):
            decoder_blocks.append(
                DecoderBlock(
                    d_model,
                    MultiHeadAttentionBlock(d_model, h, dropout),
                    MultiHeadAttentionBlock(d_model, h, dropout),
                    FeedForwardBlock(d_model, d_ff, dropout),
                    dropout
                )
            )

        encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
        decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
        projection_layer = nn.Linear(d_model, tgt_vocab_size)

        transformer = Transformer(
            encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        return transformer
