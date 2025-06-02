import torch
from torch.utils.data import Dataset
from typing import Any


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_key, tgt_key, seq_len) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_key = src_key
        self.tgt_key = tgt_key
        self.seq_len = seq_len

        self.sos_token = torch.tensor(
            [tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor(
            [tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor(
            [tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: Any) -> Any:
        sample = self.ds[idx]

        src_text = sample[self.src_key]
        tgt_text = sample[self.tgt_key]

        enc_input_token = self.tokenizer_src.encode(src_text).ids
        dec_input_token = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - \
            len(enc_input_token) - 2  # for SOS and EOS
        dec_num_padding_tokens = self.seq_len - \
            len(dec_input_token) - 2  # for SOS and EOS

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence length is too long")

        # Encoder input: SOS + tokens + EOS + padding
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_token, dtype=torch.int64),
            self.eos_token,
            self.pad_token.repeat(enc_num_padding_tokens)
        ])

        # Decoder input: SOS + tokens + padding
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_token, dtype=torch.int64),
            # +1 for EOS at label
            self.pad_token.repeat(dec_num_padding_tokens + 1)
        ])

        # Label: tokens + EOS + padding
        label = torch.cat([
            torch.tensor(dec_input_token, dtype=torch.int64),
            self.eos_token,
            self.pad_token.repeat(dec_num_padding_tokens + 1)
        ])

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len,)
            "decoder_input": decoder_input,  # (seq_len,)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & self.causal_mask(decoder_input.size(0)),
            "label": label,  # (seq_len,)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

    @staticmethod
    def causal_mask(size):
        mask = torch.triu(torch.ones(1, size, size),
                          diagonal=1).type(torch.int)
        return (mask == 0)
