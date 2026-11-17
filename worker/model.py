"""Single source of truth for the model architecture.

Start tiny (~10M params) to validate the pipeline end-to-end, then scale.
Every worker and the aggregator import build_model() so state_dicts always line up.
"""
import torch.nn as nn


def build_model(vocab_size: int = 50257, d_model: int = 256,
                n_layers: int = 4, n_heads: int = 4, seq_len: int = 512):
    layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
        batch_first=True, norm_first=True,
    )

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.pos = nn.Embedding(seq_len, d_model)
            self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx):
            import torch
            b, t = idx.shape
            pos = torch.arange(t, device=idx.device).unsqueeze(0)
            x = self.embed(idx) + self.pos(pos)
            mask = nn.Transformer.generate_square_subsequent_mask(t).to(idx.device)
            x = self.blocks(x, mask=mask, is_causal=True)
            return self.head(x)

    return TinyLM()
