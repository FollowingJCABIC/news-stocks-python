from __future__ import annotations

import torch
from torch import nn


class EventWorldEncoder(nn.Module):
    def __init__(
        self,
        event_dim: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        num_worlds: int,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(event_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.world_proj = nn.Linear(d_model, d_model * num_worlds)
        self.num_worlds = num_worlds
        self.d_model = d_model

    def forward(self, event_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # event_seq: [B, L, E]
        b, l, _ = event_seq.shape
        x = self.input_proj(event_seq)
        x = x + self.pos_emb[:, :l, :]
        h = self.encoder(x)
        pooled = self.norm(h.mean(dim=1))
        worlds = self.world_proj(pooled).view(b, self.num_worlds, self.d_model)
        return pooled, worlds, h
