from __future__ import annotations

import torch
from torch import nn


class WorldGate(nn.Module):
    def __init__(self, d_model: int, market_dim: int, num_worlds: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model + market_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_worlds),
        )

    def forward(self, pooled_event: torch.Tensor, market_x: torch.Tensor) -> torch.Tensor:
        logits = self.net(torch.cat([pooled_event, market_x], dim=-1))
        return torch.softmax(logits, dim=-1)


def weighted_world(worlds: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    # worlds: [B, K, D], weights: [B, K]
    return torch.sum(worlds * weights.unsqueeze(-1), dim=1)
