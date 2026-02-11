from __future__ import annotations

import torch
from torch import nn


class RegimeLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        market_dim: int,
        regime_dim: int,
        hidden_dim: int,
        dropout: float,
        num_regimes: int,
        use_discrete_head: bool,
    ) -> None:
        super().__init__()
        self.regime_body = nn.Sequential(
            nn.Linear(d_model + market_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, regime_dim),
        )
        self.use_discrete_head = use_discrete_head
        self.regime_logits = nn.Linear(regime_dim, num_regimes) if use_discrete_head else None

    def forward(self, weighted_world: torch.Tensor, market_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = torch.cat([weighted_world, market_x], dim=-1)
        regime_vec = self.regime_body(x)
        logits = self.regime_logits(regime_vec) if self.regime_logits is not None else None
        return regime_vec, logits
