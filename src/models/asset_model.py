from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.models.event_encoder import EventWorldEncoder
from src.models.moe import WorldGate, weighted_world
from src.models.regime import RegimeLayer


@dataclass
class ModelDims:
    event_dim: int
    market_dim: int
    asset_dim: int
    calendar_dim: int
    num_assets: int


class EventDrivenMarketModel(nn.Module):
    def __init__(self, dims: ModelDims, cfg: dict) -> None:
        super().__init__()
        mc = cfg["model"]
        d_model = int(mc["d_model"])
        self.use_resonance = bool(mc.get("use_resonance", True))
        self.use_calendar = bool(mc.get("use_calendar", True))

        self.event_encoder = EventWorldEncoder(
            event_dim=dims.event_dim,
            d_model=d_model,
            num_layers=int(mc["transformer_layers"]),
            num_heads=int(mc["transformer_heads"]),
            dropout=float(mc["dropout"]),
            num_worlds=int(mc["num_worlds"]),
            max_seq_len=int(cfg["features"]["sequence_length"]),
        )

        self.gate = WorldGate(
            d_model=d_model,
            market_dim=dims.market_dim,
            num_worlds=int(mc["num_worlds"]),
            hidden_dim=int(mc["hidden_dim"]),
            dropout=float(mc["dropout"]),
        )

        self.regime = RegimeLayer(
            d_model=d_model,
            market_dim=dims.market_dim,
            regime_dim=int(mc["regime_dim"]),
            hidden_dim=int(mc["hidden_dim"]),
            dropout=float(mc["dropout"]),
            num_regimes=int(mc["num_regimes"]),
            use_discrete_head=bool(mc.get("use_discrete_regime_head", False)),
        )

        self.asset_emb = nn.Embedding(dims.num_assets, int(mc["asset_embedding_dim"]))

        pred_in = (
            d_model
            + int(mc["regime_dim"])
            + dims.market_dim
            + dims.asset_dim
            + int(mc["asset_embedding_dim"])
            + (1 if self.use_resonance else 0)
            + (dims.calendar_dim if self.use_calendar else 0)
        )
        hidden = int(mc["hidden_dim"])
        drop = float(mc["dropout"])

        self.predictor = nn.Sequential(
            nn.Linear(pred_in, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, 2),
        )

    def forward(
        self,
        event_seq: torch.Tensor,
        market_x: torch.Tensor,
        asset_x: torch.Tensor,
        calendar_x: torch.Tensor,
        resonance_x: torch.Tensor,
        asset_id: torch.Tensor,
    ) -> dict:
        pooled_event, worlds, _ = self.event_encoder(event_seq)
        world_weights = self.gate(pooled_event, market_x)
        world = weighted_world(worlds, world_weights)

        regime_vec, regime_logits = self.regime(world, market_x)

        parts = [world, regime_vec, market_x, asset_x, self.asset_emb(asset_id)]
        if self.use_resonance:
            parts.append(resonance_x)
        if self.use_calendar and calendar_x.numel() > 0:
            parts.append(calendar_x)

        x = torch.cat(parts, dim=-1)
        out = self.predictor(x)
        ret_hat = out[:, :1]
        vol_hat = torch.abs(out[:, 1:2])

        regime_id = regime_logits.argmax(dim=-1, keepdim=True) if regime_logits is not None else None

        return {
            "ret_hat": ret_hat,
            "vol_hat": vol_hat,
            "world_weights": world_weights,
            "world_repr": world,
            "regime_vec": regime_vec,
            "regime_logits": regime_logits,
            "regime_id": regime_id,
        }
