"""Opponent Adapter: Fast-Path Online Adaptation via z_opp.

Processes the last N opponent actions (including Time-to-Action and Sizing
Patterns) through a Transformer encoder to produce a latent opponent
embedding z_opp that modulates the RSSM's transition predictions.

z_opp captures:
  - Playing style (tight/loose, passive/aggressive)
  - Timing tells (fast = confident, slow = uncertain)
  - Bet sizing patterns (polarized vs merged ranges)

The adapter updates in real-time as new actions are observed,
without retraining the world model.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model.config import WorldModelConfig


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for action sequences."""

    def __init__(self, d_model: int, max_len: int = 128) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2 + d_model % 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class OpponentAdapter(nn.Module):
    """Temporal Transformer encoder that produces z_opp from opponent action history.

    Input per timestep (adapter_action_dim=6):
        [action_type, bet_ratio, tta, sizing_pattern, street, position]

    Where:
        action_type:    0=fold, 1=check, 2=call, 3=raise (one-hot internally)
        bet_ratio:      continuous [0, 1] — bet size / stack
        tta:            Time-to-Action in seconds, log-normalized
        sizing_pattern: categorical encoding of sizing tendency
        street:         0-3 (preflop/flop/turn/river)
        position:       0=IP, 1=OOP

    Output:
        z_opp: (B, opp_embed_dim) — opponent embedding that modulates RSSM
    """

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Project raw action features to transformer dimension
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.adapter_action_dim, cfg.opp_embed_dim),
            nn.ELU(),
            nn.Linear(cfg.opp_embed_dim, cfg.opp_embed_dim),
        )

        self.pos_enc = PositionalEncoding(cfg.opp_embed_dim, max_len=cfg.adapter_context_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.opp_embed_dim,
            nhead=cfg.adapter_heads,
            dim_feedforward=cfg.opp_embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.adapter_layers
        )

        # Aggregate sequence → single vector
        self.output_proj = nn.Sequential(
            nn.Linear(cfg.opp_embed_dim, cfg.opp_embed_dim),
            nn.ELU(),
            nn.Linear(cfg.opp_embed_dim, cfg.opp_embed_dim),
        )

        # Learnable "no history" token for when we have no opponent data
        self.default_embedding = nn.Parameter(torch.randn(cfg.opp_embed_dim) * 0.01)

    def forward(
        self,
        action_history: torch.Tensor,  # (B, T, adapter_action_dim)
        mask: torch.Tensor | None = None,  # (B, T) — True for padded positions
    ) -> torch.Tensor:
        """Encode opponent action history into z_opp.

        Args:
            action_history: Opponent's recent actions with metadata.
            mask: Padding mask (True = ignore this position).

        Returns:
            z_opp: (B, opp_embed_dim)
        """
        B, T, _ = action_history.shape

        if T == 0:
            return self.default_embedding.unsqueeze(0).expand(B, -1)

        # Project and add positional encoding
        x = self.input_proj(action_history)
        x = self.pos_enc(x)

        # Create causal mask for transformer (attend only to past)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)

        # Convert padding mask: TransformerEncoder expects (B, T) with True = ignore
        src_key_padding_mask = mask if mask is not None else None

        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)

        # Aggregate: use the last non-padded position
        if mask is not None:
            # Find last valid position per batch
            lengths = (~mask).sum(dim=1).clamp(min=1)  # (B,)
            # Gather last valid token
            indices = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, self.cfg.opp_embed_dim)
            pooled = x.gather(1, indices).squeeze(1)
        else:
            pooled = x[:, -1]  # Last position

        z_opp = self.output_proj(pooled)
        return z_opp

    def encode_single_action(
        self,
        action_type: int,
        bet_ratio: float,
        tta: float,
        sizing_pattern: int,
        street: int,
        position: int,
    ) -> torch.Tensor:
        """Encode a single action into the adapter_action_dim format.

        Returns: (1, adapter_action_dim) tensor
        """
        tta_norm = math.log1p(tta)  # Log-normalize TTA
        return torch.tensor(
            [[action_type, bet_ratio, tta_norm, sizing_pattern, street, position]],
            dtype=torch.float32,
        )


class OnlineAdapterState:
    """Maintains a rolling buffer of opponent actions for real-time adaptation.

    Usage during live play:
        state = OnlineAdapterState(cfg)
        state.push(action_type=3, bet_ratio=0.75, tta=2.1, ...)
        z_opp = adapter(state.get_history(), state.get_mask())
    """

    def __init__(self, cfg: WorldModelConfig, device: torch.device | None = None) -> None:
        self.cfg = cfg
        self.device = device or torch.device("cpu")
        self.buffer: list[torch.Tensor] = []

    def push(
        self,
        action_type: int,
        bet_ratio: float,
        tta: float,
        sizing_pattern: int = 0,
        street: int = 0,
        position: int = 0,
    ) -> None:
        """Add an opponent action to the rolling buffer."""
        entry = torch.tensor(
            [action_type, bet_ratio, math.log1p(tta), sizing_pattern, street, position],
            dtype=torch.float32,
            device=self.device,
        )
        self.buffer.append(entry)
        # Keep only last N actions
        if len(self.buffer) > self.cfg.adapter_context_len:
            self.buffer = self.buffer[-self.cfg.adapter_context_len :]

    def get_history(self) -> torch.Tensor:
        """Get padded action history tensor. Shape: (1, context_len, action_dim)."""
        T = self.cfg.adapter_context_len
        if not self.buffer:
            return torch.zeros(1, T, self.cfg.adapter_action_dim, device=self.device)

        stacked = torch.stack(self.buffer)  # (num_actions, action_dim)
        # Pad to context_len
        pad_len = T - len(self.buffer)
        if pad_len > 0:
            padding = torch.zeros(pad_len, self.cfg.adapter_action_dim, device=self.device)
            stacked = torch.cat([padding, stacked], dim=0)

        return stacked.unsqueeze(0)  # (1, T, action_dim)

    def get_mask(self) -> torch.Tensor:
        """Get padding mask. Shape: (1, context_len). True = padded."""
        T = self.cfg.adapter_context_len
        num_valid = min(len(self.buffer), T)
        mask = torch.ones(1, T, dtype=torch.bool, device=self.device)
        if num_valid > 0:
            mask[0, -num_valid:] = False
        return mask

    def reset(self) -> None:
        self.buffer.clear()
