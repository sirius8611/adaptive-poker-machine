"""Recurrent State Space Model (RSSM) for HUNL Poker World Model.

The RSSM maintains a latent state s_t = (h_t, z_t) where:
  - h_t: Deterministic recurrent state (captures temporal dynamics)
  - z_t: Stochastic latent (captures uncertainty / hidden info like opponent cards)

The RSSM replaces ISMCTS determinization by learning to predict the
Public Belief State (PBS) directly in latent space.

Transition model:  h_t = f(h_{t-1}, z_{t-1}, a_{t-1}, z_opp)    [deterministic]
Prior:             z_t ~ p(z_t | h_t)                             [imagination]
Posterior:         z_t ~ q(z_t | h_t, o_t)                        [training]
Reward:            r_t = R(h_t, z_t)                              [EV prediction]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model.config import WorldModelConfig


class RSSM(nn.Module):
    """Recurrent State Space Model for poker world modeling.

    The key insight: z_opp (opponent embedding) modulates the transition
    function, allowing the model to predict different state evolutions
    for different opponent types without re-determinizing.
    """

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.stoch_dim = cfg.stoch_classes * cfg.stoch_categories

        # --- Observation Encoder ---
        # Maps raw observation features to an embedding
        self.obs_encoder = nn.Sequential(
            nn.Linear(cfg.obs_dim, cfg.embed_dim),
            nn.ELU(),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
            nn.ELU(),
        )

        # --- Transition Model (Prior) ---
        # h_t = f(h_{t-1}, z_{t-1}, a_{t-1}, z_opp)
        # Input: h_{t-1} + z_{t-1} + a_{t-1} + z_opp
        trans_input_dim = cfg.state_dim + self.stoch_dim + cfg.action_dim + cfg.opp_embed_dim
        self.trans_pre = nn.Sequential(
            nn.Linear(trans_input_dim, cfg.state_dim),
            nn.ELU(),
        )
        self.gru = nn.GRUCell(cfg.state_dim, cfg.state_dim)

        # Prior: p(z_t | h_t) — used during imagination
        self.prior_net = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.embed_dim),
            nn.ELU(),
            nn.Linear(cfg.embed_dim, self.stoch_dim),
        )

        # Posterior: q(z_t | h_t, o_t) — used during training
        self.posterior_net = nn.Sequential(
            nn.Linear(cfg.state_dim + cfg.embed_dim, cfg.embed_dim),
            nn.ELU(),
            nn.Linear(cfg.embed_dim, self.stoch_dim),
        )

        # --- Reward Predictor ---
        # R(h_t, z_t) → scalar EV
        self.reward_head = nn.Sequential(
            nn.Linear(cfg.state_dim + self.stoch_dim, cfg.embed_dim),
            nn.ELU(),
            nn.Linear(cfg.embed_dim, cfg.embed_dim // 2),
            nn.ELU(),
            nn.Linear(cfg.embed_dim // 2, 1),
        )

        # --- Observation Decoder (reconstruction) ---
        self.obs_decoder = nn.Sequential(
            nn.Linear(cfg.state_dim + self.stoch_dim, cfg.embed_dim),
            nn.ELU(),
            nn.Linear(cfg.embed_dim, cfg.obs_dim),
        )

    def initial_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return initial (h_0, z_0)."""
        h = torch.zeros(batch_size, self.cfg.state_dim, device=device)
        z = torch.zeros(batch_size, self.stoch_dim, device=device)
        return h, z

    def observe_step(
        self,
        prev_h: torch.Tensor,      # (B, state_dim)
        prev_z: torch.Tensor,      # (B, stoch_dim)
        action: torch.Tensor,      # (B, action_dim)
        z_opp: torch.Tensor,       # (B, opp_embed_dim)
        obs: torch.Tensor,         # (B, obs_dim)
    ) -> dict[str, torch.Tensor]:
        """Single observation step: compute posterior z_t given observation.

        Used during training when we have ground-truth observations.
        """
        # 1. Deterministic transition
        h = self._deterministic_transition(prev_h, prev_z, action, z_opp)

        # 2. Encode observation
        obs_embed = self.obs_encoder(obs)

        # 3. Prior p(z_t | h_t)
        prior_logits = self.prior_net(h)
        prior_logits = prior_logits.view(-1, self.cfg.stoch_classes, self.cfg.stoch_categories)

        # 4. Posterior q(z_t | h_t, o_t)
        post_input = torch.cat([h, obs_embed], dim=-1)
        post_logits = self.posterior_net(post_input)
        post_logits = post_logits.view(-1, self.cfg.stoch_classes, self.cfg.stoch_categories)

        # 5. Sample z from posterior (straight-through Gumbel-Softmax)
        z = self._sample_stochastic(post_logits)
        z_flat = z.view(-1, self.stoch_dim)

        # 6. Predictions
        state_feat = torch.cat([h, z_flat], dim=-1)
        reward_pred = self.reward_head(state_feat)
        obs_pred = self.obs_decoder(state_feat)

        return {
            "h": h,
            "z": z_flat,
            "prior_logits": prior_logits,
            "post_logits": post_logits,
            "reward_pred": reward_pred.squeeze(-1),
            "obs_pred": obs_pred,
        }

    def imagine_step(
        self,
        prev_h: torch.Tensor,      # (B, state_dim)
        prev_z: torch.Tensor,      # (B, stoch_dim)
        action: torch.Tensor,      # (B, action_dim)
        z_opp: torch.Tensor,       # (B, opp_embed_dim)
    ) -> dict[str, torch.Tensor]:
        """Single imagination step: use prior (no observation).

        Used during planning / latent look-ahead.
        """
        # 1. Deterministic transition
        h = self._deterministic_transition(prev_h, prev_z, action, z_opp)

        # 2. Prior only (no observation available)
        prior_logits = self.prior_net(h)
        prior_logits = prior_logits.view(-1, self.cfg.stoch_classes, self.cfg.stoch_categories)
        z = self._sample_stochastic(prior_logits)
        z_flat = z.view(-1, self.stoch_dim)

        # 3. Predictions
        state_feat = torch.cat([h, z_flat], dim=-1)
        reward_pred = self.reward_head(state_feat)

        return {
            "h": h,
            "z": z_flat,
            "prior_logits": prior_logits,
            "reward_pred": reward_pred.squeeze(-1),
        }

    def _deterministic_transition(
        self,
        prev_h: torch.Tensor,
        prev_z: torch.Tensor,
        action: torch.Tensor,
        z_opp: torch.Tensor,
    ) -> torch.Tensor:
        """h_t = GRU(f(h_{t-1}, z_{t-1}, a_{t-1}, z_opp), h_{t-1})."""
        x = torch.cat([prev_h, prev_z, action, z_opp], dim=-1)
        x = self.trans_pre(x)
        h = self.gru(x, prev_h)
        return h

    def _sample_stochastic(self, logits: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
        """Sample from categorical distribution with straight-through gradients.

        Args:
            logits: (B, stoch_classes, stoch_categories)
        Returns:
            One-hot samples: (B, stoch_classes, stoch_categories)
        """
        if self.training:
            # Straight-through Gumbel-Softmax
            z = F.gumbel_softmax(logits, tau=temp, hard=True, dim=-1)
        else:
            # Argmax at inference
            indices = logits.argmax(dim=-1)
            z = F.one_hot(indices, self.cfg.stoch_categories).float()
        return z

    def get_features(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Concatenate h and z into a single feature vector for downstream heads."""
        return torch.cat([h, z], dim=-1)

    @property
    def feature_dim(self) -> int:
        return self.cfg.state_dim + self.stoch_dim
