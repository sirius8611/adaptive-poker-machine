"""Policy and Value heads for the World Model agent.

Policy Head: Outputs a continuous bet sizing distribution via TanhNormal,
  plus a discrete action type (fold/check/call/raise).

Value Head: Critic that estimates expected value from latent state.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

from world_model.config import WorldModelConfig


class TanhNormal:
    """Tanh-squashed Normal distribution for bounded continuous actions.

    Maps unbounded Normal samples to [-1, 1] via tanh, then rescales to [0, 1]
    for bet ratio representation.
    """

    def __init__(self, mu: torch.Tensor, log_std: torch.Tensor) -> None:
        self.mu = mu
        self.log_std = log_std
        self.std = log_std.exp()
        self.base_dist = Normal(mu, self.std)
        self.transforms = [TanhTransform(cache_size=1)]

    def sample(self) -> torch.Tensor:
        x = self.base_dist.rsample()
        for t in self.transforms:
            x = t(x)
        return (x + 1) / 2  # [-1, 1] → [0, 1]

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        # Map [0, 1] → [-1, 1] → atanh → normal log_prob + Jacobian
        y = value * 2 - 1  # [0, 1] → [-1, 1]
        y = y.clamp(-0.999, 0.999)
        x = torch.atanh(y)
        log_p = self.base_dist.log_prob(x)
        # Jacobian correction for tanh: -log(1 - tanh^2(x))
        log_p = log_p - torch.log(1 - y.pow(2) + 1e-6)
        # Jacobian for [0,1] rescaling: log(2)
        log_p = log_p - torch.log(torch.tensor(2.0))
        return log_p

    @property
    def mode(self) -> torch.Tensor:
        return (torch.tanh(self.mu) + 1) / 2


class PolicyHead(nn.Module):
    """Continuous policy head with discrete action type selection.

    Outputs:
        1. action_type_logits: (B, 4) — [fold, check, call, raise]
        2. bet_distribution: TanhNormal over [0, 1] — bet_size / stack
           (only meaningful when action_type = raise)

    Input: latent features from RSSM (h_t, z_t concatenated).
    """

    def __init__(self, feature_dim: int, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.shared = nn.Sequential(
            nn.Linear(feature_dim, cfg.policy_hidden),
            nn.ELU(),
            nn.Linear(cfg.policy_hidden, cfg.policy_hidden),
            nn.ELU(),
        )

        # Discrete: which action type
        self.action_type_head = nn.Linear(cfg.policy_hidden, 4)

        # Continuous: bet sizing (mu, log_std)
        self.bet_mu = nn.Linear(cfg.policy_hidden, 1)
        self.bet_log_std = nn.Linear(cfg.policy_hidden, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor | TanhNormal]:
        """Compute policy outputs.

        Args:
            features: (B, feature_dim) from RSSM.get_features()

        Returns:
            action_type_logits: (B, 4)
            bet_dist: TanhNormal distribution over [0, 1]
            bet_mode: (B, 1) — deterministic bet size (mode of distribution)
        """
        h = self.shared(features)

        action_type_logits = self.action_type_head(h)

        mu = self.bet_mu(h)
        log_std = self.bet_log_std(h)
        log_std = log_std.clamp(self.cfg.min_log_std, self.cfg.max_log_std)

        bet_dist = TanhNormal(mu, log_std)

        return {
            "action_type_logits": action_type_logits,
            "bet_dist": bet_dist,
            "bet_mode": bet_dist.mode,
        }

    def sample_action(
        self, features: torch.Tensor, deterministic: bool = False
    ) -> dict[str, torch.Tensor]:
        """Sample a complete action (type + bet size).

        Returns:
            action_type: (B,) — integer action type
            bet_ratio: (B, 1) — continuous bet size in [0, 1]
            action_type_logits: (B, 4)
            bet_log_prob: (B, 1)
        """
        out = self.forward(features)
        logits = out["action_type_logits"]
        bet_dist: TanhNormal = out["bet_dist"]

        if deterministic:
            action_type = logits.argmax(dim=-1)
            bet_ratio = bet_dist.mode
        else:
            action_type = torch.distributions.Categorical(logits=logits).sample()
            bet_ratio = bet_dist.sample()

        bet_log_prob = bet_dist.log_prob(bet_ratio)

        return {
            "action_type": action_type,
            "bet_ratio": bet_ratio,
            "action_type_logits": logits,
            "bet_log_prob": bet_log_prob,
        }


class ValueHead(nn.Module):
    """Value critic: estimates expected value from latent features.

    V(s_t) = E[sum of discounted rewards from s_t]
    """

    def __init__(self, feature_dim: int, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, cfg.policy_hidden),
            nn.ELU(),
            nn.Linear(cfg.policy_hidden, cfg.policy_hidden),
            nn.ELU(),
            nn.Linear(cfg.policy_hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns value estimate (B, 1)."""
        return self.net(features)
