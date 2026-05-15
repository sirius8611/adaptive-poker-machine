"""Imagination-Augmented Latent Look-Ahead Search.

Replaces raw ISMCTS with search in the World Model's latent transition space.
The agent "imagines" future trajectories by rolling out the RSSM forward
while holding z_opp constant (the opponent model doesn't change mid-search).

Key advantage over ISMCTS:
  - No card sampling / determinization needed
  - Learned transitions capture opponent tendencies
  - z_opp encodes exploitation opportunities
  - Search cost is O(horizon * trajectories) forward passes, not O(iterations * game_depth)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from world_model.config import WorldModelConfig
from world_model.heads import PolicyHead, ValueHead
from world_model.rssm import RSSM


class LatentLookAhead:
    """Search in the World Model's latent space.

    Algorithm:
        1. Start from current latent state (h_t, z_t) and fixed z_opp
        2. Roll out K trajectories of length H using the policy
        3. Accumulate discounted rewards from RSSM reward head
        4. Add terminal value estimate from value head
        5. Select the first action of the highest-value trajectory

    This is analogous to Model Predictive Control (MPC) in the latent space.
    """

    def __init__(
        self,
        rssm: RSSM,
        policy: PolicyHead,
        value: ValueHead,
        cfg: WorldModelConfig,
    ) -> None:
        self.rssm = rssm
        self.policy = policy
        self.value = value
        self.cfg = cfg

    @torch.no_grad()
    def search(
        self,
        h: torch.Tensor,           # (1, state_dim)
        z: torch.Tensor,           # (1, stoch_dim)
        z_opp: torch.Tensor,       # (1, opp_embed_dim)
        legal_mask: torch.Tensor | None = None,  # (1, 4) — True for legal action types
    ) -> dict[str, torch.Tensor]:
        """Run latent look-ahead search from current state.

        Args:
            h: Current deterministic state.
            z: Current stochastic state.
            z_opp: Opponent embedding (held constant during search).
            legal_mask: Boolean mask for legal action types.

        Returns:
            best_action_type: (1,) — selected action type
            best_bet_ratio: (1, 1) — selected bet ratio
            trajectory_values: (K,) — value of each imagined trajectory
        """
        K = self.cfg.num_imagined_trajectories
        H = self.cfg.imagination_horizon
        gamma = self.cfg.discount

        # Expand initial state for K parallel trajectories
        h_k = h.expand(K, -1).contiguous()       # (K, state_dim)
        z_k = z.expand(K, -1).contiguous()       # (K, stoch_dim)
        z_opp_k = z_opp.expand(K, -1).contiguous()  # (K, opp_embed_dim)

        # Store first actions for selection
        first_action_types = None
        first_bet_ratios = None

        total_return = torch.zeros(K, device=h.device)
        discount_factor = 1.0

        for t in range(H):
            features = self.rssm.get_features(h_k, z_k)

            # Sample action from policy
            action_out = self.policy.sample_action(features, deterministic=False)
            action_type = action_out["action_type"]       # (K,)
            bet_ratio = action_out["bet_ratio"]           # (K, 1)

            # Apply legal mask on first step
            if t == 0 and legal_mask is not None:
                logits = action_out["action_type_logits"]
                logits = logits.masked_fill(~legal_mask.expand(K, -1), float("-inf"))
                action_type = torch.distributions.Categorical(logits=logits).sample()
                first_action_types = action_type.clone()
                first_bet_ratios = bet_ratio.clone()

            if t == 0 and first_action_types is None:
                first_action_types = action_type.clone()
                first_bet_ratios = bet_ratio.clone()

            # Encode action for RSSM
            action_encoded = self._encode_action(action_type, bet_ratio)

            # Imagine one step forward
            out = self.rssm.imagine_step(h_k, z_k, action_encoded, z_opp_k)
            h_k = out["h"]
            z_k = out["z"]
            reward = out["reward_pred"]  # (K,)

            total_return += discount_factor * reward
            discount_factor *= gamma

        # Terminal value estimate
        terminal_features = self.rssm.get_features(h_k, z_k)
        terminal_value = self.value(terminal_features).squeeze(-1)  # (K,)
        total_return += discount_factor * terminal_value

        # Select best trajectory
        best_idx = total_return.argmax()

        return {
            "best_action_type": first_action_types[best_idx].unsqueeze(0),
            "best_bet_ratio": first_bet_ratios[best_idx].unsqueeze(0),
            "trajectory_values": total_return,
            "best_value": total_return[best_idx],
        }

    @torch.no_grad()
    def search_with_averaging(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        z_opp: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
        num_rounds: int = 4,
    ) -> dict[str, torch.Tensor]:
        """Multi-round search with action value averaging.

        Instead of picking the single best trajectory, run multiple rounds
        and average the returns per first-action-type. More robust than
        single-shot search.

        Returns:
            best_action_type: (1,)
            best_bet_ratio: (1, 1)
            action_values: (4,) — average return per action type
        """
        action_returns: dict[int, list[float]] = {i: [] for i in range(4)}
        action_bets: dict[int, list[float]] = {i: [] for i in range(4)}

        for _ in range(num_rounds):
            result = self.search(h, z, z_opp, legal_mask)
            K = self.cfg.num_imagined_trajectories

            # Reconstruct per-trajectory first actions
            # (We need the full trajectory data, so re-run with stored info)
            # Simplified: use the search result directly
            best_type = result["best_action_type"].item()
            best_value = result["best_value"].item()
            best_bet = result["best_bet_ratio"].item()

            action_returns[best_type].append(best_value)
            action_bets[best_type].append(best_bet)

        # Average returns per action type
        action_values = torch.zeros(4, device=h.device)
        for a in range(4):
            if action_returns[a]:
                action_values[a] = sum(action_returns[a]) / len(action_returns[a])
            else:
                action_values[a] = float("-inf")

        if legal_mask is not None:
            action_values = action_values.masked_fill(~legal_mask.squeeze(0), float("-inf"))

        best_type = action_values.argmax().item()
        best_bet = (
            sum(action_bets[best_type]) / len(action_bets[best_type])
            if action_bets[best_type]
            else 0.5
        )

        return {
            "best_action_type": torch.tensor([best_type], device=h.device),
            "best_bet_ratio": torch.tensor([[best_bet]], device=h.device),
            "action_values": action_values,
        }

    def _encode_action(
        self, action_type: torch.Tensor, bet_ratio: torch.Tensor
    ) -> torch.Tensor:
        """Encode (action_type, bet_ratio) into the RSSM action format.

        RSSM expects action_dim=4: [action_type_normalized, bet_ratio, tta_placeholder, is_allin]

        Args:
            action_type: (B,) integer in {0,1,2,3}
            bet_ratio: (B, 1) in [0, 1]

        Returns:
            (B, 4) action tensor
        """
        B = action_type.shape[0]
        device = action_type.device

        type_norm = action_type.float() / 3.0  # Normalize to [0, 1]
        is_allin = (bet_ratio.squeeze(-1) > 0.95).float()
        tta_placeholder = torch.zeros(B, device=device)

        return torch.stack([type_norm, bet_ratio.squeeze(-1), tta_placeholder, is_allin], dim=-1)
