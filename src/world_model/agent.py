"""World Model Agent: ties RSSM + Adapter + Policy + Search together.

This is the top-level agent that:
  1. Maintains the RSSM latent state across a hand
  2. Updates z_opp via the OpponentAdapter as actions arrive
  3. Uses LatentLookAhead search (or direct policy) for action selection
  4. Bridges to the poker environment's action format
"""

from __future__ import annotations

import torch
import torch.nn as nn

from world_model.adapter import OnlineAdapterState, OpponentAdapter
from world_model.config import WorldModelConfig
from world_model.data import encode_action, encode_observation
from world_model.heads import PolicyHead, ValueHead
from world_model.rssm import RSSM
from world_model.search import LatentLookAhead


class WorldModelAgent(nn.Module):
    """Complete World Model agent for HUNL poker."""

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.rssm = RSSM(cfg)
        self.adapter = OpponentAdapter(cfg)
        self.policy = PolicyHead(self.rssm.feature_dim, cfg)
        self.value = ValueHead(self.rssm.feature_dim, cfg)
        self.search = LatentLookAhead(self.rssm, self.policy, self.value, cfg)

    def forward_train(
        self,
        observations: torch.Tensor,      # (B, T, obs_dim)
        actions: torch.Tensor,            # (B, T, action_dim)
        opponent_actions: torch.Tensor,   # (B, T_opp, adapter_action_dim)
        opponent_mask: torch.Tensor,      # (B, T_opp)
    ) -> dict[str, torch.Tensor]:
        """Forward pass for training: run RSSM with posterior (teacher forcing).

        Returns all intermediate predictions for loss computation.
        """
        B, T, _ = observations.shape
        device = observations.device

        # Encode opponent history → z_opp (constant for the hand)
        z_opp = self.adapter(opponent_actions, opponent_mask)  # (B, opp_embed_dim)

        # Initialize RSSM state
        h, z = self.rssm.initial_state(B, device)

        # Collect outputs
        all_prior_logits = []
        all_post_logits = []
        all_reward_preds = []
        all_obs_preds = []
        all_h = []
        all_z = []

        for t in range(T):
            action_t = actions[:, t]  # (B, action_dim)
            obs_t = observations[:, t]  # (B, obs_dim)

            out = self.rssm.observe_step(h, z, action_t, z_opp, obs_t)
            h = out["h"]
            z = out["z"]

            all_prior_logits.append(out["prior_logits"])
            all_post_logits.append(out["post_logits"])
            all_reward_preds.append(out["reward_pred"])
            all_obs_preds.append(out["obs_pred"])
            all_h.append(h)
            all_z.append(z)

        return {
            "prior_logits": torch.stack(all_prior_logits, dim=1),     # (B, T, C, K)
            "post_logits": torch.stack(all_post_logits, dim=1),       # (B, T, C, K)
            "reward_pred": torch.stack(all_reward_preds, dim=1),      # (B, T)
            "obs_pred": torch.stack(all_obs_preds, dim=1),            # (B, T, obs_dim)
            "h_seq": torch.stack(all_h, dim=1),                       # (B, T, state_dim)
            "z_seq": torch.stack(all_z, dim=1),                       # (B, T, stoch_dim)
            "z_opp": z_opp,                                           # (B, opp_embed_dim)
        }

    def imagine_trajectories(
        self,
        h_start: torch.Tensor,    # (B, state_dim)
        z_start: torch.Tensor,    # (B, stoch_dim)
        z_opp: torch.Tensor,      # (B, opp_embed_dim)
        horizon: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Roll out imagined trajectories for policy training.

        Uses the policy to generate actions, RSSM prior for state transitions.
        No observations — pure imagination.

        Returns:
            action_type_logits: (B, H, 4)
            action_types: (B, H)
            bet_log_probs: (B, H)
            rewards: (B, H)
            values: (B, H)
        """
        H = horizon or self.cfg.imagination_horizon
        B = h_start.shape[0]
        device = h_start.device

        h, z = h_start, z_start

        all_logits = []
        all_types = []
        all_bet_lps = []
        all_rewards = []
        all_values = []

        for t in range(H):
            features = self.rssm.get_features(h, z)

            # Policy action
            action_out = self.policy.sample_action(features, deterministic=False)
            value = self.value(features).squeeze(-1)

            # Encode action for RSSM
            action_encoded = self.search._encode_action(
                action_out["action_type"], action_out["bet_ratio"]
            )

            # Imagine step (prior only)
            out = self.rssm.imagine_step(h, z, action_encoded, z_opp)
            h = out["h"]
            z = out["z"]

            all_logits.append(action_out["action_type_logits"])
            all_types.append(action_out["action_type"])
            all_bet_lps.append(action_out["bet_log_prob"].squeeze(-1))
            all_rewards.append(out["reward_pred"])
            all_values.append(value)

        return {
            "action_type_logits": torch.stack(all_logits, dim=1),
            "action_types": torch.stack(all_types, dim=1),
            "bet_log_probs": torch.stack(all_bet_lps, dim=1),
            "rewards": torch.stack(all_rewards, dim=1),
            "values": torch.stack(all_values, dim=1),
        }


class LiveAgent:
    """Stateful wrapper for live play against a human.

    Maintains the RSSM state and opponent adapter state across
    actions within a single hand.
    """

    def __init__(
        self,
        agent: WorldModelAgent,
        cfg: WorldModelConfig,
        device: torch.device | None = None,
        use_search: bool = True,
    ) -> None:
        self.agent = agent
        self.cfg = cfg
        self.device = device or torch.device("cpu")
        self.use_search = use_search

        self.agent.to(self.device)
        self.agent.eval()

        # Persistent state
        self.adapter_state = OnlineAdapterState(cfg, self.device)
        self.h: torch.Tensor | None = None
        self.z: torch.Tensor | None = None

    def new_hand(self) -> None:
        """Reset state for a new hand."""
        self.h, self.z = self.agent.rssm.initial_state(1, self.device)

    def observe_opponent_action(
        self,
        action_type: int,
        bet_ratio: float,
        tta: float,
        sizing_pattern: int = 0,
        street: int = 0,
        position: int = 0,
    ) -> None:
        """Record an opponent action for adapter updates."""
        self.adapter_state.push(action_type, bet_ratio, tta, sizing_pattern, street, position)

    def observe_and_act(
        self,
        hero_cards: tuple[int, int],
        board: list[int],
        pot: float,
        stack: float,
        street: int,
        position: int,
        bet_facing: float,
        num_actions_street: int,
        stack_size: float,
        legal_action_types: list[int] | None = None,
    ) -> dict[str, float]:
        """Given current game state, decide an action.

        Returns:
            action_type: int (0=fold, 1=check, 2=call, 3=raise)
            bet_ratio: float in [0, 1] (bet_size / stack, only if raise)
        """
        import numpy as np

        with torch.no_grad():
            # Encode observation
            obs_np = encode_observation(
                hero_cards, board, pot, stack, street, position,
                bet_facing, num_actions_street, stack_size,
            )
            obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Get z_opp
            opp_hist = self.adapter_state.get_history()
            opp_mask = self.adapter_state.get_mask()
            z_opp = self.agent.adapter(opp_hist, opp_mask)

            # Update RSSM state with observation
            # Use a dummy action for the first observation
            if self.h is None:
                self.new_hand()

            dummy_action = torch.zeros(1, self.cfg.action_dim, device=self.device)
            out = self.agent.rssm.observe_step(self.h, self.z, dummy_action, z_opp, obs)
            self.h = out["h"]
            self.z = out["z"]

            # Decide action
            if self.use_search:
                legal_mask = None
                if legal_action_types is not None:
                    legal_mask = torch.zeros(1, 4, dtype=torch.bool, device=self.device)
                    for a in legal_action_types:
                        if 0 <= a < 4:
                            legal_mask[0, a] = True

                result = self.agent.search.search(self.h, self.z, z_opp, legal_mask)
                return {
                    "action_type": result["best_action_type"].item(),
                    "bet_ratio": result["best_bet_ratio"].item(),
                }
            else:
                features = self.agent.rssm.get_features(self.h, self.z)
                action_out = self.agent.policy.sample_action(features, deterministic=True)
                return {
                    "action_type": action_out["action_type"].item(),
                    "bet_ratio": action_out["bet_ratio"].item(),
                }

    def reset_opponent_model(self) -> None:
        """Reset the opponent model (e.g., when facing a new opponent)."""
        self.adapter_state.reset()
