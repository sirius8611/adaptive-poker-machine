"""Training loop for the World Model agent.

Two-phase training:
  Phase 1: World Model pre-training on PokerBench
    - Train RSSM (transition, observation, reward prediction)
    - Train Adapter (contrastive loss on opponent types)

  Phase 2: Policy optimization in imagination
    - Freeze world model
    - Train policy + value heads via imagined rollouts (actor-critic)
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from world_model.agent import WorldModelAgent
from world_model.config import WorldModelConfig
from world_model.data import collate_sequences, create_dataloader
from world_model.losses import contrastive_adapter_loss, policy_loss, transition_loss

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        agent: WorldModelAgent,
        cfg: WorldModelConfig,
        device: torch.device | None = None,
    ) -> None:
        self.agent = agent
        self.cfg = cfg
        self.device = device or torch.device("cpu")
        self.agent.to(self.device)

        # Separate optimizers for different training phases
        self.world_model_optimizer = optim.Adam(
            list(agent.rssm.parameters()) + list(agent.adapter.parameters()),
            lr=cfg.lr,
        )
        self.policy_optimizer = optim.Adam(
            list(agent.policy.parameters()) + list(agent.value.parameters()),
            lr=cfg.lr,
        )

        self.global_step = 0

    def train_world_model_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Single world model training step.

        Trains RSSM + Adapter on a batch of hand histories.
        """
        self.agent.train()

        obs = batch["observations"].to(self.device)
        acts = batch["actions"].to(self.device)
        rews = batch["rewards"].to(self.device)
        opp_acts = batch["opponent_actions"].to(self.device)
        opp_mask = batch["opponent_mask"].to(self.device)
        opp_types = batch["opponent_types"].to(self.device)

        # Forward pass through world model
        out = self.agent.forward_train(obs, acts, opp_acts, opp_mask)

        # Transition loss
        trans = transition_loss(
            obs_pred=out["obs_pred"],
            obs_target=obs,
            reward_pred=out["reward_pred"],
            reward_target=rews,
            prior_logits=out["prior_logits"],
            post_logits=out["post_logits"],
            cfg=self.cfg,
        )

        # Contrastive adapter loss
        # Create positive/negative pairs from batch based on opponent types
        z_opp = out["z_opp"]  # (B, opp_embed_dim)
        contrast = self._compute_contrastive_loss(z_opp, opp_types)

        total_loss = trans["total"] + self.cfg.contrastive_scale * contrast

        self.world_model_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.agent.rssm.parameters()) + list(self.agent.adapter.parameters()),
            self.cfg.grad_clip,
        )
        self.world_model_optimizer.step()

        self.global_step += 1

        return {
            "total_loss": total_loss.item(),
            "obs_loss": trans["obs_loss"].item(),
            "reward_loss": trans["reward_loss"].item(),
            "kl_loss": trans["kl_loss"].item(),
            "contrastive_loss": contrast.item(),
        }

    def train_policy_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Single policy training step in imagination.

        Freezes world model, trains policy + value via imagined rollouts.
        """
        self.agent.rssm.eval()
        self.agent.adapter.eval()
        self.agent.policy.train()
        self.agent.value.train()

        obs = batch["observations"].to(self.device)
        acts = batch["actions"].to(self.device)
        opp_acts = batch["opponent_actions"].to(self.device)
        opp_mask = batch["opponent_mask"].to(self.device)

        with torch.no_grad():
            # Get RSSM states from real data
            out = self.agent.forward_train(obs, acts, opp_acts, opp_mask)
            # Sample random starting points for imagination
            B, T = out["h_seq"].shape[:2]
            t_start = torch.randint(0, max(T - 1, 1), (B,))
            h_start = out["h_seq"][torch.arange(B), t_start]
            z_start = out["z_seq"][torch.arange(B), t_start]
            z_opp = out["z_opp"]

        # Imagine trajectories (gradients flow through policy/value only)
        imagined = self.agent.imagine_trajectories(h_start, z_start, z_opp)

        # Policy gradient loss
        pol = policy_loss(
            action_type_logits=imagined["action_type_logits"],
            action_types=imagined["action_types"],
            bet_log_probs=imagined["bet_log_probs"],
            rewards=imagined["rewards"],
            values=imagined["values"],
            discount=self.cfg.discount,
        )

        self.policy_optimizer.zero_grad()
        pol["total"].backward()
        nn.utils.clip_grad_norm_(
            list(self.agent.policy.parameters()) + list(self.agent.value.parameters()),
            self.cfg.grad_clip,
        )
        self.policy_optimizer.step()

        return {
            "actor_loss": pol["actor_loss"].item(),
            "critic_loss": pol["critic_loss"].item(),
            "policy_total": pol["total"].item(),
        }

    def _compute_contrastive_loss(
        self, z_opp: torch.Tensor, opp_types: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive loss by pairing same/different opponent types."""
        B = z_opp.shape[0]
        if B < 4:
            return torch.tensor(0.0, device=z_opp.device)

        total_loss = torch.tensor(0.0, device=z_opp.device)
        num_pairs = 0

        for i in range(B):
            # Find positive (same type) and negatives (different type)
            same_mask = opp_types == opp_types[i]
            diff_mask = ~same_mask
            same_mask[i] = False  # Exclude self

            same_indices = same_mask.nonzero(as_tuple=True)[0]
            diff_indices = diff_mask.nonzero(as_tuple=True)[0]

            if len(same_indices) == 0 or len(diff_indices) == 0:
                continue

            # Pick one positive and up to 3 negatives
            pos_idx = same_indices[torch.randint(len(same_indices), (1,))].item()
            num_neg = min(len(diff_indices), 3)
            neg_perm = torch.randperm(len(diff_indices))[:num_neg]
            neg_idx = diff_indices[neg_perm]

            anchor = z_opp[i].unsqueeze(0)
            positive = z_opp[pos_idx].unsqueeze(0)
            negatives = z_opp[neg_idx].unsqueeze(0).unsqueeze(0)

            total_loss = total_loss + contrastive_adapter_loss(anchor, positive, negatives)
            num_pairs += 1

        return total_loss / max(num_pairs, 1)

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        phase: str = "world_model",
    ) -> dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: PokerBench data.
            phase: "world_model" or "policy"
        """
        epoch_metrics: dict[str, list[float]] = {}

        for batch in dataloader:
            if phase == "world_model":
                metrics = self.train_world_model_step(batch)
            else:
                metrics = self.train_policy_step(batch)

            for k, v in metrics.items():
                epoch_metrics.setdefault(k, []).append(v)

        return {k: sum(v) / len(v) for k, v in epoch_metrics.items()}

    def save(self, path: str) -> None:
        """Save agent checkpoint."""
        torch.save(
            {
                "agent_state_dict": self.agent.state_dict(),
                "world_model_optimizer": self.world_model_optimizer.state_dict(),
                "policy_optimizer": self.policy_optimizer.state_dict(),
                "global_step": self.global_step,
                "config": self.cfg,
            },
            path,
        )
        logger.info(f"Saved checkpoint to {path}")

    def load(self, path: str) -> None:
        """Load agent checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(ckpt["agent_state_dict"])
        self.world_model_optimizer.load_state_dict(ckpt["world_model_optimizer"])
        self.policy_optimizer.load_state_dict(ckpt["policy_optimizer"])
        self.global_step = ckpt["global_step"]
        logger.info(f"Loaded checkpoint from {path} (step {self.global_step})")
