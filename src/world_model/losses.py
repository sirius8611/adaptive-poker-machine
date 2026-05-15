"""Loss functions for training the World Model agent.

Three core objectives:

1. Transition Loss (L_trans): Train RSSM to predict next states accurately
   - Observation reconstruction loss
   - Reward prediction loss
   - KL divergence between prior and posterior (regularizes stochastic state)

2. Contrastive Adapter Loss (L_contrast): Train z_opp to differentiate
   opponent types by their TTA and sizing patterns
   - Pulls together embeddings of same-style opponents
   - Pushes apart embeddings of different-style opponents

3. Policy Loss (L_policy): Actor-critic in imagination
   - Policy gradient with value baseline in latent trajectories

Mathematical formulation:

  L_trans = E_t[ ||o_t - ô_t||² + ||r_t - r̂_t||² + β·KL[q(z|h,o) || p(z|h)] ]

  L_contrast = -log( exp(sim(z_i, z_j⁺)/τ) / Σ_k exp(sim(z_i, z_k)/τ) )
    where z_j⁺ is from same opponent type, z_k are all samples

  L_policy = -E_π[ Σ_t γ^t · (R_t - V(s_t)) · log π(a_t|s_t) ]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model.config import WorldModelConfig


def transition_loss(
    obs_pred: torch.Tensor,        # (B, T, obs_dim) — reconstructed observations
    obs_target: torch.Tensor,      # (B, T, obs_dim) — ground truth observations
    reward_pred: torch.Tensor,     # (B, T) — predicted rewards
    reward_target: torch.Tensor,   # (B, T) — ground truth rewards
    prior_logits: torch.Tensor,    # (B, T, stoch_classes, stoch_categories)
    post_logits: torch.Tensor,     # (B, T, stoch_classes, stoch_categories)
    cfg: WorldModelConfig,
) -> dict[str, torch.Tensor]:
    """Compute world model transition loss.

    L_trans = L_recon + L_reward + β * L_kl

    The KL term uses categorical distributions (matching RSSM stochastic state).
    KL balancing: we compute KL in both directions and weight to prevent
    posterior collapse while keeping the prior informative.
    """
    # Observation reconstruction loss (MSE)
    obs_loss = F.mse_loss(obs_pred, obs_target)

    # Reward prediction loss (MSE)
    reward_loss = F.mse_loss(reward_pred, reward_target)

    # KL divergence between posterior and prior (categorical)
    # prior_logits, post_logits: (B, T, classes, categories)
    prior_probs = F.softmax(prior_logits, dim=-1)
    post_probs = F.softmax(post_logits, dim=-1)

    # KL[posterior || prior]
    kl_fwd = (post_probs * (post_probs.log() - prior_probs.log() + 1e-8)).sum(dim=-1)
    kl_fwd = kl_fwd.sum(dim=-1).mean()  # Sum over classes, mean over batch/time

    # KL balancing: 80% pull prior toward posterior, 20% pull posterior toward prior
    # This prevents posterior collapse
    kl_rev = (prior_probs * (prior_probs.log() - post_probs.log() + 1e-8)).sum(dim=-1)
    kl_rev = kl_rev.sum(dim=-1).mean()

    kl_loss = 0.8 * kl_fwd + 0.2 * kl_rev

    total = obs_loss + reward_loss + cfg.kl_scale * kl_loss

    return {
        "total": total,
        "obs_loss": obs_loss,
        "reward_loss": reward_loss,
        "kl_loss": kl_loss,
    }


def contrastive_adapter_loss(
    z_anchor: torch.Tensor,     # (B, opp_embed_dim) — anchor embeddings
    z_positive: torch.Tensor,   # (B, opp_embed_dim) — same-type embeddings
    z_negatives: torch.Tensor,  # (B, N, opp_embed_dim) — different-type embeddings
    temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE contrastive loss for opponent adapter.

    Encourages z_opp to cluster by opponent type:
    - "Tight-passive" vs "Loose-aggressive" separated in embedding space
    - Fast TTA (confident) vs Slow TTA (uncertain) distinguished

    L = -log( exp(sim(anchor, positive)/τ) / (exp(sim(anchor, positive)/τ) + Σ exp(sim(anchor, neg_k)/τ)) )

    Args:
        z_anchor: Embeddings from one sequence of opponent actions.
        z_positive: Embeddings from another sequence of the SAME opponent type.
        z_negatives: Embeddings from sequences of DIFFERENT opponent types.
        temperature: Softmax temperature (lower = sharper discrimination).

    Returns:
        Scalar contrastive loss.
    """
    B = z_anchor.shape[0]

    # Normalize embeddings for cosine similarity
    z_anchor = F.normalize(z_anchor, dim=-1)
    z_positive = F.normalize(z_positive, dim=-1)
    z_negatives = F.normalize(z_negatives, dim=-1)

    # Positive similarity: (B,)
    pos_sim = (z_anchor * z_positive).sum(dim=-1) / temperature

    # Negative similarities: (B, N)
    neg_sim = torch.bmm(z_negatives, z_anchor.unsqueeze(-1)).squeeze(-1) / temperature

    # InfoNCE: log-softmax over [positive, negatives]
    logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # (B, 1+N)
    labels = torch.zeros(B, dtype=torch.long, device=z_anchor.device)  # positive is index 0

    loss = F.cross_entropy(logits, labels)
    return loss


def policy_loss(
    action_type_logits: torch.Tensor,  # (B, H, 4)
    action_types: torch.Tensor,         # (B, H) — sampled action types
    bet_log_probs: torch.Tensor,        # (B, H)
    rewards: torch.Tensor,              # (B, H)
    values: torch.Tensor,               # (B, H)
    discount: float = 0.99,
) -> dict[str, torch.Tensor]:
    """Actor-critic policy gradient loss in imagined trajectories.

    L_actor = -E[ (R_λ - V(s_t)) · log π(a_t | s_t) ]
    L_critic = E[ (R_λ - V(s_t))² ]

    Where R_λ is the lambda-return computed from imagined rewards.

    Args:
        action_type_logits: Logits from policy head at each step.
        action_types: Actions actually taken (sampled).
        bet_log_probs: Log-probs of continuous bet ratio.
        rewards: Imagined rewards from RSSM.
        values: Value estimates at each step.
        discount: Discount factor.
    """
    B, H = rewards.shape

    # Compute lambda-returns (GAE-like target)
    lambda_returns = _compute_lambda_returns(rewards, values, discount, lambda_=0.95)

    # Advantage
    advantages = (lambda_returns - values).detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Actor loss: discrete + continuous
    action_type_log_probs = F.log_softmax(action_type_logits, dim=-1)
    action_type_log_probs = action_type_log_probs.gather(
        -1, action_types.unsqueeze(-1)
    ).squeeze(-1)  # (B, H)

    total_log_prob = action_type_log_probs + bet_log_probs
    actor_loss = -(total_log_prob * advantages).mean()

    # Critic loss
    critic_loss = F.mse_loss(values, lambda_returns.detach())

    return {
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "total": actor_loss + 0.5 * critic_loss,
    }


def _compute_lambda_returns(
    rewards: torch.Tensor,   # (B, H)
    values: torch.Tensor,    # (B, H)
    discount: float,
    lambda_: float,
) -> torch.Tensor:
    """Compute lambda-returns for generalized advantage estimation.

    R_λ(t) = r_t + γ·[(1-λ)·V(s_{t+1}) + λ·R_λ(t+1)]
    """
    B, H = rewards.shape
    returns = torch.zeros_like(rewards)

    # Bootstrap from last value
    last_value = values[:, -1]
    returns[:, -1] = rewards[:, -1] + discount * last_value

    for t in reversed(range(H - 1)):
        next_return = returns[:, t + 1]
        next_value = values[:, t + 1]
        returns[:, t] = rewards[:, t] + discount * (
            (1 - lambda_) * next_value + lambda_ * next_return
        )

    return returns
