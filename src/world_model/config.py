"""Configuration for the World Model agent."""

from dataclasses import dataclass


@dataclass
class WorldModelConfig:
    # Dimensions
    state_dim: int = 256          # Deterministic state h_t
    stoch_dim: int = 64           # Stochastic state z_t (categorical: classes * categories)
    stoch_classes: int = 8        # Number of categorical distributions
    stoch_categories: int = 8     # Categories per distribution
    action_dim: int = 4           # [action_type, bet_ratio, tta_normalized, is_allin]
    obs_dim: int = 30             # Observation feature vector size
    embed_dim: int = 128          # Observation embedding
    opp_embed_dim: int = 64       # Opponent embedding z_opp
    reward_dim: int = 1           # Scalar EV prediction

    # Opponent Adapter
    adapter_context_len: int = 32   # Last N actions for temporal encoding
    adapter_heads: int = 4         # Transformer attention heads
    adapter_layers: int = 2        # Transformer layers
    adapter_action_dim: int = 6    # [action_type, bet_ratio, tta, sizing_pattern, street, position]

    # Policy Head
    policy_hidden: int = 256
    min_log_std: float = -5.0
    max_log_std: float = 2.0

    # Search
    imagination_horizon: int = 8   # Latent look-ahead depth
    num_imagined_trajectories: int = 64
    discount: float = 0.99

    # Training
    lr: float = 3e-4
    batch_size: int = 64
    seq_len: int = 50              # Sequence length for training
    kl_scale: float = 0.1          # KL balancing weight
    contrastive_scale: float = 0.5
    grad_clip: float = 100.0

    # PokerBench
    pokerbench_path: str = "data/pokerbench"
