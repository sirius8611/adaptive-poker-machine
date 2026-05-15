import torch
from world_model.config import WorldModelConfig
from world_model.losses import contrastive_adapter_loss, policy_loss, transition_loss


class TestTransitionLoss:
    def test_shape_and_finite(self):
        cfg = WorldModelConfig()
        B, T = 4, 10

        out = transition_loss(
            obs_pred=torch.randn(B, T, cfg.obs_dim),
            obs_target=torch.randn(B, T, cfg.obs_dim),
            reward_pred=torch.randn(B, T),
            reward_target=torch.randn(B, T),
            prior_logits=torch.randn(B, T, cfg.stoch_classes, cfg.stoch_categories),
            post_logits=torch.randn(B, T, cfg.stoch_classes, cfg.stoch_categories),
            cfg=cfg,
        )

        assert torch.isfinite(out["total"])
        assert torch.isfinite(out["obs_loss"])
        assert torch.isfinite(out["reward_loss"])
        assert torch.isfinite(out["kl_loss"])

    def test_zero_loss_on_perfect_prediction(self):
        cfg = WorldModelConfig()
        B, T = 4, 10
        obs = torch.randn(B, T, cfg.obs_dim)
        rew = torch.randn(B, T)
        logits = torch.randn(B, T, cfg.stoch_classes, cfg.stoch_categories)

        out = transition_loss(obs, obs, rew, rew, logits, logits, cfg)
        assert out["obs_loss"].item() < 1e-6
        assert out["reward_loss"].item() < 1e-6


class TestContrastiveLoss:
    def test_shape(self):
        B, D, N = 8, 64, 3
        loss = contrastive_adapter_loss(
            z_anchor=torch.randn(B, D),
            z_positive=torch.randn(B, D),
            z_negatives=torch.randn(B, N, D),
        )
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_low_loss_for_similar(self):
        B, D, N = 8, 64, 3
        anchor = torch.randn(B, D)
        positive = anchor + 0.01 * torch.randn(B, D)  # Very similar
        negatives = torch.randn(B, N, D)  # Random

        loss = contrastive_adapter_loss(anchor, positive, negatives, temperature=0.1)
        # Loss should be relatively low since positive is very similar
        assert loss.item() < 2.0


class TestPolicyLoss:
    def test_shape(self):
        B, H = 4, 8
        out = policy_loss(
            action_type_logits=torch.randn(B, H, 4),
            action_types=torch.randint(0, 4, (B, H)),
            bet_log_probs=torch.randn(B, H),
            rewards=torch.randn(B, H),
            values=torch.randn(B, H),
        )
        assert torch.isfinite(out["actor_loss"])
        assert torch.isfinite(out["critic_loss"])
        assert torch.isfinite(out["total"])
