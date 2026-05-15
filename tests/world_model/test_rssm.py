import torch
from world_model.config import WorldModelConfig
from world_model.rssm import RSSM


class TestRSSM:
    def setup_method(self):
        self.cfg = WorldModelConfig()
        self.rssm = RSSM(self.cfg)
        self.B = 4

    def test_initial_state_shape(self):
        h, z = self.rssm.initial_state(self.B, torch.device("cpu"))
        assert h.shape == (self.B, self.cfg.state_dim)
        assert z.shape == (self.B, self.cfg.stoch_classes * self.cfg.stoch_categories)

    def test_observe_step_shapes(self):
        h, z = self.rssm.initial_state(self.B, torch.device("cpu"))
        action = torch.randn(self.B, self.cfg.action_dim)
        z_opp = torch.randn(self.B, self.cfg.opp_embed_dim)
        obs = torch.randn(self.B, self.cfg.obs_dim)

        out = self.rssm.observe_step(h, z, action, z_opp, obs)

        assert out["h"].shape == (self.B, self.cfg.state_dim)
        assert out["z"].shape == (self.B, self.cfg.stoch_classes * self.cfg.stoch_categories)
        assert out["prior_logits"].shape == (self.B, self.cfg.stoch_classes, self.cfg.stoch_categories)
        assert out["post_logits"].shape == (self.B, self.cfg.stoch_classes, self.cfg.stoch_categories)
        assert out["reward_pred"].shape == (self.B,)
        assert out["obs_pred"].shape == (self.B, self.cfg.obs_dim)

    def test_imagine_step_shapes(self):
        h, z = self.rssm.initial_state(self.B, torch.device("cpu"))
        action = torch.randn(self.B, self.cfg.action_dim)
        z_opp = torch.randn(self.B, self.cfg.opp_embed_dim)

        out = self.rssm.imagine_step(h, z, action, z_opp)

        assert out["h"].shape == (self.B, self.cfg.state_dim)
        assert out["z"].shape == (self.B, self.cfg.stoch_classes * self.cfg.stoch_categories)
        assert out["reward_pred"].shape == (self.B,)

    def test_feature_dim(self):
        assert self.rssm.feature_dim == self.cfg.state_dim + self.cfg.stoch_classes * self.cfg.stoch_categories

    def test_deterministic_transition(self):
        """Same input → same deterministic state h."""
        torch.manual_seed(42)
        h, z = self.rssm.initial_state(1, torch.device("cpu"))
        action = torch.randn(1, self.cfg.action_dim)
        z_opp = torch.randn(1, self.cfg.opp_embed_dim)

        self.rssm.eval()
        h1 = self.rssm._deterministic_transition(h, z, action, z_opp)
        h2 = self.rssm._deterministic_transition(h, z, action, z_opp)
        assert torch.allclose(h1, h2)
