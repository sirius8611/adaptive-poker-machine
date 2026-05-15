import torch
from world_model.config import WorldModelConfig
from world_model.heads import PolicyHead, TanhNormal, ValueHead
from world_model.rssm import RSSM


class TestTanhNormal:
    def test_sample_in_range(self):
        mu = torch.zeros(100, 1)
        log_std = torch.zeros(100, 1)
        dist = TanhNormal(mu, log_std)
        samples = dist.sample()
        assert (samples >= 0).all() and (samples <= 1).all()

    def test_log_prob_finite(self):
        mu = torch.zeros(10, 1)
        log_std = torch.zeros(10, 1)
        dist = TanhNormal(mu, log_std)
        samples = dist.sample()
        lp = dist.log_prob(samples)
        assert torch.isfinite(lp).all()

    def test_mode_in_range(self):
        mu = torch.randn(10, 1)
        log_std = torch.zeros(10, 1)
        dist = TanhNormal(mu, log_std)
        assert (dist.mode >= 0).all() and (dist.mode <= 1).all()


class TestPolicyHead:
    def setup_method(self):
        self.cfg = WorldModelConfig()
        self.rssm = RSSM(self.cfg)
        self.policy = PolicyHead(self.rssm.feature_dim, self.cfg)
        self.B = 4

    def test_forward_shapes(self):
        features = torch.randn(self.B, self.rssm.feature_dim)
        out = self.policy(features)
        assert out["action_type_logits"].shape == (self.B, 4)
        assert out["bet_mode"].shape == (self.B, 1)

    def test_sample_action(self):
        features = torch.randn(self.B, self.rssm.feature_dim)
        out = self.policy.sample_action(features)
        assert out["action_type"].shape == (self.B,)
        assert out["bet_ratio"].shape == (self.B, 1)
        assert (out["bet_ratio"] >= 0).all() and (out["bet_ratio"] <= 1).all()
        assert (out["action_type"] >= 0).all() and (out["action_type"] < 4).all()

    def test_deterministic_mode(self):
        features = torch.randn(1, self.rssm.feature_dim)
        a1 = self.policy.sample_action(features, deterministic=True)
        a2 = self.policy.sample_action(features, deterministic=True)
        assert a1["action_type"] == a2["action_type"]
        assert torch.allclose(a1["bet_ratio"], a2["bet_ratio"])


class TestValueHead:
    def test_output_shape(self):
        cfg = WorldModelConfig()
        rssm = RSSM(cfg)
        value = ValueHead(rssm.feature_dim, cfg)
        features = torch.randn(4, rssm.feature_dim)
        v = value(features)
        assert v.shape == (4, 1)
