import torch
from world_model.agent import WorldModelAgent
from world_model.config import WorldModelConfig


class TestWorldModelAgent:
    def setup_method(self):
        self.cfg = WorldModelConfig()
        self.agent = WorldModelAgent(self.cfg)
        self.B = 4
        self.T = 10

    def test_forward_train_shapes(self):
        obs = torch.randn(self.B, self.T, self.cfg.obs_dim)
        acts = torch.randn(self.B, self.T, self.cfg.action_dim)
        opp_acts = torch.randn(self.B, 8, self.cfg.adapter_action_dim)
        opp_mask = torch.zeros(self.B, 8, dtype=torch.bool)

        out = self.agent.forward_train(obs, acts, opp_acts, opp_mask)

        assert out["prior_logits"].shape == (self.B, self.T, self.cfg.stoch_classes, self.cfg.stoch_categories)
        assert out["post_logits"].shape == (self.B, self.T, self.cfg.stoch_classes, self.cfg.stoch_categories)
        assert out["reward_pred"].shape == (self.B, self.T)
        assert out["obs_pred"].shape == (self.B, self.T, self.cfg.obs_dim)
        assert out["z_opp"].shape == (self.B, self.cfg.opp_embed_dim)

    def test_imagine_trajectories_shapes(self):
        self.cfg.imagination_horizon = 4
        h, z = self.agent.rssm.initial_state(self.B, torch.device("cpu"))
        z_opp = torch.randn(self.B, self.cfg.opp_embed_dim)

        out = self.agent.imagine_trajectories(h, z, z_opp, horizon=4)

        assert out["action_type_logits"].shape == (self.B, 4, 4)
        assert out["action_types"].shape == (self.B, 4)
        assert out["bet_log_probs"].shape == (self.B, 4)
        assert out["rewards"].shape == (self.B, 4)
        assert out["values"].shape == (self.B, 4)

    def test_gradients_flow(self):
        """Ensure gradients flow through the full forward pass."""
        obs = torch.randn(self.B, self.T, self.cfg.obs_dim)
        acts = torch.randn(self.B, self.T, self.cfg.action_dim)
        opp_acts = torch.randn(self.B, 8, self.cfg.adapter_action_dim)
        opp_mask = torch.zeros(self.B, 8, dtype=torch.bool)

        self.agent.train()
        out = self.agent.forward_train(obs, acts, opp_acts, opp_mask)
        loss = out["reward_pred"].mean()
        loss.backward()

        # Check at least some gradients are non-zero
        has_grads = False
        for p in self.agent.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grads = True
                break
        assert has_grads
