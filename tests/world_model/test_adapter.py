import torch
from world_model.adapter import OnlineAdapterState, OpponentAdapter
from world_model.config import WorldModelConfig


class TestOpponentAdapter:
    def setup_method(self):
        self.cfg = WorldModelConfig()
        self.adapter = OpponentAdapter(self.cfg)
        self.B = 4
        self.T = 10

    def test_output_shape(self):
        actions = torch.randn(self.B, self.T, self.cfg.adapter_action_dim)
        z_opp = self.adapter(actions)
        assert z_opp.shape == (self.B, self.cfg.opp_embed_dim)

    def test_with_mask(self):
        actions = torch.randn(self.B, self.T, self.cfg.adapter_action_dim)
        mask = torch.zeros(self.B, self.T, dtype=torch.bool)
        mask[:, :3] = True  # First 3 positions are padding
        z_opp = self.adapter(actions, mask)
        assert z_opp.shape == (self.B, self.cfg.opp_embed_dim)

    def test_empty_history_uses_default(self):
        actions = torch.randn(self.B, 0, self.cfg.adapter_action_dim)
        z_opp = self.adapter(actions)
        assert z_opp.shape == (self.B, self.cfg.opp_embed_dim)

    def test_different_inputs_different_outputs(self):
        a1 = torch.randn(1, self.T, self.cfg.adapter_action_dim)
        a2 = torch.randn(1, self.T, self.cfg.adapter_action_dim)
        z1 = self.adapter(a1)
        z2 = self.adapter(a2)
        assert not torch.allclose(z1, z2, atol=1e-3)


class TestOnlineAdapterState:
    def setup_method(self):
        self.cfg = WorldModelConfig()
        self.state = OnlineAdapterState(self.cfg)

    def test_empty_state(self):
        hist = self.state.get_history()
        mask = self.state.get_mask()
        assert hist.shape == (1, self.cfg.adapter_context_len, self.cfg.adapter_action_dim)
        assert mask.all()  # All masked

    def test_push_and_retrieve(self):
        self.state.push(action_type=3, bet_ratio=0.5, tta=2.0, street=1)
        hist = self.state.get_history()
        mask = self.state.get_mask()
        assert not mask[0, -1]  # Last position unmasked
        assert mask[0, 0]       # First position still masked

    def test_rolling_buffer(self):
        for i in range(self.cfg.adapter_context_len + 10):
            self.state.push(action_type=1, bet_ratio=0.3, tta=1.0)
        assert len(self.state.buffer) == self.cfg.adapter_context_len

    def test_reset(self):
        self.state.push(action_type=2, bet_ratio=0.5, tta=1.0)
        self.state.reset()
        assert len(self.state.buffer) == 0
