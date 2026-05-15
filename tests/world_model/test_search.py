import torch
from world_model.config import WorldModelConfig
from world_model.heads import PolicyHead, ValueHead
from world_model.rssm import RSSM
from world_model.search import LatentLookAhead


class TestLatentLookAhead:
    def setup_method(self):
        self.cfg = WorldModelConfig(
            num_imagined_trajectories=8,
            imagination_horizon=4,
        )
        self.rssm = RSSM(self.cfg)
        self.policy = PolicyHead(self.rssm.feature_dim, self.cfg)
        self.value = ValueHead(self.rssm.feature_dim, self.cfg)
        self.search = LatentLookAhead(self.rssm, self.policy, self.value, self.cfg)

        self.rssm.eval()
        self.policy.eval()
        self.value.eval()

    def test_search_returns_valid_action(self):
        h, z = self.rssm.initial_state(1, torch.device("cpu"))
        z_opp = torch.randn(1, self.cfg.opp_embed_dim)

        result = self.search.search(h, z, z_opp)
        assert result["best_action_type"].shape == (1,)
        assert 0 <= result["best_action_type"].item() < 4
        assert result["best_bet_ratio"].numel() == 1
        assert 0 <= result["best_bet_ratio"].item() <= 1

    def test_legal_mask_respected(self):
        h, z = self.rssm.initial_state(1, torch.device("cpu"))
        z_opp = torch.randn(1, self.cfg.opp_embed_dim)

        # Only allow check (1) and raise (3)
        legal_mask = torch.tensor([[False, True, False, True]])
        result = self.search.search(h, z, z_opp, legal_mask)
        assert result["best_action_type"].item() in [1, 3]

    def test_trajectory_values_shape(self):
        h, z = self.rssm.initial_state(1, torch.device("cpu"))
        z_opp = torch.randn(1, self.cfg.opp_embed_dim)

        result = self.search.search(h, z, z_opp)
        assert result["trajectory_values"].shape == (self.cfg.num_imagined_trajectories,)

    def test_search_with_averaging(self):
        h, z = self.rssm.initial_state(1, torch.device("cpu"))
        z_opp = torch.randn(1, self.cfg.opp_embed_dim)

        result = self.search.search_with_averaging(h, z, z_opp, num_rounds=2)
        assert result["best_action_type"].shape == (1,)
        assert result["action_values"].shape == (4,)
