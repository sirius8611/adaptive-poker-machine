import numpy as np
import torch
from world_model.config import WorldModelConfig
from world_model.data import (
    classify_opponent_type,
    encode_action,
    encode_observation,
    encode_opponent_action,
)


class TestEncodeObservation:
    def test_shape(self):
        obs = encode_observation(
            hero_cards=(48, 50),
            board=[0, 5, 10],
            pot=3.0,
            stack=99.0,
            street=1,
            position=0,
            bet_facing=1.0,
            num_actions_street=2,
            stack_size=100.0,
        )
        assert obs.shape == (30,)
        assert obs.dtype == np.float32

    def test_values_bounded(self):
        obs = encode_observation(
            hero_cards=(0, 4),
            board=[8, 12, 16, 20, 24],
            pot=50.0,
            stack=50.0,
            street=3,
            position=1,
            bet_facing=25.0,
            num_actions_street=4,
            stack_size=100.0,
        )
        # Most features should be in [0, 1]
        assert (obs >= -0.1).all()
        assert (obs <= 2.1).all()

    def test_empty_board(self):
        obs = encode_observation(
            hero_cards=(48, 50),
            board=[],
            pot=1.5,
            stack=99.5,
            street=0,
            position=0,
            bet_facing=0.0,
            num_actions_street=0,
            stack_size=100.0,
        )
        assert obs.shape == (30,)


class TestEncodeAction:
    def test_shape(self):
        act = encode_action(action_type=3, bet_ratio=0.5, tta=2.0, is_allin=False)
        assert act.shape == (4,)

    def test_values(self):
        act = encode_action(action_type=3, bet_ratio=0.75, tta=0.0, is_allin=True)
        assert act[0] == 1.0  # 3/3
        assert act[1] == 0.75
        assert act[3] == 1.0  # is_allin


class TestEncodeOpponentAction:
    def test_shape(self):
        opp = encode_opponent_action(
            action_type=3, bet_ratio=0.5, tta=2.0,
            sizing_pattern=2, street=1, position=1,
        )
        assert opp.shape == (6,)


class TestClassifyOpponentType:
    def test_aggressive(self):
        actions = [
            {"type": "raise", "tta": 1.0},
            {"type": "raise", "tta": 0.5},
            {"type": "raise", "tta": 1.5},
            {"type": "call", "tta": 2.0},
        ]
        opp_type = classify_opponent_type(actions)
        # Mostly raises + fast TTA → aggressive
        assert opp_type in [1, 3]  # tight_aggressive or loose_aggressive

    def test_passive(self):
        actions = [
            {"type": "call", "tta": 5.0},
            {"type": "check", "tta": 4.0},
            {"type": "call", "tta": 6.0},
            {"type": "check", "tta": 5.0},
        ]
        opp_type = classify_opponent_type(actions)
        assert opp_type in [0, 2]  # tight_passive or loose_passive

    def test_empty(self):
        assert classify_opponent_type([]) == 0  # Default tight_passive
