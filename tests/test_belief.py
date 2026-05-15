import random

from poker import actions as A
from poker.belief import resample_history
from poker.environment import BIG_BLIND, DEFAULT_STACK, HUNLEnvironment, SMALL_BLIND


class TestResampleHistory:
    def setup_method(self):
        self.env = HUNLEnvironment()

    def test_player_hand_preserved(self):
        state = self.env.new_initial_state(random.Random(42))
        obs = self.env.get_observation(state, player_id=0)
        resampled = resample_history(
            obs, DEFAULT_STACK, SMALL_BLIND, BIG_BLIND,
            rng=random.Random(99), original_state=state,
        )
        assert resampled.hole_cards[0] == obs.my_hand

    def test_board_preserved(self):
        state = self.env.new_initial_state(random.Random(42))
        state = self.env.apply_action(state, A.CALL)
        state = self.env.apply_action(state, A.CHECK)  # -> flop
        obs = self.env.get_observation(state, player_id=0)
        resampled = resample_history(
            obs, DEFAULT_STACK, SMALL_BLIND, BIG_BLIND,
            rng=random.Random(99), original_state=state,
        )
        assert resampled.board == state.board

    def test_opponent_hand_differs(self):
        state = self.env.new_initial_state(random.Random(42))
        obs = self.env.get_observation(state, player_id=0)
        # Over many resamples, opponent hand should vary
        opponent_hands = set()
        for seed in range(100):
            resampled = resample_history(
                obs, DEFAULT_STACK, SMALL_BLIND, BIG_BLIND,
                rng=random.Random(seed), original_state=state,
            )
            opponent_hands.add(resampled.hole_cards[1])
        # Should see many different opponent hands
        assert len(opponent_hands) > 20

    def test_no_card_overlap(self):
        state = self.env.new_initial_state(random.Random(42))
        state = self.env.apply_action(state, A.CALL)
        state = self.env.apply_action(state, A.CHECK)  # -> flop
        obs = self.env.get_observation(state, player_id=0)
        for seed in range(50):
            resampled = resample_history(
                obs, DEFAULT_STACK, SMALL_BLIND, BIG_BLIND,
                rng=random.Random(seed), original_state=state,
            )
            all_known = (
                set(resampled.hole_cards[0])
                | set(resampled.hole_cards[1])
                | set(resampled.board)
            )
            # No duplicates
            total = (
                len(resampled.hole_cards[0])
                + len(resampled.hole_cards[1])
                + len(resampled.board)
            )
            assert len(all_known) == total

    def test_action_history_preserved(self):
        state = self.env.new_initial_state(random.Random(42))
        state = self.env.apply_action(state, A.CALL)
        obs = self.env.get_observation(state, player_id=1)
        resampled = resample_history(
            obs, DEFAULT_STACK, SMALL_BLIND, BIG_BLIND,
            rng=random.Random(99), original_state=state,
        )
        assert resampled.action_history == state.action_history

    def test_public_state_matches(self):
        state = self.env.new_initial_state(random.Random(42))
        state = self.env.apply_action(state, A.CALL)
        state = self.env.apply_action(state, A.CHECK)  # -> flop
        obs = self.env.get_observation(state, player_id=1)
        resampled = resample_history(
            obs, DEFAULT_STACK, SMALL_BLIND, BIG_BLIND,
            rng=random.Random(99), original_state=state,
        )
        assert resampled.pot == state.pot
        assert resampled.stacks == state.stacks
        assert resampled.street == state.street
        assert resampled.current_player == state.current_player
