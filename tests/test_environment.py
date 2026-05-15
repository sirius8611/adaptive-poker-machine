import random

from poker import actions as A
from poker.environment import BIG_BLIND, DEFAULT_STACK, HUNLEnvironment, SMALL_BLIND
from poker.state import FLOP, PREFLOP, RIVER, TURN


class TestNewInitialState:
    def setup_method(self):
        self.env = HUNLEnvironment()

    def test_blinds_posted(self):
        state = self.env.new_initial_state(random.Random(42))
        assert state.pot == SMALL_BLIND + BIG_BLIND
        assert state.stacks[0] == DEFAULT_STACK - SMALL_BLIND
        assert state.stacks[1] == DEFAULT_STACK - BIG_BLIND

    def test_sb_acts_first_preflop(self):
        state = self.env.new_initial_state(random.Random(42))
        assert state.current_player == 0
        assert state.street == PREFLOP

    def test_bet_to_call_preflop(self):
        state = self.env.new_initial_state(random.Random(42))
        # SB needs to call 50 more to match BB
        assert state.bet_to_call == BIG_BLIND - SMALL_BLIND

    def test_hole_cards_dealt(self):
        state = self.env.new_initial_state(random.Random(42))
        assert len(state.hole_cards[0]) == 2
        assert len(state.hole_cards[1]) == 2
        # No overlap
        all_cards = set(state.hole_cards[0]) | set(state.hole_cards[1])
        assert len(all_cards) == 4

    def test_determinism(self):
        s1 = self.env.new_initial_state(random.Random(42))
        s2 = self.env.new_initial_state(random.Random(42))
        assert s1 == s2


class TestApplyAction:
    def setup_method(self):
        self.env = HUNLEnvironment()

    def test_fold_is_terminal(self):
        state = self.env.new_initial_state(random.Random(42))
        state = self.env.apply_action(state, A.FOLD)
        assert self.env.is_terminal(state)
        assert state.is_folded
        assert state.folding_player == 0

    def test_call_preflop(self):
        state = self.env.new_initial_state(random.Random(42))
        # SB calls (puts in 50 more)
        state = self.env.apply_action(state, A.CALL)
        assert state.current_player == 1  # Now BB's turn
        assert state.pot == SMALL_BLIND + BIG_BLIND + (BIG_BLIND - SMALL_BLIND)

    def test_limp_check_goes_to_flop(self):
        state = self.env.new_initial_state(random.Random(42))
        # SB calls (limps)
        state = self.env.apply_action(state, A.CALL)
        # BB checks
        state = self.env.apply_action(state, A.CHECK)
        assert state.street == FLOP
        assert len(state.board) == 3
        assert state.current_player == 1  # BB acts first postflop

    def test_check_check_flop_goes_to_turn(self):
        state = self.env.new_initial_state(random.Random(42))
        state = self.env.apply_action(state, A.CALL)  # SB limps
        state = self.env.apply_action(state, A.CHECK)  # BB checks -> flop
        state = self.env.apply_action(state, A.CHECK)  # BB checks
        state = self.env.apply_action(state, A.CHECK)  # SB checks -> turn
        assert state.street == TURN
        assert len(state.board) == 4

    def test_full_check_down_to_showdown(self):
        state = self.env.new_initial_state(random.Random(42))
        state = self.env.apply_action(state, A.CALL)  # SB limps
        state = self.env.apply_action(state, A.CHECK)  # BB checks -> flop
        # Flop
        state = self.env.apply_action(state, A.CHECK)  # BB
        state = self.env.apply_action(state, A.CHECK)  # SB -> turn
        # Turn
        state = self.env.apply_action(state, A.CHECK)  # BB
        state = self.env.apply_action(state, A.CHECK)  # SB -> river
        assert state.street == RIVER
        assert len(state.board) == 5
        # River
        state = self.env.apply_action(state, A.CHECK)  # BB
        state = self.env.apply_action(state, A.CHECK)  # SB -> showdown
        assert self.env.is_terminal(state)
        assert state.is_showdown

    def test_raise_and_call(self):
        state = self.env.new_initial_state(random.Random(42))
        # SB raises pot
        legal = self.env.get_legal_actions(state)
        assert A.RAISE_100 in legal
        state = self.env.apply_action(state, A.RAISE_100)
        assert state.current_player == 1  # BB to act
        assert state.bet_to_call > 0
        # BB calls
        state = self.env.apply_action(state, A.CALL)
        assert state.street == FLOP

    def test_illegal_action_raises(self):
        state = self.env.new_initial_state(random.Random(42))
        try:
            # CHECK is illegal when facing a bet
            self.env.apply_action(state, A.CHECK)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestRewards:
    def setup_method(self):
        self.env = HUNLEnvironment()

    def test_fold_rewards_zero_sum(self):
        state = self.env.new_initial_state(random.Random(42))
        state = self.env.apply_action(state, A.FOLD)
        r0, r1 = self.env.get_rewards(state)
        assert r0 + r1 == 0

    def test_sb_fold_loses_sb(self):
        state = self.env.new_initial_state(random.Random(42))
        state = self.env.apply_action(state, A.FOLD)
        r0, r1 = self.env.get_rewards(state)
        assert r0 == -SMALL_BLIND
        assert r1 == SMALL_BLIND

    def test_showdown_zero_sum(self):
        state = self.env.new_initial_state(random.Random(42))
        state = self.env.apply_action(state, A.CALL)
        state = self.env.apply_action(state, A.CHECK)
        for _ in range(3):  # flop, turn, river
            state = self.env.apply_action(state, A.CHECK)
            state = self.env.apply_action(state, A.CHECK)
        r0, r1 = self.env.get_rewards(state)
        assert r0 + r1 == 0

    def test_pot_conservation(self):
        """Stacks + pot should always equal 2 * initial stack."""
        env = self.env
        rng = random.Random(42)
        for _ in range(100):
            state = env.new_initial_state(rng)
            while not env.is_terminal(state):
                legal = env.get_legal_actions(state)
                action = rng.choice(legal)
                state = env.apply_action(state, action)
                assert state.stacks[0] + state.stacks[1] + state.pot == 2 * DEFAULT_STACK, (
                    f"Conservation violated: stacks={state.stacks}, pot={state.pot}"
                )


class TestDeterminism:
    def test_same_seed_same_result(self):
        env = HUNLEnvironment()
        actions_seq = [A.CALL, A.CHECK, A.CHECK, A.CHECK]

        s1 = env.new_initial_state(random.Random(99))
        for a in actions_seq:
            s1 = env.apply_action(s1, a)

        s2 = env.new_initial_state(random.Random(99))
        for a in actions_seq:
            s2 = env.apply_action(s2, a)

        assert s1 == s2


class TestLegalActions:
    def setup_method(self):
        self.env = HUNLEnvironment()

    def test_preflop_sb_can_fold_call_raise(self):
        state = self.env.new_initial_state(random.Random(42))
        legal = self.env.get_legal_actions(state)
        assert A.FOLD in legal
        assert A.CALL in legal
        assert A.CHECK not in legal
        assert A.ALL_IN in legal

    def test_no_actions_at_terminal(self):
        state = self.env.new_initial_state(random.Random(42))
        state = self.env.apply_action(state, A.FOLD)
        assert self.env.get_legal_actions(state) == []

    def test_bb_can_check_after_limp(self):
        state = self.env.new_initial_state(random.Random(42))
        state = self.env.apply_action(state, A.CALL)  # SB limps
        legal = self.env.get_legal_actions(state)
        assert A.CHECK in legal
        assert A.FOLD not in legal
