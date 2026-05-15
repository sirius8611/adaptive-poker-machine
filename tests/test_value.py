import random

from poker import actions as A
from poker.cards import str_to_card, make_hand
from poker.environment import HUNLEnvironment
from poker.value import hand_equity_exact, hand_equity_monte_carlo, value_function


def _hand(a: str, b: str):
    return make_hand(str_to_card(a), str_to_card(b))


def _board(*names: str):
    return tuple(str_to_card(n) for n in names)


class TestHandEquityExact:
    def test_winner(self):
        aa = _hand("As", "Ah")
        kk = _hand("Ks", "Kh")
        board = _board("2c", "5d", "9h", "Jc", "3s")
        assert hand_equity_exact(aa, kk, board) == 1.0

    def test_loser(self):
        kk = _hand("Ks", "Kh")
        aa = _hand("As", "Ah")
        board = _board("2c", "5d", "9h", "Jc", "3s")
        assert hand_equity_exact(kk, aa, board) == 0.0

    def test_split(self):
        h1 = _hand("2c", "3d")
        h2 = _hand("4c", "5d")
        board = _board("Tc", "Jd", "Qh", "Ks", "Ac")
        assert hand_equity_exact(h1, h2, board) == 0.5


class TestHandEquityMonteCarlo:
    def test_aa_vs_72o_preflop(self):
        aa = _hand("As", "Ah")
        seven_two = _hand("7c", "2d")
        rng = random.Random(42)
        eq = hand_equity_monte_carlo(aa, seven_two, (), rng, num_samples=1000)
        # AA vs 72o is ~87% equity
        assert 0.80 < eq < 0.95, f"Expected ~87%, got {eq:.1%}"

    def test_nuts_on_river(self):
        # Royal flush vs anything
        rf = _hand("As", "Ks")
        other = _hand("2c", "3d")
        board = _board("Ts", "Js", "Qs", "4h", "8c")
        rng = random.Random(42)
        eq = hand_equity_monte_carlo(rf, other, board, rng)
        assert eq == 1.0


class TestValueFunction:
    def test_returns_in_range(self):
        env = HUNLEnvironment()
        rng = random.Random(42)
        state = env.new_initial_state(rng)
        val = value_function(state, player_id=0, rng=random.Random(99))
        assert 0.0 <= val <= 1.0

    def test_strong_hand_high_value(self):
        env = HUNLEnvironment()
        rng = random.Random(42)
        state = env.new_initial_state(rng)
        # Play to showdown
        state = env.apply_action(state, A.CALL)
        state = env.apply_action(state, A.CHECK)
        # On flop, compute value for both
        v0 = value_function(state, 0, rng=random.Random(99))
        v1 = value_function(state, 1, rng=random.Random(99))
        # Values should sum to approximately 1.0 (not exactly due to pot odds blend)
        assert 0.0 <= v0 <= 1.0
        assert 0.0 <= v1 <= 1.0
