from poker.actions import (
    ALL_IN,
    CALL,
    CHECK,
    FOLD,
    NUM_ACTIONS,
    RAISE_100,
    RAISE_33,
    action_to_str,
    raise_amount,
)


class TestActionStr:
    def test_all_actions_have_names(self):
        for a in range(NUM_ACTIONS):
            name = action_to_str(a)
            assert isinstance(name, str)
            assert len(name) > 0

    def test_known(self):
        assert action_to_str(FOLD) == "fold"
        assert action_to_str(ALL_IN) == "all_in"


class TestRaiseAmount:
    def test_pot_sized_raise(self):
        # pot=1000, raise_100 = 1.0 * pot = 1000
        amount = raise_amount(RAISE_100, pot=1000, min_raise=200, stack=5000)
        assert amount == 1000

    def test_clamp_to_min_raise(self):
        # pot=100, raise_33 = 0.33 * 100 = 33, but min_raise=200
        amount = raise_amount(RAISE_33, pot=100, min_raise=200, stack=5000)
        assert amount == 200

    def test_clamp_to_stack(self):
        # pot=10000, raise_100 = 10000, but stack=3000
        amount = raise_amount(RAISE_100, pot=10000, min_raise=200, stack=3000)
        assert amount == 3000

    def test_invalid_action_raises(self):
        try:
            raise_amount(FOLD, pot=100, min_raise=100, stack=1000)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
