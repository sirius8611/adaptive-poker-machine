from poker.cards import str_to_card
from poker.evaluator import compare_hands, evaluate_hand


def _cards(*names: str) -> tuple[int, ...]:
    return tuple(str_to_card(n) for n in names)


def _hand(a: str, b: str) -> tuple[int, int]:
    return (str_to_card(a), str_to_card(b))


class TestEvaluateHand:
    def test_royal_flush_beats_straight_flush(self):
        royal = _cards("Ts", "Js", "Qs", "Ks", "As")
        straight_flush = _cards("9s", "Ts", "Js", "Qs", "Ks")
        assert evaluate_hand(royal) < evaluate_hand(straight_flush)

    def test_four_of_a_kind_beats_full_house(self):
        quads = _cards("As", "Ah", "Ad", "Ac", "Ks")
        full_house = _cards("As", "Ah", "Ad", "Ks", "Kh")
        assert evaluate_hand(quads) < evaluate_hand(full_house)

    def test_flush_beats_straight(self):
        flush = _cards("2s", "5s", "7s", "9s", "Js")
        straight = _cards("5c", "6d", "7h", "8s", "9c")
        assert evaluate_hand(flush) < evaluate_hand(straight)

    def test_seven_card_evaluation(self):
        # AA vs KK on a dry board — AA should win
        cards_aa = _cards("As", "Ah", "2c", "5d", "9h", "Jc", "3s")
        cards_kk = _cards("Ks", "Kh", "2c", "5d", "9h", "Jc", "3s")
        assert evaluate_hand(cards_aa) < evaluate_hand(cards_kk)


class TestCompareHands:
    def test_aa_vs_kk(self):
        board = _cards("2c", "5d", "9h", "Jc", "3s")
        aa = _hand("As", "Ah")
        kk = _hand("Ks", "Kh")
        assert compare_hands(aa, kk, board) == 1

    def test_kk_loses_to_aa(self):
        board = _cards("2c", "5d", "9h", "Jc", "3s")
        aa = _hand("As", "Ah")
        kk = _hand("Ks", "Kh")
        assert compare_hands(kk, aa, board) == -1

    def test_split_pot(self):
        # Same straight on board, no improvement
        board = _cards("Tc", "Jd", "Qh", "Ks", "Ac")
        h1 = _hand("2c", "3d")
        h2 = _hand("4c", "5d")
        assert compare_hands(h1, h2, board) == 0

    def test_flush_over_straight(self):
        board = _cards("2s", "5s", "8d", "9c", "Ts")
        flush_hand = _hand("Js", "3s")  # spade flush
        straight_hand = _hand("7c", "6h")  # 6-T straight
        assert compare_hands(flush_hand, straight_hand, board) == 1
