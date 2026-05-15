import random

from poker.cards import (
    NUM_CARDS,
    Card,
    Deck,
    card_to_str,
    make_hand,
    str_to_card,
)


class TestCardEncoding:
    def test_roundtrip(self):
        for card in range(NUM_CARDS):
            s = card_to_str(card)
            assert str_to_card(s) == card

    def test_known_cards(self):
        assert card_to_str(0) == "2c"
        assert card_to_str(3) == "2s"
        assert card_to_str(48) == "Ac"
        assert card_to_str(51) == "As"
        assert str_to_card("Ah") == 50
        assert str_to_card("Td") == 33

    def test_all_unique(self):
        strs = [card_to_str(c) for c in range(NUM_CARDS)]
        assert len(set(strs)) == NUM_CARDS


class TestMakeHand:
    def test_sorted(self):
        h = make_hand(51, 0)
        assert h == (0, 51)

    def test_already_sorted(self):
        h = make_hand(10, 20)
        assert h == (10, 20)


class TestDeck:
    def test_deal_52(self):
        deck = Deck()
        deck.shuffle(random.Random(42))
        cards = deck.deal(52)
        assert len(cards) == 52
        assert len(set(cards)) == 52

    def test_deal_too_many_raises(self):
        deck = Deck()
        deck.shuffle(random.Random(42))
        deck.deal(50)
        try:
            deck.deal(5)
            assert False, "Should have raised"
        except ValueError:
            pass

    def test_deal_excluding(self):
        deck = Deck()
        excluded = {0, 1, 2, 3}
        rng = random.Random(42)
        dealt = deck.deal_excluding(excluded, 5, rng)
        assert len(dealt) == 5
        for c in dealt:
            assert c not in excluded

    def test_shuffle_deterministic(self):
        d1 = Deck()
        d1.shuffle(random.Random(123))
        c1 = d1.deal(5)

        d2 = Deck()
        d2.shuffle(random.Random(123))
        c2 = d2.deal(5)

        assert c1 == c2

    def test_remaining(self):
        deck = Deck()
        deck.shuffle(random.Random(42))
        deck.deal(10)
        rem = deck.remaining()
        assert len(rem) == 42
