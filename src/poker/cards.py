"""Card encoding and Deck class for HUNL poker.

Card encoding: card_id = rank * 4 + suit
  rank: 0=2, 1=3, ..., 8=T, 9=J, 10=Q, 11=K, 12=A
  suit: 0=clubs, 1=diamonds, 2=hearts, 3=spades
"""

from __future__ import annotations

import random
from typing import TypeAlias

Card: TypeAlias = int
Hand: TypeAlias = tuple[Card, Card]
Board: TypeAlias = tuple[Card, ...]

NUM_CARDS = 52
NUM_RANKS = 13
NUM_SUITS = 4

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"


def card_to_str(card: Card) -> str:
    rank = card // 4
    suit = card % 4
    return RANK_CHARS[rank] + SUIT_CHARS[suit]


def str_to_card(s: str) -> Card:
    if len(s) != 2:
        raise ValueError(f"Invalid card string: {s!r}")
    rank = RANK_CHARS.index(s[0])
    suit = SUIT_CHARS.index(s[1])
    return rank * 4 + suit


def make_hand(c1: Card, c2: Card) -> Hand:
    return (min(c1, c2), max(c1, c2))


class Deck:
    def __init__(self) -> None:
        self._cards: list[Card] = list(range(NUM_CARDS))
        self._pos: int = 0

    def shuffle(self, rng: random.Random) -> None:
        rng.shuffle(self._cards)
        self._pos = 0

    def deal(self, n: int) -> list[Card]:
        if self._pos + n > len(self._cards):
            raise ValueError(f"Cannot deal {n} cards, only {len(self._cards) - self._pos} remain")
        dealt = self._cards[self._pos : self._pos + n]
        self._pos += n
        return dealt

    def deal_excluding(self, excluded: set[Card], n: int, rng: random.Random) -> list[Card]:
        available = [c for c in range(NUM_CARDS) if c not in excluded]
        rng.shuffle(available)
        if n > len(available):
            raise ValueError(f"Cannot deal {n} cards, only {len(available)} available")
        return available[:n]

    def remaining(self) -> list[Card]:
        return self._cards[self._pos :]

    @property
    def position(self) -> int:
        return self._pos
