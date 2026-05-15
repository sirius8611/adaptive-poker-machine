"""Hand evaluation adapter over the treys library.

Converts our card encoding (0-51, rank*4+suit) to treys format and back.
Lower evaluation rank = stronger hand.
"""

from __future__ import annotations

from treys import Card as TreysCard
from treys import Evaluator as TreysEvaluator

from poker.cards import RANK_CHARS, SUIT_CHARS, Board, Card, Hand

_evaluator = TreysEvaluator()

# treys uses rank chars: 2-9, T, J, Q, K, A  and suit chars: s, h, d, c
_TREYS_SUIT_MAP = {0: "c", 1: "d", 2: "h", 3: "s"}


def _to_treys(card: Card) -> int:
    rank = card // 4
    suit = card % 4
    rank_char = RANK_CHARS[rank]
    suit_char = _TREYS_SUIT_MAP[suit]
    return TreysCard.new(rank_char + suit_char)


def evaluate_hand(cards: tuple[Card, ...]) -> int:
    """Evaluate a 5-7 card poker hand.

    Returns an integer rank where LOWER is BETTER.
    Range: 1 (Royal Flush) to 7462 (worst high card).
    """
    if len(cards) == 5:
        treys_cards = [_to_treys(c) for c in cards]
        return _evaluator.evaluate([], treys_cards)
    elif len(cards) in (6, 7):
        # treys expects (board, hand) split for 5+ cards
        # We pass first 5 as board placeholder and rest as hand,
        # but treys.evaluate actually wants board (3-5) + hand (2)
        # For 7 cards: board=5, hand=2
        # For 6 cards: board=4, hand=2
        treys_cards = [_to_treys(c) for c in cards]
        hand_size = 2
        board = treys_cards[: len(treys_cards) - hand_size]
        hand = treys_cards[len(treys_cards) - hand_size :]
        return _evaluator.evaluate(board, hand)
    else:
        raise ValueError(f"Need 5-7 cards, got {len(cards)}")


def compare_hands(hand_a: Hand, hand_b: Hand, board: Board) -> int:
    """Compare two hole-card hands given a board (3-5 community cards).

    Returns:
        +1 if hand_a wins
        -1 if hand_b wins
         0 if tie
    """
    cards_a = board + hand_a
    cards_b = board + hand_b
    rank_a = evaluate_hand(cards_a)
    rank_b = evaluate_hand(cards_b)
    # Lower rank = better hand
    if rank_a < rank_b:
        return 1
    elif rank_a > rank_b:
        return -1
    return 0
