"""Game state representations for HUNL No-Limit Texas Hold'em.

All chip amounts are in centiBB (1 BB = 100 units).
"""

from __future__ import annotations

from dataclasses import dataclass

from poker.cards import Board, Card, Hand

# Street constants
PREFLOP = 0
FLOP = 1
TURN = 2
RIVER = 3

# Number of community cards dealt per street transition
COMMUNITY_CARDS_PER_STREET = {PREFLOP: 0, FLOP: 3, TURN: 1, RIVER: 1}
TOTAL_COMMUNITY_BY_STREET = {PREFLOP: 0, FLOP: 3, TURN: 4, RIVER: 5}


@dataclass(frozen=True)
class HUNLState:
    # Private info
    hole_cards: tuple[Hand, Hand]  # (player_0_hand, player_1_hand)
    deck_remaining: tuple[Card, ...]  # Pre-shuffled remaining deck for deterministic dealing

    # Public info
    board: Board  # Community cards dealt so far
    pot: int  # Total chips in pot
    stacks: tuple[int, int]  # Remaining stacks
    street: int  # PREFLOP=0, FLOP=1, TURN=2, RIVER=3
    current_player: int  # 0 or 1
    bet_to_call: int  # Amount current player must call (0 if checked to)
    min_raise: int  # Minimum raise size
    last_raise: int  # Size of the last raise (for min-raise calc)
    num_actions_this_street: int  # How many actions taken this street
    action_history: tuple[tuple[int, ...], ...]  # Per-street action sequences

    # Terminal
    is_folded: bool
    folding_player: int  # -1 if nobody folded
    is_showdown: bool  # True if we reached showdown


@dataclass(frozen=True)
class PublicState:
    board: Board
    pot: int
    stacks: tuple[int, int]
    street: int
    current_player: int
    bet_to_call: int
    action_history: tuple[tuple[int, ...], ...]
    is_folded: bool
    is_showdown: bool


@dataclass(frozen=True)
class Observation:
    my_hand: Hand
    public: PublicState
    player_id: int
