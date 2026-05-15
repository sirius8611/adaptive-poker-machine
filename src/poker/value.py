"""Heuristic value function for depth-limited ISMCTS.

Estimates win probability for a player given a fully-specified state
(both hands known, as in a determinized/sampled world).
"""

from __future__ import annotations

import random

from poker.cards import Board, Card, Hand, NUM_CARDS
from poker.evaluator import compare_hands
from poker.state import HUNLState, RIVER


def hand_equity_exact(hand: Hand, opponent_hand: Hand, board: Board) -> float:
    """Exact equity when board has 5 cards. Returns 1.0/0.5/0.0."""
    result = compare_hands(hand, opponent_hand, board)
    if result == 1:
        return 1.0
    elif result == -1:
        return 0.0
    return 0.5


def hand_equity_monte_carlo(
    hand: Hand,
    opponent_hand: Hand,
    board: Board,
    rng: random.Random,
    num_samples: int = 200,
) -> float:
    """Monte Carlo equity estimation by sampling board completions.

    Both hands are known (we're in a sampled world). We sample the
    remaining community cards to estimate equity.
    """
    if len(board) == 5:
        return hand_equity_exact(hand, opponent_hand, board)

    known_cards = set(hand) | set(opponent_hand) | set(board)
    available = [c for c in range(NUM_CARDS) if c not in known_cards]
    cards_needed = 5 - len(board)

    wins = 0.0
    for _ in range(num_samples):
        rng.shuffle(available)
        full_board = board + tuple(available[:cards_needed])
        result = compare_hands(hand, opponent_hand, full_board)
        if result == 1:
            wins += 1.0
        elif result == 0:
            wins += 0.5

    return wins / num_samples


def value_function(
    state: HUNLState,
    player_id: int,
    rng: random.Random | None = None,
    num_samples: int = 200,
) -> float:
    """Estimate win probability for player_id in [0.0, 1.0].

    In a determinized world (both hands known), this computes equity
    and adjusts for pot odds to produce an EV-based value estimate.

    Args:
        state: Fully-specified game state (both hands known).
        player_id: Which player to evaluate for.
        rng: RNG for Monte Carlo sampling (required if pre-river).
        num_samples: Number of board completions to sample.

    Returns:
        Value estimate in [0.0, 1.0].
    """
    hand = state.hole_cards[player_id]
    opp_hand = state.hole_cards[1 - player_id]

    if len(state.board) == 5:
        equity = hand_equity_exact(hand, opp_hand, state.board)
    else:
        if rng is None:
            rng = random.Random()
        equity = hand_equity_monte_carlo(
            hand, opp_hand, state.board, rng, num_samples
        )

    # Blend with pot odds for EV-aware value
    pot = state.pot
    my_stack = state.stacks[player_id]
    total_invested = pot  # What's already in the pot
    # Simple EV: equity * pot normalized
    # We normalize by the maximum possible payoff to get [0, 1]
    max_payoff = pot + state.stacks[1 - player_id]
    if max_payoff > 0:
        ev = equity * pot / max_payoff
        # Blend: 70% raw equity, 30% pot-adjusted EV
        value = 0.7 * equity + 0.3 * ev
    else:
        value = equity

    return max(0.0, min(1.0, value))
