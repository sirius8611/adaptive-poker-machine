"""Belief state and history resampling for ISMCTS.

The resample_history function takes a player's observation and generates
a complete HUNLState consistent with what the player can see. This is the
"determinization" step in Information Set MCTS.
"""

from __future__ import annotations

import random

from poker.cards import NUM_CARDS, Card, make_hand
from poker.state import TOTAL_COMMUNITY_BY_STREET, HUNLState, Observation


def resample_history(
    observation: Observation,
    env_stack_size: int,
    env_small_blind: int,
    env_big_blind: int,
    rng: random.Random,
    original_state: HUNLState | None = None,
) -> HUNLState:
    """Sample a complete HUNLState consistent with a player's observation.

    This generates a "possible world" where:
    - The player's own hole cards match exactly
    - The board cards match exactly
    - The opponent's hole cards are sampled uniformly from remaining cards
    - The remaining deck is shuffled consistently

    Args:
        observation: What the player can see.
        env_stack_size: Starting stack size.
        env_small_blind: Small blind amount.
        env_big_blind: Big blind amount.
        rng: Random number generator for sampling.
        original_state: If provided, copies non-resampled fields directly.

    Returns:
        A fully-specified HUNLState consistent with the observation.
    """
    player_id = observation.player_id
    pub = observation.public
    my_hand = observation.my_hand

    # Cards we know about
    known_cards: set[Card] = set(my_hand) | set(pub.board)

    # Sample opponent's hole cards from remaining deck
    available = [c for c in range(NUM_CARDS) if c not in known_cards]
    rng.shuffle(available)
    opp_hand = make_hand(available[0], available[1])
    remaining_available = available[2:]

    # Figure out how many more community cards need to come
    total_community_needed = 5  # Always need 5 total for potential showdown
    already_dealt = len(pub.board)
    future_community = total_community_needed - already_dealt

    # The deck_remaining must produce the right community cards when streets advance.
    # Since we're reconstructing a state mid-hand, the deck_remaining should contain
    # exactly the cards that would be dealt for future streets.
    # We shuffle the remaining cards and use them as deck_remaining.
    rng.shuffle(remaining_available)
    deck_remaining = tuple(remaining_available)

    # Build hole_cards tuple in correct player order
    if player_id == 0:
        hole_cards = (my_hand, opp_hand)
    else:
        hole_cards = (opp_hand, my_hand)

    if original_state is not None:
        # Copy structure from original state, just replace hidden info
        return HUNLState(
            hole_cards=hole_cards,
            deck_remaining=deck_remaining,
            board=original_state.board,
            pot=original_state.pot,
            stacks=original_state.stacks,
            street=original_state.street,
            current_player=original_state.current_player,
            bet_to_call=original_state.bet_to_call,
            min_raise=original_state.min_raise,
            last_raise=original_state.last_raise,
            num_actions_this_street=original_state.num_actions_this_street,
            action_history=original_state.action_history,
            is_folded=original_state.is_folded,
            folding_player=original_state.folding_player,
            is_showdown=original_state.is_showdown,
        )

    return HUNLState(
        hole_cards=hole_cards,
        deck_remaining=deck_remaining,
        board=pub.board,
        pot=pub.pot,
        stacks=pub.stacks,
        street=pub.street,
        current_player=pub.current_player,
        bet_to_call=pub.bet_to_call,
        min_raise=env_big_blind,  # Conservative default
        last_raise=env_big_blind,
        num_actions_this_street=sum(len(s) for s in pub.action_history),
        action_history=pub.action_history,
        is_folded=pub.is_folded,
        folding_player=-1,
        is_showdown=pub.is_showdown,
    )
