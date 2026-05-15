"""HUNL No-Limit Texas Hold'em environment with OpenSpiel-style API.

The environment provides deterministic state transitions. All randomness
is resolved at deal time (deck shuffle), making apply_action a pure function.

HUNL position rules:
  - Player 0 = Dealer/Button/SB
  - Player 1 = BB
  - Preflop: Player 0 (SB) acts first
  - Postflop: Player 1 (BB) acts first
"""

from __future__ import annotations

import random
from dataclasses import replace

from poker import actions as A
from poker.cards import Board, Card, Deck, Hand, make_hand
from poker.evaluator import compare_hands
from poker.state import (
    FLOP,
    PREFLOP,
    RIVER,
    TOTAL_COMMUNITY_BY_STREET,
    TURN,
    HUNLState,
    Observation,
    PublicState,
)

DEFAULT_STACK = 10_000  # 100 BB in centiBB
SMALL_BLIND = 50
BIG_BLIND = 100


class HUNLEnvironment:
    def __init__(
        self,
        stack_size: int = DEFAULT_STACK,
        small_blind: int = SMALL_BLIND,
        big_blind: int = BIG_BLIND,
    ) -> None:
        self.stack_size = stack_size
        self.small_blind = small_blind
        self.big_blind = big_blind

    def new_initial_state(self, rng: random.Random) -> HUNLState:
        deck = Deck()
        deck.shuffle(rng)

        p0_cards = deck.deal(2)
        p1_cards = deck.deal(2)
        remaining = tuple(deck.remaining())

        return HUNLState(
            hole_cards=(make_hand(*p0_cards), make_hand(*p1_cards)),
            deck_remaining=remaining,
            board=(),
            pot=self.small_blind + self.big_blind,
            stacks=(
                self.stack_size - self.small_blind,
                self.stack_size - self.big_blind,
            ),
            street=PREFLOP,
            current_player=0,  # SB acts first preflop in HUNL
            bet_to_call=self.big_blind - self.small_blind,  # SB must call 50 more
            min_raise=self.big_blind,  # Min raise = 1 BB
            last_raise=self.big_blind,
            num_actions_this_street=0,
            action_history=((),),
            is_folded=False,
            folding_player=-1,
            is_showdown=False,
        )

    def apply_action(self, state: HUNLState, action: int) -> HUNLState:
        if self.is_terminal(state):
            raise ValueError("Cannot apply action to terminal state")

        legal = self.get_legal_actions(state)
        if action not in legal:
            raise ValueError(
                f"Illegal action {A.action_to_str(action)} "
                f"(legal: {[A.action_to_str(a) for a in legal]})"
            )

        cp = state.current_player
        opp = 1 - cp
        stacks = list(state.stacks)
        pot = state.pot
        bet_to_call = state.bet_to_call
        min_raise = state.min_raise
        last_raise = state.last_raise
        is_folded = False
        folding_player = -1

        # Append action to current street history
        street_history = list(state.action_history)
        current_street_actions = street_history[-1] + (action,)
        street_history[-1] = current_street_actions
        num_actions = state.num_actions_this_street + 1

        if action == A.FOLD:
            is_folded = True
            folding_player = cp
            return replace(
                state,
                stacks=tuple(stacks),
                pot=pot,
                is_folded=True,
                folding_player=cp,
                action_history=tuple(street_history),
                num_actions_this_street=num_actions,
            )

        elif action == A.CHECK:
            # Check — no chips move
            pass

        elif action == A.CALL:
            call_amount = min(bet_to_call, stacks[cp])
            stacks[cp] -= call_amount
            pot += call_amount
            bet_to_call = 0

        elif action == A.ALL_IN:
            shove_amount = stacks[cp]
            net_raise = shove_amount - bet_to_call
            if net_raise > last_raise:
                last_raise = net_raise
                min_raise = net_raise
            pot += shove_amount
            stacks[cp] = 0
            bet_to_call = max(0, shove_amount - (state.stacks[cp] - stacks[cp] + state.bet_to_call))
            # Recalculate: opponent now needs to call the excess
            # The shove puts shove_amount in. If there was a bet_to_call,
            # the net new bet is shove_amount - bet_to_call
            bet_to_call = max(0, shove_amount - state.bet_to_call)

        else:
            # Raise action (RAISE_33 .. RAISE_200)
            amount = A.raise_amount(action, pot, min_raise, stacks[cp])
            total_put_in = state.bet_to_call + amount
            if total_put_in >= stacks[cp]:
                # This becomes an all-in
                total_put_in = stacks[cp]
            stacks[cp] -= total_put_in
            pot += total_put_in
            last_raise = amount
            min_raise = amount
            bet_to_call = amount

        # Check if street/hand is over
        street = state.street
        board = state.board
        deck_remaining = state.deck_remaining
        is_showdown = False

        # Determine if the betting round is over
        street_over = self._is_street_over(action, num_actions, bet_to_call, state)

        if street_over and not is_folded:
            # Check if anyone is all-in
            anyone_allin = stacks[0] == 0 or stacks[1] == 0

            if anyone_allin or street == RIVER:
                # Deal remaining community cards and go to showdown
                board, deck_remaining = self._deal_to_river(board, deck_remaining)
                is_showdown = True
                return replace(
                    state,
                    board=board,
                    deck_remaining=deck_remaining,
                    pot=pot,
                    stacks=tuple(stacks),
                    is_showdown=True,
                    bet_to_call=0,
                    min_raise=min_raise,
                    last_raise=last_raise,
                    action_history=tuple(street_history),
                    num_actions_this_street=num_actions,
                )
            else:
                # Advance to next street
                new_street = street + 1
                board, deck_remaining = self._deal_street(
                    new_street, board, deck_remaining
                )
                street_history.append(())  # New street, empty action list
                return replace(
                    state,
                    board=board,
                    deck_remaining=deck_remaining,
                    pot=pot,
                    stacks=tuple(stacks),
                    street=new_street,
                    current_player=1,  # BB acts first postflop
                    bet_to_call=0,
                    min_raise=self.big_blind,
                    last_raise=self.big_blind,
                    num_actions_this_street=0,
                    action_history=tuple(street_history),
                    is_folded=False,
                    folding_player=-1,
                    is_showdown=False,
                )

        # Switch to other player
        return replace(
            state,
            pot=pot,
            stacks=tuple(stacks),
            current_player=opp,
            bet_to_call=bet_to_call,
            min_raise=min_raise,
            last_raise=last_raise,
            num_actions_this_street=num_actions,
            action_history=tuple(street_history),
        )

    def _is_street_over(
        self, action: int, num_actions: int, bet_to_call: int, state: HUNLState
    ) -> bool:
        if action == A.FOLD:
            return True
        # Street is over when:
        # 1. Both players have acted at least once (num_actions >= 2)
        # 2. And there's nothing to call (bets are equal)
        # Special case: preflop BB can check to end the street (after SB completes)
        if num_actions >= 2 and (action == A.CHECK or action == A.CALL):
            return True
        return False

    def _deal_street(
        self, street: int, board: Board, deck_remaining: tuple[Card, ...]
    ) -> tuple[Board, tuple[Card, ...]]:
        num_needed = TOTAL_COMMUNITY_BY_STREET[street] - len(board)
        new_cards = deck_remaining[:num_needed]
        return board + tuple(new_cards), deck_remaining[num_needed:]

    def _deal_to_river(
        self, board: Board, deck_remaining: tuple[Card, ...]
    ) -> tuple[Board, tuple[Card, ...]]:
        num_needed = 5 - len(board)
        if num_needed <= 0:
            return board, deck_remaining
        new_cards = deck_remaining[:num_needed]
        return board + tuple(new_cards), deck_remaining[num_needed:]

    def get_legal_actions(self, state: HUNLState) -> list[int]:
        if self.is_terminal(state):
            return []

        cp = state.current_player
        stack = state.stacks[cp]
        legal: list[int] = []

        if state.bet_to_call > 0:
            # Facing a bet
            legal.append(A.FOLD)
            legal.append(A.CALL)
        else:
            # No bet to face
            legal.append(A.CHECK)

        # Raises: only if player has chips beyond calling
        chips_after_call = stack - min(state.bet_to_call, stack)
        if chips_after_call > 0:
            for raise_action in range(A.RAISE_33, A.RAISE_200 + 1):
                amount = A.raise_amount(
                    raise_action, state.pot, state.min_raise, chips_after_call
                )
                if amount >= state.min_raise or amount == chips_after_call:
                    legal.append(raise_action)

            # Deduplicate: if multiple raise buckets collapse to same amount, keep first
            seen_amounts: set[int] = set()
            deduped_raises: list[int] = []
            for a in legal:
                if a >= A.RAISE_33:
                    amt = A.raise_amount(a, state.pot, state.min_raise, chips_after_call)
                    if amt not in seen_amounts:
                        seen_amounts.add(amt)
                        deduped_raises.append(a)
                else:
                    deduped_raises.append(a)
            legal = deduped_raises

            # All-in is always available if player has chips
            if stack > 0:
                legal.append(A.ALL_IN)

        # If player can only call or fold (no chips for raise), and calling = all-in,
        # fold is still legal (already added above)

        return legal

    def get_rewards(self, state: HUNLState) -> tuple[int, int]:
        if not self.is_terminal(state):
            return (0, 0)

        if state.is_folded:
            winner = 1 - state.folding_player
            # Reward = what they win from the pot minus what they put in
            # Simpler: each player's reward = final_stack - initial_stack
            r0 = state.stacks[0] - self.stack_size
            r1 = state.stacks[1] - self.stack_size
            # The winner gets the pot
            if winner == 0:
                r0 += state.pot
            else:
                r1 += state.pot
            return (r0, r1)

        if state.is_showdown:
            result = compare_hands(
                state.hole_cards[0], state.hole_cards[1], state.board
            )
            if result == 1:
                # Player 0 wins
                r0 = state.stacks[0] + state.pot - self.stack_size
                r1 = state.stacks[1] - self.stack_size
            elif result == -1:
                # Player 1 wins
                r0 = state.stacks[0] - self.stack_size
                r1 = state.stacks[1] + state.pot - self.stack_size
            else:
                # Split pot
                half = state.pot // 2
                remainder = state.pot % 2  # odd chip goes to player 0
                r0 = state.stacks[0] + half + remainder - self.stack_size
                r1 = state.stacks[1] + half - self.stack_size
            return (r0, r1)

        return (0, 0)

    def is_terminal(self, state: HUNLState) -> bool:
        return state.is_folded or state.is_showdown

    def get_public_state(self, state: HUNLState) -> PublicState:
        return PublicState(
            board=state.board,
            pot=state.pot,
            stacks=state.stacks,
            street=state.street,
            current_player=state.current_player,
            bet_to_call=state.bet_to_call,
            action_history=state.action_history,
            is_folded=state.is_folded,
            is_showdown=state.is_showdown,
        )

    def get_observation(self, state: HUNLState, player_id: int) -> Observation:
        return Observation(
            my_hand=state.hole_cards[player_id],
            public=self.get_public_state(state),
            player_id=player_id,
        )
