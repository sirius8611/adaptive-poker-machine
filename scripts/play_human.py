"""Play HUNL poker against the ISMCTS agent interactively.

Usage:
    python scripts/play_human.py [--iterations 200] [--stack 10000]

You are Player 1 (BB). The AI is Player 0 (SB/Dealer).
"""

import argparse
import random

from poker import actions as A
from poker.cards import card_to_str
from poker.environment import HUNLEnvironment
from poker.ismcts import ISMCTS
from poker.state import PREFLOP, FLOP, TURN, RIVER

STREET_NAMES = {PREFLOP: "Preflop", FLOP: "Flop", TURN: "Turn", RIVER: "River"}


def format_hand(hand):
    return f"[{card_to_str(hand[0])} {card_to_str(hand[1])}]"


def format_board(board):
    if not board:
        return "---"
    return " ".join(card_to_str(c) for c in board)


def display_state(state, player_id):
    print()
    print(f"{'='*40}")
    print(f"  Street:  {STREET_NAMES[state.street]}")
    print(f"  Board:   {format_board(state.board)}")
    print(f"  Pot:     {state.pot / 100:.1f} BB")
    print(f"  Your stack:  {state.stacks[player_id] / 100:.1f} BB")
    print(f"  AI stack:    {state.stacks[1 - player_id] / 100:.1f} BB")
    print(f"  Your hand:   {format_hand(state.hole_cards[player_id])}")
    if state.bet_to_call > 0 and state.current_player == player_id:
        print(f"  To call: {state.bet_to_call / 100:.1f} BB")
    print(f"{'='*40}")


def get_human_action(env, state):
    legal = env.get_legal_actions(state)
    print("\nYour actions:")
    for i, action in enumerate(legal):
        name = A.action_to_str(action)
        if action >= A.RAISE_33 and action <= A.RAISE_200:
            chips_after_call = state.stacks[state.current_player] - min(state.bet_to_call, state.stacks[state.current_player])
            amt = A.raise_amount(action, state.pot, state.min_raise, chips_after_call)
            print(f"  [{i}] {name} ({amt / 100:.1f} BB)")
        elif action == A.CALL:
            call_amt = min(state.bet_to_call, state.stacks[state.current_player])
            print(f"  [{i}] {name} ({call_amt / 100:.1f} BB)")
        elif action == A.ALL_IN:
            print(f"  [{i}] {name} ({state.stacks[state.current_player] / 100:.1f} BB)")
        else:
            print(f"  [{i}] {name}")

    while True:
        try:
            choice = input("\nChoose action [number]: ").strip()
            if choice.lower() == "q":
                return None
            idx = int(choice)
            if 0 <= idx < len(legal):
                return legal[idx]
            print(f"Pick 0-{len(legal)-1}")
        except (ValueError, EOFError):
            print("Enter a number or 'q' to quit")


def main():
    parser = argparse.ArgumentParser(description="Play HUNL poker vs ISMCTS")
    parser.add_argument("--iterations", type=int, default=200, help="ISMCTS iterations (higher=stronger)")
    parser.add_argument("--stack", type=int, default=10000, help="Starting stack in centiBB (default 10000 = 100BB)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    args = parser.parse_args()

    env = HUNLEnvironment(stack_size=args.stack)
    rng = random.Random(args.seed)
    human_player = 1  # You are BB
    ai_player = 0     # AI is SB/Dealer

    total_human = 0
    total_ai = 0
    hand_num = 0

    print("=" * 40)
    print("  HUNL Poker vs ISMCTS Agent")
    print(f"  AI strength: {args.iterations} iterations")
    print(f"  Stack: {args.stack / 100:.0f} BB")
    print(f"  You are BB (Player 1)")
    print(f"  Type 'q' to quit")
    print("=" * 40)

    while True:
        hand_num += 1
        print(f"\n{'#'*40}")
        print(f"  HAND #{hand_num}")
        print(f"  Score: You {total_human / 100:+.1f} BB | AI {total_ai / 100:+.1f} BB")
        print(f"{'#'*40}")

        state = env.new_initial_state(rng)
        prev_street = -1

        while not env.is_terminal(state):
            if state.street != prev_street:
                display_state(state, human_player)
                prev_street = state.street

            cp = state.current_player
            if cp == ai_player:
                # AI's turn
                obs = env.get_observation(state, ai_player)
                legal = env.get_legal_actions(state)
                searcher = ISMCTS(
                    env, player_id=ai_player,
                    num_iterations=args.iterations,
                    rng=random.Random(rng.randint(0, 2**32)),
                )
                action = searcher.search(obs, real_legal_actions=legal)
                name = A.action_to_str(action)
                if action >= A.RAISE_33 and action <= A.RAISE_200:
                    chips_after_call = state.stacks[ai_player] - min(state.bet_to_call, state.stacks[ai_player])
                    amt = A.raise_amount(action, state.pot, state.min_raise, chips_after_call)
                    print(f"\n  AI: {name} ({amt / 100:.1f} BB)")
                elif action == A.CALL:
                    amt = min(state.bet_to_call, state.stacks[ai_player])
                    print(f"\n  AI: {name} ({amt / 100:.1f} BB)")
                elif action == A.ALL_IN:
                    print(f"\n  AI: ALL IN ({state.stacks[ai_player] / 100:.1f} BB)")
                else:
                    print(f"\n  AI: {name}")
                state = env.apply_action(state, action)
            else:
                # Human's turn
                if state.street != prev_street:
                    display_state(state, human_player)
                    prev_street = state.street
                action = get_human_action(env, state)
                if action is None:
                    print(f"\nFinal score: You {total_human / 100:+.1f} BB | AI {total_ai / 100:+.1f} BB over {hand_num - 1} hands")
                    return
                state = env.apply_action(state, action)

        # Show result
        r0, r1 = env.get_rewards(state)
        human_reward = r1  # human is player 1
        ai_reward = r0
        total_human += human_reward
        total_ai += ai_reward

        print(f"\n{'~'*40}")
        if state.is_showdown:
            print(f"  SHOWDOWN!")
            print(f"  Your hand:  {format_hand(state.hole_cards[human_player])}")
            print(f"  AI hand:    {format_hand(state.hole_cards[ai_player])}")
            print(f"  Board:      {format_board(state.board)}")
        elif state.is_folded:
            who = "AI" if state.folding_player == ai_player else "You"
            print(f"  {who} folded.")

        if human_reward > 0:
            print(f"  You WIN +{human_reward / 100:.1f} BB!")
        elif human_reward < 0:
            print(f"  You LOSE {human_reward / 100:.1f} BB")
        else:
            print(f"  Split pot!")
        print(f"{'~'*40}")

        input("\nPress Enter for next hand...")


if __name__ == "__main__":
    main()
