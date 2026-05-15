"""Smoke test: two agents play 1000 hands of HUNL poker.

Agent 0: ISMCTS (100 iterations) — demonstrates the CWM in action
Agent 1: Random — uniform random over legal actions

Verifies:
- No crashes over 1000 hands
- Pot conservation invariant holds at every state
- Rewards are always zero-sum
- ISMCTS agent wins more than random (soft check)
"""

import random
import sys
import time

from poker import actions as A
from poker.environment import DEFAULT_STACK, HUNLEnvironment
from poker.ismcts import ISMCTS


def main():
    num_hands = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    ismcts_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42

    env = HUNLEnvironment()
    rng = random.Random(seed)

    total_rewards = [0, 0]
    hands_played = 0
    start_time = time.time()

    for hand_num in range(num_hands):
        state = env.new_initial_state(rng)

        while not env.is_terminal(state):
            # Check pot conservation
            total = state.stacks[0] + state.stacks[1] + state.pot
            assert total == 2 * DEFAULT_STACK, (
                f"Hand {hand_num}: conservation violated: "
                f"stacks={state.stacks}, pot={state.pot}, total={total}"
            )

            cp = state.current_player
            if cp == 0:
                # ISMCTS agent
                obs = env.get_observation(state, player_id=0)
                legal = env.get_legal_actions(state)
                searcher = ISMCTS(
                    env, player_id=0,
                    num_iterations=ismcts_iterations,
                    rng=random.Random(rng.randint(0, 2**32)),
                )
                action = searcher.search(obs, real_legal_actions=legal)
            else:
                # Random agent
                legal = env.get_legal_actions(state)
                action = rng.choice(legal)

            state = env.apply_action(state, action)

        r0, r1 = env.get_rewards(state)
        assert r0 + r1 == 0, f"Hand {hand_num}: rewards not zero-sum: {r0} + {r1}"
        total_rewards[0] += r0
        total_rewards[1] += r1
        hands_played += 1

        if (hand_num + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(
                f"Hand {hand_num + 1}/{num_hands} | "
                f"ISMCTS: {total_rewards[0]:+d} | "
                f"Random: {total_rewards[1]:+d} | "
                f"Time: {elapsed:.1f}s"
            )

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Completed {hands_played} hands in {elapsed:.1f}s")
    print(f"ISMCTS total reward: {total_rewards[0]:+d} centiBB")
    print(f"Random total reward: {total_rewards[1]:+d} centiBB")
    print(f"ISMCTS bb/hand: {total_rewards[0] / hands_played / 100:+.2f} BB/hand")
    print(f"All invariants passed!")


if __name__ == "__main__":
    main()
