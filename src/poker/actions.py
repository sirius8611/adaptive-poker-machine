"""Action encoding for HUNL No-Limit Texas Hold'em.

Discrete action space:
  0 = FOLD
  1 = CHECK
  2 = CALL
  3-8 = RAISE at pot fractions (0.33, 0.5, 0.75, 1.0, 1.5, 2.0)
  9 = ALL_IN
"""

from __future__ import annotations

FOLD = 0
CHECK = 1
CALL = 2
RAISE_33 = 3
RAISE_50 = 4
RAISE_75 = 5
RAISE_100 = 6
RAISE_150 = 7
RAISE_200 = 8
ALL_IN = 9

NUM_ACTIONS = 10

RAISE_BUCKETS = (0.33, 0.5, 0.75, 1.0, 1.5, 2.0)

ACTION_NAMES = [
    "fold", "check", "call",
    "raise_33", "raise_50", "raise_75",
    "raise_100", "raise_150", "raise_200",
    "all_in",
]


def action_to_str(action: int) -> str:
    if 0 <= action < NUM_ACTIONS:
        return ACTION_NAMES[action]
    raise ValueError(f"Invalid action: {action}")


def raise_amount(action: int, pot: int, min_raise: int, stack: int) -> int:
    """Convert a raise action to a chip amount (total bet size).

    Args:
        action: Action ID (must be RAISE_33..RAISE_200).
        pot: Current pot size.
        min_raise: Minimum legal raise amount.
        stack: Acting player's remaining stack.

    Returns:
        The chip amount for this raise, clamped to [min_raise, stack].
    """
    if action < RAISE_33 or action > RAISE_200:
        raise ValueError(f"Not a raise action: {action}")
    bucket_idx = action - RAISE_33
    fraction = RAISE_BUCKETS[bucket_idx]
    amount = int(pot * fraction)
    amount = max(amount, min_raise)
    amount = min(amount, stack)
    return amount
