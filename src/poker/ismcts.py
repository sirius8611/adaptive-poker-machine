"""Information Set Monte Carlo Tree Search (ISMCTS) for HUNL poker.

Key ISMCTS property: at each iteration, a new determinization (sampled world)
is generated. The search tree is shared across determinizations, but node
selection only follows edges that are legal in the current determinization.
"""

from __future__ import annotations

import math
import random
from typing import Callable

from poker import actions as A
from poker.belief import resample_history
from poker.environment import BIG_BLIND, DEFAULT_STACK, HUNLEnvironment, SMALL_BLIND
from poker.state import HUNLState, Observation
from poker.value import value_function


class ISMCTSNode:
    __slots__ = (
        "parent", "action_from_parent", "children",
        "visit_count", "total_value",
    )

    def __init__(
        self,
        parent: ISMCTSNode | None = None,
        action_from_parent: int | None = None,
    ) -> None:
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children: dict[int, ISMCTSNode] = {}
        self.visit_count: int = 0
        self.total_value: float = 0.0

    @property
    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


class ISMCTS:
    """Information Set MCTS for HUNL poker.

    Usage:
        env = HUNLEnvironment()
        state = env.new_initial_state(rng)
        obs = env.get_observation(state, player_id=0)

        searcher = ISMCTS(env, player_id=0)
        best_action = searcher.search(obs)
    """

    def __init__(
        self,
        env: HUNLEnvironment,
        player_id: int,
        num_iterations: int = 1000,
        exploration_constant: float = 1.41,
        max_depth: int = 40,
        value_fn: Callable[[HUNLState, int], float] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.env = env
        self.player_id = player_id
        self.num_iterations = num_iterations
        self.exploration_constant = exploration_constant
        self.max_depth = max_depth
        self.value_fn = value_fn or self._default_value_fn
        self.rng = rng or random.Random()

    def _default_value_fn(self, state: HUNLState, player_id: int) -> float:
        return value_function(state, player_id, rng=self.rng, num_samples=50)

    def search(self, observation: Observation, real_legal_actions: list[int] | None = None) -> int:
        """Run ISMCTS and return the best action.

        Args:
            observation: The player's observation.
            real_legal_actions: Legal actions in the actual game state. If provided,
                the final action selection is constrained to these. This handles
                edge cases where raise bucket deduplication differs between
                determinized and real states.
        """
        root = ISMCTSNode()

        for _ in range(self.num_iterations):
            # 1. DETERMINIZE: sample a complete state consistent with observation
            det_state = resample_history(
                observation,
                self.env.stack_size,
                self.env.small_blind,
                self.env.big_blind,
                self.rng,
            )

            # 2. SELECT + EXPAND
            node, state, depth = self._select(root, det_state)

            # 3. EVALUATE
            if self.env.is_terminal(state):
                rewards = self.env.get_rewards(state)
                # Normalize reward to [0, 1]
                max_reward = self.env.stack_size
                val = (rewards[self.player_id] + max_reward) / (2 * max_reward)
            elif depth >= self.max_depth:
                val = self.value_fn(state, self.player_id)
            else:
                # Rollout with random play
                val = self._rollout(state, depth)

            # 4. BACKPROPAGATE
            self._backpropagate(node, val)

        # Choose action with highest visit count, constrained to real legal actions
        if not root.children:
            if real_legal_actions:
                return real_legal_actions[0]
            det = resample_history(
                observation,
                self.env.stack_size,
                self.env.small_blind,
                self.env.big_blind,
                self.rng,
            )
            legal = self.env.get_legal_actions(det)
            return legal[0] if legal else A.FOLD

        candidates = root.children
        if real_legal_actions is not None:
            candidates = {a: c for a, c in candidates.items() if a in real_legal_actions}
            if not candidates:
                # No overlap — fall back to most-visited real legal action
                return real_legal_actions[0]

        best_action = max(candidates, key=lambda a: candidates[a].visit_count)
        return best_action

    def _select(
        self, root: ISMCTSNode, state: HUNLState
    ) -> tuple[ISMCTSNode, HUNLState, int]:
        """Select + expand: walk the tree using UCB1, expand one new node."""
        node = root
        depth = 0

        while not self.env.is_terminal(state) and depth < self.max_depth:
            legal = self.env.get_legal_actions(state)
            if not legal:
                break

            # Find untried actions (legal in this determinization but not yet in tree)
            untried = [a for a in legal if a not in node.children]

            if untried:
                # EXPAND: pick a random untried action
                action = self.rng.choice(untried)
                child = ISMCTSNode(parent=node, action_from_parent=action)
                node.children[action] = child
                state = self.env.apply_action(state, action)
                return child, state, depth + 1

            # All legal actions have been tried — select by UCB1
            # Only consider actions that are legal in this determinization
            legal_children = {a: node.children[a] for a in legal if a in node.children}
            if not legal_children:
                break

            best_action = max(
                legal_children,
                key=lambda a: self._ucb1(node, legal_children[a]),
            )
            node = legal_children[best_action]
            state = self.env.apply_action(state, best_action)
            depth += 1

        return node, state, depth

    def _rollout(self, state: HUNLState, depth: int) -> float:
        """Random rollout to terminal or depth limit."""
        while not self.env.is_terminal(state) and depth < self.max_depth:
            legal = self.env.get_legal_actions(state)
            if not legal:
                break
            action = self.rng.choice(legal)
            state = self.env.apply_action(state, action)
            depth += 1

        if self.env.is_terminal(state):
            rewards = self.env.get_rewards(state)
            max_reward = self.env.stack_size
            return (rewards[self.player_id] + max_reward) / (2 * max_reward)

        return self.value_fn(state, self.player_id)

    def _backpropagate(self, node: ISMCTSNode, value: float) -> None:
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node = node.parent

    def _ucb1(self, parent: ISMCTSNode, child: ISMCTSNode) -> float:
        if child.visit_count == 0:
            return float("inf")
        exploitation = child.mean_value
        exploration = self.exploration_constant * math.sqrt(
            math.log(parent.visit_count) / child.visit_count
        )
        return exploitation + exploration

    def get_action_stats(self, root: ISMCTSNode | None = None) -> dict[str, dict]:
        """Get statistics for each action at the root (for debugging)."""
        if root is None:
            return {}
        stats = {}
        for action, child in root.children.items():
            stats[A.action_to_str(action)] = {
                "visits": child.visit_count,
                "mean_value": round(child.mean_value, 4),
            }
        return stats
