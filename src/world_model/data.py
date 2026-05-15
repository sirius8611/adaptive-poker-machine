"""PokerBench data loading and feature encoding.

PokerBench provides solver-optimal play data for HUNL poker.
This module handles:
  1. Loading and parsing hand histories
  2. Encoding game states as observation vectors
  3. Encoding actions with psychological metadata (TTA, sizing)
  4. Creating training sequences for the RSSM
  5. Labeling opponent types for contrastive adapter training

Observation vector (obs_dim=30):
  [0:13]   hole cards (one-hot rank encoding, 2 cards)
  [13:17]  hole card suits (one-hot, 4 suits)
  [17:22]  board texture features (num_cards, flush_draw, straight_draw, paired, suited)
  [22:24]  pot odds and stack-to-pot ratio
  [24:26]  street one-hot (4 → compressed to 2 bits)
  [26:28]  position (IP/OOP), num_players_acted
  [28:30]  bet_facing_ratio, aggression_this_street
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from world_model.config import WorldModelConfig

# Opponent type labels for contrastive learning
OPPONENT_TYPES = {
    "tight_passive": 0,
    "tight_aggressive": 1,
    "loose_passive": 2,
    "loose_aggressive": 3,
}


@dataclass
class HandRecord:
    """A single parsed hand history."""

    hand_id: str
    hero_cards: tuple[int, int]
    villain_cards: tuple[int, int]
    board: list[int]
    actions: list[dict]  # [{type, amount, street, player, tta}, ...]
    result: float  # Final EV for hero
    hero_position: int  # 0=SB, 1=BB
    stack_size: float


@dataclass
class TrainingSequence:
    """A sequence of (observation, action, reward) tuples for RSSM training."""

    observations: torch.Tensor  # (T, obs_dim)
    actions: torch.Tensor  # (T, action_dim)
    rewards: torch.Tensor  # (T,)
    opponent_actions: torch.Tensor  # (T_opp, adapter_action_dim)
    opponent_type: int  # Label for contrastive loss


def encode_observation(
    hero_cards: tuple[int, int],
    board: list[int],
    pot: float,
    stack: float,
    street: int,
    position: int,
    bet_facing: float,
    num_actions_street: int,
    stack_size: float,
) -> np.ndarray:
    """Encode a game state into a fixed-size observation vector.

    Returns: (obs_dim=30,) numpy array.
    """
    obs = np.zeros(30, dtype=np.float32)

    # Hole cards: rank encoding (0-12 for each card)
    for i, card in enumerate(hero_cards):
        rank = card // 4
        obs[rank] = 1.0  # One-hot ranks (overlapping is fine for 2 cards)

    # Hole card suits
    for card in hero_cards:
        suit = card % 4
        obs[13 + suit] = 1.0

    # Board features
    obs[17] = len(board) / 5.0  # Normalized card count

    if board:
        suits = [c % 4 for c in board]
        ranks = [c // 4 for c in board]

        # Flush draw potential
        suit_counts = [suits.count(s) for s in range(4)]
        obs[18] = max(suit_counts) / 5.0

        # Straight draw potential (connected ranks)
        sorted_ranks = sorted(set(ranks))
        max_connected = 1
        curr_connected = 1
        for i in range(1, len(sorted_ranks)):
            if sorted_ranks[i] - sorted_ranks[i - 1] <= 2:
                curr_connected += 1
                max_connected = max(max_connected, curr_connected)
            else:
                curr_connected = 1
        obs[19] = max_connected / 5.0

        # Paired board
        obs[20] = 1.0 if len(ranks) != len(set(ranks)) else 0.0

        # Suited board (3+ same suit)
        obs[21] = 1.0 if max(suit_counts) >= 3 else 0.0

    # Pot odds and SPR
    obs[22] = min(bet_facing / max(pot, 1), 2.0) / 2.0  # Pot odds normalized
    obs[23] = min(stack / max(pot, 1), 20.0) / 20.0  # SPR normalized

    # Street encoding (2 bits)
    obs[24] = street / 3.0
    obs[25] = 1.0 if street >= 2 else 0.0

    # Position and action count
    obs[26] = float(position)
    obs[27] = min(num_actions_street, 6) / 6.0

    # Bet facing and aggression
    obs[28] = min(bet_facing / max(stack_size, 1), 1.0)
    obs[29] = min(num_actions_street, 4) / 4.0  # Proxy for street aggression

    return obs


def encode_action(
    action_type: int,
    bet_ratio: float,
    tta: float = 0.0,
    is_allin: bool = False,
) -> np.ndarray:
    """Encode an action for RSSM input.

    Returns: (action_dim=4,) numpy array.
    [action_type_normalized, bet_ratio, tta_normalized, is_allin]
    """
    return np.array(
        [action_type / 3.0, bet_ratio, np.log1p(tta), float(is_allin)],
        dtype=np.float32,
    )


def encode_opponent_action(
    action_type: int,
    bet_ratio: float,
    tta: float,
    sizing_pattern: int = 0,
    street: int = 0,
    position: int = 0,
) -> np.ndarray:
    """Encode an opponent action for the adapter.

    Returns: (adapter_action_dim=6,) numpy array.
    """
    return np.array(
        [action_type, bet_ratio, np.log1p(tta), sizing_pattern, street, position],
        dtype=np.float32,
    )


def classify_opponent_type(actions: list[dict]) -> int:
    """Classify opponent type from their action history.

    Uses VPIP (voluntarily put in pot) and PFR (preflop raise) heuristics:
      - Tight: VPIP < 25%
      - Loose: VPIP >= 25%
      - Passive: aggression factor < 1.0
      - Aggressive: aggression factor >= 1.0

    Also considers TTA patterns:
      - Fast actions (< 2s) with large bets → aggressive
      - Slow actions (> 5s) with small bets → passive
    """
    if not actions:
        return OPPONENT_TYPES["tight_passive"]

    raises = sum(1 for a in actions if a.get("type") == "raise")
    calls = sum(1 for a in actions if a.get("type") == "call")
    checks = sum(1 for a in actions if a.get("type") == "check")
    total = raises + calls + checks
    if total == 0:
        return OPPONENT_TYPES["tight_passive"]

    vpip = (raises + calls) / total
    agg_factor = raises / max(calls, 1)

    avg_tta = np.mean([a.get("tta", 3.0) for a in actions])

    # Adjust for timing tells
    if avg_tta < 2.0:
        agg_factor *= 1.2  # Fast players tend to be more aggressive

    is_tight = vpip < 0.35
    is_aggressive = agg_factor >= 1.0

    if is_tight and is_aggressive:
        return OPPONENT_TYPES["tight_aggressive"]
    elif is_tight:
        return OPPONENT_TYPES["tight_passive"]
    elif is_aggressive:
        return OPPONENT_TYPES["loose_aggressive"]
    else:
        return OPPONENT_TYPES["loose_passive"]


class PokerBenchDataset(Dataset):
    """PyTorch Dataset for PokerBench hand histories.

    Expected data format (JSON lines):
    {
        "hand_id": "123",
        "hero_cards": [48, 50],
        "villain_cards": [44, 45],
        "board": [0, 5, 10, 20, 30],
        "actions": [
            {"type": "raise", "amount": 0.5, "street": 0, "player": "hero", "tta": 1.2},
            {"type": "call", "amount": 0.5, "street": 0, "player": "villain", "tta": 3.5},
            ...
        ],
        "result": 2.5,
        "hero_position": 0,
        "stack_size": 100.0
    }
    """

    def __init__(self, data_path: str, cfg: WorldModelConfig, max_samples: int | None = None) -> None:
        self.cfg = cfg
        self.sequences: list[TrainingSequence] = []
        self._load_data(data_path, max_samples)

    def _load_data(self, data_path: str, max_samples: int | None) -> None:
        path = Path(data_path)
        if not path.exists():
            return  # Empty dataset — will be populated when data is available

        files = sorted(path.glob("*.jsonl")) + sorted(path.glob("*.json"))
        count = 0

        for f in files:
            with open(f) as fp:
                for line in fp:
                    if max_samples and count >= max_samples:
                        return
                    try:
                        record = json.loads(line.strip())
                        seq = self._process_record(record)
                        if seq is not None:
                            self.sequences.append(seq)
                            count += 1
                    except (json.JSONDecodeError, KeyError):
                        continue

    def _process_record(self, record: dict) -> TrainingSequence | None:
        """Convert a raw JSON record to a training sequence."""
        actions = record.get("actions", [])
        if not actions:
            return None

        hero_cards = tuple(record["hero_cards"])
        board = record.get("board", [])
        stack_size = record.get("stack_size", 100.0)
        hero_position = record.get("hero_position", 0)

        obs_list = []
        act_list = []
        reward_list = []
        opp_act_list = []

        pot = 1.5  # Blinds
        stack = stack_size - (0.5 if hero_position == 0 else 1.0)
        bet_facing = 0.0
        current_board: list[int] = []
        num_actions_street = 0
        current_street = 0

        for i, action in enumerate(actions):
            street = action.get("street", current_street)
            if street != current_street:
                current_street = street
                num_actions_street = 0
                # Update board
                cards_needed = {0: 0, 1: 3, 2: 4, 3: 5}.get(street, 0)
                current_board = board[:cards_needed]

            # Encode observation at this step
            obs = encode_observation(
                hero_cards, current_board, pot, stack,
                street, hero_position, bet_facing,
                num_actions_street, stack_size,
            )
            obs_list.append(obs)

            # Encode action
            action_type_map = {"fold": 0, "check": 1, "call": 2, "raise": 3}
            atype = action_type_map.get(action["type"], 1)
            bet_ratio = action.get("amount", 0.0) / max(stack_size, 1.0)
            tta = action.get("tta", 3.0)
            is_allin = action.get("is_allin", False)

            act = encode_action(atype, bet_ratio, tta, is_allin)
            act_list.append(act)

            # Update state
            if action["player"] == "hero":
                amount = action.get("amount", 0.0)
                stack -= amount
                pot += amount
                bet_facing = 0.0
            else:
                amount = action.get("amount", 0.0)
                pot += amount
                bet_facing = amount
                # Opponent action for adapter
                sizing_pattern = self._classify_sizing(amount, pot)
                opp_act = encode_opponent_action(
                    atype, bet_ratio, tta, sizing_pattern, street, 1 - hero_position
                )
                opp_act_list.append(opp_act)

            num_actions_street += 1

            # Intermediate reward is 0 except at terminal
            if i == len(actions) - 1:
                reward_list.append(record.get("result", 0.0))
            else:
                reward_list.append(0.0)

        # Classify opponent
        villain_actions = [a for a in actions if a.get("player") == "villain"]
        opp_type = classify_opponent_type(villain_actions)

        # Pad or truncate to seq_len
        T = len(obs_list)
        if T == 0:
            return None

        return TrainingSequence(
            observations=torch.tensor(np.stack(obs_list), dtype=torch.float32),
            actions=torch.tensor(np.stack(act_list), dtype=torch.float32),
            rewards=torch.tensor(reward_list, dtype=torch.float32),
            opponent_actions=(
                torch.tensor(np.stack(opp_act_list), dtype=torch.float32)
                if opp_act_list
                else torch.zeros(1, self.cfg.adapter_action_dim)
            ),
            opponent_type=opp_type,
        )

    def _classify_sizing(self, amount: float, pot: float) -> int:
        """Classify bet sizing pattern: 0=small, 1=medium, 2=large, 3=overbet."""
        if pot <= 0:
            return 1
        ratio = amount / pot
        if ratio < 0.4:
            return 0
        elif ratio < 0.8:
            return 1
        elif ratio < 1.5:
            return 2
        return 3

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> TrainingSequence:
        return self.sequences[idx]


def collate_sequences(
    batch: list[TrainingSequence], cfg: WorldModelConfig
) -> dict[str, torch.Tensor]:
    """Collate variable-length sequences into padded tensors.

    Returns dict with:
        observations: (B, T, obs_dim)
        actions: (B, T, action_dim)
        rewards: (B, T)
        opponent_actions: (B, T_opp, adapter_action_dim)
        opponent_mask: (B, T_opp) — True for padded positions
        opponent_types: (B,)
        seq_mask: (B, T) — True for padded positions
    """
    B = len(batch)
    max_T = max(seq.observations.shape[0] for seq in batch)
    max_T_opp = max(seq.opponent_actions.shape[0] for seq in batch)

    # Cap at config limits
    max_T = min(max_T, cfg.seq_len)
    max_T_opp = min(max_T_opp, cfg.adapter_context_len)

    obs = torch.zeros(B, max_T, cfg.obs_dim)
    acts = torch.zeros(B, max_T, cfg.action_dim)
    rews = torch.zeros(B, max_T)
    opp_acts = torch.zeros(B, max_T_opp, cfg.adapter_action_dim)
    seq_mask = torch.ones(B, max_T, dtype=torch.bool)
    opp_mask = torch.ones(B, max_T_opp, dtype=torch.bool)
    opp_types = torch.zeros(B, dtype=torch.long)

    for i, seq in enumerate(batch):
        T = min(seq.observations.shape[0], max_T)
        obs[i, :T] = seq.observations[:T]
        acts[i, :T] = seq.actions[:T]
        rews[i, :T] = seq.rewards[:T]
        seq_mask[i, :T] = False

        T_opp = min(seq.opponent_actions.shape[0], max_T_opp)
        opp_acts[i, :T_opp] = seq.opponent_actions[:T_opp]
        opp_mask[i, :T_opp] = False

        opp_types[i] = seq.opponent_type

    return {
        "observations": obs,
        "actions": acts,
        "rewards": rews,
        "opponent_actions": opp_acts,
        "opponent_mask": opp_mask,
        "opponent_types": opp_types,
        "seq_mask": seq_mask,
    }


def create_dataloader(
    data_path: str, cfg: WorldModelConfig, shuffle: bool = True
) -> DataLoader:
    """Create a DataLoader for PokerBench data."""
    dataset = PokerBenchDataset(data_path, cfg)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_sequences(batch, cfg),
        drop_last=True,
    )
