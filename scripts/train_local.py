"""Quick local training to produce world_model.pt for the UI.

Generates ~500 synthetic self-play hands and trains the World Model for a few
epochs. CPU-friendly; expect ~10-20 min total. Output: ./world_model.pt

Usage:
    python scripts/train_local.py [--hands 500] [--wm-epochs 4] [--policy-epochs 3]
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from poker import actions as A
from poker.environment import HUNLEnvironment
from world_model.agent import WorldModelAgent
from world_model.config import WorldModelConfig
from world_model.data import (
    OPPONENT_TYPES,
    TrainingSequence,
    collate_sequences,
    encode_action,
    encode_observation,
    encode_opponent_action,
)
from world_model.train import Trainer


STYLE_PREFS = {
    "tight_passive":    [A.CHECK, A.CALL, A.FOLD, A.RAISE_50],
    "tight_aggressive": [A.RAISE_75, A.RAISE_100, A.CALL, A.FOLD],
    "loose_passive":    [A.CALL, A.CHECK, A.RAISE_33],
    "loose_aggressive": [A.RAISE_100, A.RAISE_150, A.CALL, A.ALL_IN],
}
STYLE_TTA_BASE = {"tight_passive": 4.0, "tight_aggressive": 1.5,
                  "loose_passive": 3.0, "loose_aggressive": 1.0}


def pick(style: str, legal: list[int], rng: random.Random) -> int:
    for pref in STYLE_PREFS[style]:
        if pref in legal:
            return pref
    return rng.choice(legal)


def tta_for(style: str, action: int, rng: random.Random) -> float:
    base = STYLE_TTA_BASE[style]
    if action in (A.RAISE_75, A.RAISE_100, A.RAISE_150, A.RAISE_200, A.ALL_IN):
        base *= 0.7
    return max(0.1, base + rng.uniform(-0.4, 0.6))


def play_hand(env: HUNLEnvironment, rng: random.Random, style: str, cfg: WorldModelConfig):
    state = env.new_initial_state(rng)
    hero, villain = 0, 1
    obs_list, act_list, rew_list, opp_act_list = [], [], [], []
    stack_size = env.stack_size

    while not env.is_terminal(state):
        cp = state.current_player
        legal = env.get_legal_actions(state)
        if cp == hero:
            action = rng.choice(legal)
            obs = encode_observation(
                hero_cards=state.hole_cards[hero], board=list(state.board),
                pot=state.pot / 100, stack=state.stacks[hero] / 100,
                street=state.street, position=hero,
                bet_facing=state.bet_to_call / 100,
                num_actions_street=state.num_actions_this_street,
                stack_size=stack_size / 100,
            )
            atype = 0 if action == A.FOLD else 1 if action == A.CHECK else 2 if action == A.CALL else 3
            if A.RAISE_33 <= action <= A.RAISE_200:
                chips_after_call = state.stacks[hero] - min(state.bet_to_call, state.stacks[hero])
                amt = A.raise_amount(action, state.pot, state.min_raise, chips_after_call)
                bet_ratio = amt / max(state.stacks[hero], 1)
            elif action == A.ALL_IN:
                bet_ratio = 1.0
            else:
                bet_ratio = 0.0
            obs_list.append(obs)
            act_list.append(encode_action(atype, bet_ratio, rng.uniform(0.8, 3.5), action == A.ALL_IN))
            rew_list.append(0.0)
        else:
            action = pick(style, legal, rng)
            atype = 0 if action == A.FOLD else 1 if action == A.CHECK else 2 if action == A.CALL else 3
            if A.RAISE_33 <= action <= A.RAISE_200:
                chips_after_call = state.stacks[villain] - min(state.bet_to_call, state.stacks[villain])
                amt = A.raise_amount(action, state.pot, state.min_raise, chips_after_call)
                bet_ratio = amt / max(state.stacks[villain], 1)
            elif action == A.ALL_IN:
                bet_ratio = 1.0
            else:
                bet_ratio = 0.0
            sizing = 0 if bet_ratio < 0.2 else 1 if bet_ratio < 0.5 else 2 if bet_ratio < 1.0 else 3
            opp_act_list.append(encode_opponent_action(
                atype, bet_ratio, tta_for(style, action, rng), sizing, state.street, villain,
            ))
        state = env.apply_action(state, action)

    r0, _ = env.get_rewards(state)
    if rew_list:
        rew_list[-1] = r0 / stack_size
    if not obs_list:
        return None
    return TrainingSequence(
        observations=torch.tensor(np.stack(obs_list), dtype=torch.float32),
        actions=torch.tensor(np.stack(act_list), dtype=torch.float32),
        rewards=torch.tensor(rew_list, dtype=torch.float32),
        opponent_actions=(
            torch.tensor(np.stack(opp_act_list), dtype=torch.float32)
            if opp_act_list else torch.zeros(1, cfg.adapter_action_dim)
        ),
        opponent_type=OPPONENT_TYPES[style],
    )


class InMemoryDataset(Dataset):
    def __init__(self, seqs): self.seqs = seqs
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i): return self.seqs[i]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands", type=int, default=500)
    parser.add_argument("--wm-epochs", type=int, default=4)
    parser.add_argument("--policy-epochs", type=int, default=3)
    parser.add_argument("--out", type=str, default="world_model.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = WorldModelConfig(batch_size=16, seq_len=32, num_imagined_trajectories=16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Generating {args.hands} hands...")
    env = HUNLEnvironment()
    rng = random.Random(args.seed)
    styles = list(STYLE_PREFS.keys())
    seqs = []
    t0 = time.time()
    for i in range(args.hands):
        seq = play_hand(env, rng, rng.choice(styles), cfg)
        if seq is not None:
            seqs.append(seq)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{args.hands} hands ({time.time()-t0:.1f}s)")
    print(f"Generated {len(seqs)} sequences in {time.time()-t0:.1f}s")

    dataset = InMemoryDataset(seqs)
    dl = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                    collate_fn=lambda b: collate_sequences(b, cfg), drop_last=True)

    agent = WorldModelAgent(cfg).to(device)
    trainer = Trainer(agent, cfg, device=device)
    print(f"Agent params: {sum(p.numel() for p in agent.parameters())/1e6:.2f}M | batches/epoch: {len(dl)}")

    for ep in range(args.wm_epochs):
        t0 = time.time()
        m = trainer.train_epoch(dl, phase="world_model")
        print(f"[WM {ep+1}/{args.wm_epochs}] " + " | ".join(f"{k}={v:.4f}" for k, v in m.items())
              + f"  ({time.time()-t0:.1f}s)")

    for ep in range(args.policy_epochs):
        t0 = time.time()
        m = trainer.train_epoch(dl, phase="policy")
        print(f"[Pol {ep+1}/{args.policy_epochs}] " + " | ".join(f"{k}={v:.4f}" for k, v in m.items())
              + f"  ({time.time()-t0:.1f}s)")

    trainer.save(args.out)
    print(f"\nSaved checkpoint to: {args.out}")
    print(f"Now run:  python scripts/play_ui.py --ckpt {args.out}")


if __name__ == "__main__":
    main()
