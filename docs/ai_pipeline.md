# AI Decision Pipeline: From Input to Action

## Overview

```
Observation → Determinize → Search (ISMCTS) → Best Action
```

The agent receives **partial information** (its own cards + public state), imagines many possible worlds, and searches each one to find the best action.

---

## Pipeline Steps

### 1. Observation (Input)

The agent sees only what a real player would see:

```
Observation
├── my_hand: (Card, Card)          # e.g. [As Kh]
├── player_id: int                 # 0 (SB) or 1 (BB)
└── public:
    ├── board: (Card, ...)         # e.g. [Tc Jd Qh] on flop
    ├── pot: int                   # chips in pot
    ├── stacks: (int, int)         # remaining chips per player
    ├── street: int                # preflop/flop/turn/river
    ├── current_player: int
    ├── bet_to_call: int
    └── action_history: ((int,...), ...)  # all actions per street
```

**What's hidden**: opponent's hole cards, remaining deck order.

---

### 2. Determinize (Belief Sampling)

> `belief.py → resample_history()`

Since we don't know the opponent's cards, we **sample a possible world**:

```
Known cards (my hand + board)
        │
        ▼
Remaining 45-48 cards
        │
        ├──→ Sample 2 cards → Opponent's hand
        └──→ Shuffle rest   → Future community cards
        │
        ▼
Complete HUNLState (fully specified, deterministic)
```

This runs **once per ISMCTS iteration**. Each iteration imagines a different opponent hand.

---

### 3. ISMCTS Search (Core Loop)

> `ismcts.py → ISMCTS.search()`

Runs N iterations (default: 200). Each iteration:

```
┌─────────────────────────────────────────────────┐
│  For each iteration i = 1..N:                   │
│                                                 │
│  ① DETERMINIZE                                  │
│     Sample a new possible world                 │
│     (new opponent hand + deck each time)        │
│                                                 │
│  ② SELECT                                       │
│     Walk the shared tree using UCB1             │
│     UCB1 = mean_value + C * √(ln(parent)/child) │
│     Only follow actions legal in THIS world     │
│                                                 │
│  ③ EXPAND                                       │
│     Hit an untried action? Add new tree node    │
│                                                 │
│  ④ EVALUATE                                     │
│     Terminal state? → exact reward              │
│     Depth limit?    → heuristic value function  │
│     Otherwise?      → random rollout            │
│                                                 │
│  ⑤ BACKPROPAGATE                                │
│     Update visit_count and total_value          │
│     from leaf back up to root                   │
└─────────────────────────────────────────────────┘
```

**Key insight**: the tree is shared across all sampled worlds, but each traversal only considers actions legal in that specific world. This averages over uncertainty about the opponent's hand.

---

### 4. Value Function (Leaf Evaluation)

> `value.py → value_function()`

When search hits a non-terminal leaf:

```
Both hands known (in sampled world)
        │
        ├── River (5 cards)? → Exact hand comparison
        │
        └── Pre-river? → Monte Carlo equity
                │
                ├── Sample N board completions
                ├── Evaluate both hands each time
                └── Average win rate = equity
        │
        ▼
Blend: 0.7 × equity + 0.3 × (equity × pot / max_payoff)
        │
        ▼
Value ∈ [0.0, 1.0]
```

---

### 5. Action Selection (Output)

After all iterations complete:

```
Root node children:
  fold      → 12 visits, mean 0.31
  call      → 45 visits, mean 0.52
  raise_75  → 89 visits, mean 0.61  ← highest visits
  raise_100 → 41 visits, mean 0.58
  all_in    → 13 visits, mean 0.44

Selected: raise_75 (most visited)
```

The action with the **highest visit count** wins (not highest value — visit count is more robust).

---

## Full Diagram

```
┌──────────────┐
│  Game State   │
│  (partial)    │
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌───────────────────────────────────┐
│  Observation  │────▶│  ISMCTS Search (N iterations)     │
│  - my hand    │     │                                   │
│  - board      │     │  ┌─────────┐    ┌──────────────┐ │
│  - pot/stacks │     │  │Resample │───▶│ Select (UCB1)│ │
│  - history    │     │  │opponent │    └──────┬───────┘ │
│               │     │  │hand     │           │         │
│               │     │  └─────────┘    ┌──────▼───────┐ │
│               │     │                 │   Expand     │ │
│               │     │                 └──────┬───────┘ │
│               │     │                 ┌──────▼───────┐ │
│               │     │                 │  Evaluate    │ │
│               │     │                 │  (value fn)  │ │
│               │     │                 └──────┬───────┘ │
│               │     │                 ┌──────▼───────┐ │
│               │     │                 │Backpropagate │ │
│               │     │                 └──────────────┘ │
│               │     └──────────────────┬──────────────┘
└──────────────┘                         │
                                         ▼
                                  ┌──────────────┐
                                  │  Best Action  │
                                  │  (most visits)│
                                  └──────────────┘
```

---

## Complexity

| Component | Per iteration | Total (N=200) |
|-----------|--------------|---------------|
| Resample | O(52) shuffle | 200 × O(52) |
| Select | O(depth) tree walk | 200 × O(40) |
| Evaluate | O(samples) MC equity | 200 × O(50) hand evals |
| **Total** | ~milliseconds | **~50-200ms per decision** |
