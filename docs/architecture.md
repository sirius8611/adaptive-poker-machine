# Architecture: Adaptive Poker Machine

This document describes the two complementary approaches that make up the adaptive
poker machine for Heads-Up No-Limit Texas Hold'em (HUNL):

1. **Classical search** — an Information Set MCTS (ISMCTS) agent that plans by
   sampling possible worlds and searching each one. Acts as the deterministic
   baseline / fallback.
2. **Learned world model** — a Recurrent State-Space Model (RSSM) with an
   Opponent Adapter that learns to plan in latent space and adapts to specific
   opponents online.

Both share the same poker engine ([src/poker/environment.py](src/poker/environment.py))
and action / observation conventions, so a single environment can drive either
agent.

---

## 1. Shared poker engine

### 1.1 Card and action encoding

- [src/poker/cards.py](src/poker/cards.py) — Cards are encoded as ints in
  `[0, 52)` via `card_id = rank*4 + suit`. `Deck.shuffle`, `Deck.deal`, and
  `Deck.deal_excluding` provide deterministic dealing given an RNG.
- [src/poker/actions.py](src/poker/actions.py) — Discrete action space of
  size 10: `FOLD, CHECK, CALL`, six pot-fraction raises
  `(0.33, 0.5, 0.75, 1.0, 1.5, 2.0)`, and `ALL_IN`.
  `raise_amount(action, pot, min_raise, stack)` clamps a bucket to the legal
  raise range.

### 1.2 State and environment

- [src/poker/state.py](src/poker/state.py) defines three frozen dataclasses:
  - `HUNLState` — full game state (both hole cards, deck, board, pot, stacks,
    street, betting bookkeeping, terminal flags).
  - `PublicState` — what's visible to all players.
  - `Observation` — `PublicState` + the acting player's own hand.
- [src/poker/environment.py](src/poker/environment.py) — `HUNLEnvironment`
  exposes an OpenSpiel-style API:
  - `new_initial_state(rng)` deals hole cards and posts blinds (SB acts first
    preflop, BB acts first postflop).
  - `apply_action(state, action)` returns a new `HUNLState`. Pure function:
    randomness is resolved at deal time via the pre-shuffled `deck_remaining`.
  - `get_legal_actions(state)` enumerates legal discrete actions, deduplicating
    raise buckets that collapse to the same chip amount.
  - `get_rewards(state)` returns terminal stack deltas (fold or showdown).
  - `get_observation(state, player_id)` projects to the player's view.
  - Helpers `_is_street_over`, `_deal_street`, `_deal_to_river` advance betting
    rounds and run out the board on all-ins.

### 1.3 Hand evaluation

- [src/poker/evaluator.py](src/poker/evaluator.py) wraps the `treys` library:
  - `evaluate_hand(cards)` — returns a strength rank where lower is better.
  - `compare_hands(hand_a, hand_b, board)` — returns `+1 / -1 / 0`.

---

## 2. Approach A — Information Set MCTS

End-to-end pipeline for the classical agent (also documented in
[docs/ai_pipeline.md](docs/ai_pipeline.md)):

```
Observation → Determinize → ISMCTS search → Best action (most visits)
```

### 2.1 Determinization

- [src/poker/belief.py](src/poker/belief.py) — `resample_history(observation,
  ...)` produces a fully-specified `HUNLState` consistent with what the player
  sees: the player's own hole cards and the board are kept fixed; the
  opponent's hole cards and the remaining deck are sampled uniformly from the
  unseen cards. Optional `original_state` lets callers preserve betting
  bookkeeping (e.g. `min_raise`, `last_raise`) that isn't recoverable from
  `PublicState` alone.

### 2.2 ISMCTS search

- [src/poker/ismcts.py](src/poker/ismcts.py)
  - `ISMCTSNode` — a tree node with `parent`, `action_from_parent`,
    `children`, `visit_count`, `total_value`, and a `mean_value` property.
  - `ISMCTS.search(observation, real_legal_actions=None)` — runs
    `num_iterations` (default 1000) iterations of:
    1. **Determinize** with `resample_history`.
    2. **Select + expand** via `_select`: walk the shared tree using UCB1
       (`_ucb1`), but only consider edges legal in the current sample. The
       first untried legal action is expanded as a new child.
    3. **Evaluate**: terminal → exact normalized reward; depth-limited →
       `value_fn`; otherwise → `_rollout` (random play to terminal or depth).
    4. **Backpropagate** through `_backpropagate`, incrementing
       `visit_count` and `total_value` up to the root.
    Final action selection uses the **most-visited** root child (more robust
    than mean value), constrained to `real_legal_actions` when provided.
  - `get_action_stats(root)` — debug summary of root children.

### 2.3 Heuristic value function

- [src/poker/value.py](src/poker/value.py) — used at leaves of depth-limited
  search and as the rollout fallback:
  - `hand_equity_exact(hand, opp_hand, board)` — closed form on a 5-card
    board, returns `1.0 / 0.5 / 0.0`.
  - `hand_equity_monte_carlo(hand, opp_hand, board, rng, num_samples)` —
    samples board completions and averages win rate.
  - `value_function(state, player_id, ...)` — picks exact vs MC equity by
    board length, then blends with a pot-odds term:
    `value = 0.7 · equity + 0.3 · (equity · pot / max_payoff)`, clipped to
    `[0, 1]`.

---

## 3. Approach B — World Model agent

The world-model agent ([src/world_model/](src/world_model/)) replaces card
sampling with a learned latent transition model conditioned on a per-opponent
embedding. It is designed to **adapt online**: as opponent actions are
observed, the embedding `z_opp` is updated and the same world model produces
different predicted dynamics — no retraining needed.

### 3.1 Configuration

- [src/world_model/config.py](src/world_model/config.py) — `WorldModelConfig`
  centralizes dimensions (`state_dim`, `stoch_classes × stoch_categories`,
  `obs_dim=30`, `action_dim=4`, `opp_embed_dim=64`), training hyperparameters
  (`lr`, `batch_size`, `seq_len`, `kl_scale`, `contrastive_scale`,
  `grad_clip`), and search hyperparameters
  (`imagination_horizon=8`, `num_imagined_trajectories=64`, `discount=0.99`).

### 3.2 Recurrent State-Space Model (RSSM)

- [src/world_model/rssm.py](src/world_model/rssm.py) — latent state
  `s_t = (h_t, z_t)` with deterministic GRU recurrence and categorical
  stochastic latents.
  - `obs_encoder` — MLP that embeds the 30-D observation.
  - `_deterministic_transition(prev_h, prev_z, action, z_opp)` — concatenates
    inputs, applies a pre-MLP, then a `GRUCell` to produce `h_t`.
  - `prior_net` — `p(z_t | h_t)`, used during imagination.
  - `posterior_net` — `q(z_t | h_t, o_t)`, used during training (teacher
    forcing).
  - `_sample_stochastic` — categorical sampling: hard Gumbel-Softmax with
    straight-through gradients during training, argmax at inference.
  - `reward_head` — predicts scalar EV from `(h_t, z_t)`.
  - `obs_decoder` — reconstructs `o_t` from the latent (used for the
    reconstruction loss).
  - `observe_step` — one training step: returns posterior `z`, prior/posterior
    logits, predicted reward, predicted obs.
  - `imagine_step` — one rollout step using only the prior (no observation).
  - `initial_state`, `get_features`, `feature_dim` — utilities for downstream
    heads.

### 3.3 Opponent Adapter

- [src/world_model/adapter.py](src/world_model/adapter.py) — produces the
  opponent embedding `z_opp` from a window of recent opponent actions. Each
  action is the 6-tuple
  `[action_type, bet_ratio, log1p(tta), sizing_pattern, street, position]`,
  capturing playing style **and** psychological signals (Time-to-Action,
  sizing patterns).
  - `PositionalEncoding` — sinusoidal positional encoding.
  - `OpponentAdapter` — input projection MLP → positional encoding →
    `TransformerEncoder` (`adapter_layers=2`, `adapter_heads=4`, causal mask)
    → output projection. Pools the last non-padded token. A learnable
    `default_embedding` is returned when the buffer is empty.
  - `OnlineAdapterState` — rolling buffer (`adapter_context_len=32`) used at
    inference; `push`, `get_history`, `get_mask`, `reset`. This is the
    **fast-path adaptation** mechanism: opponent observations update `z_opp`
    in real time without touching model weights.

### 3.4 Policy and Value heads

- [src/world_model/heads.py](src/world_model/heads.py)
  - `TanhNormal` — tanh-squashed Normal, rescaled to `[0, 1]` for bet ratios.
    Provides `sample`, `log_prob` (with tanh + rescale Jacobians), and `mode`.
  - `PolicyHead` — shared MLP trunk producing a 4-way action-type categorical
    (`fold/check/call/raise`) **and** a continuous bet-ratio `TanhNormal`
    parameterized by `(mu, log_std)` clamped to `[min_log_std, max_log_std]`.
    `sample_action(features, deterministic)` returns sampled or modal
    (action_type, bet_ratio) plus the continuous log-prob.
  - `ValueHead` — MLP critic producing `V(s_t)` for use as a baseline and as
    the terminal bootstrap during search.

### 3.5 Latent Look-Ahead search

- [src/world_model/search.py](src/world_model/search.py) — replaces ISMCTS
  with MPC-style rollouts in latent space; opponent embedding is held fixed.
  - `LatentLookAhead.search(h, z, z_opp, legal_mask)`:
    1. Tile `(h, z, z_opp)` to `K = num_imagined_trajectories` parallel
       trajectories.
    2. For `t = 0..H-1`: sample a policy action; on `t == 0` apply the legal
       action mask via masked logits and record the first action; encode the
       action and call `rssm.imagine_step`; accumulate
       `discount_factor · reward_pred`.
    3. Add discounted terminal `value(features)` as bootstrap.
    4. Pick the trajectory with the highest total return; return its first
       `(action_type, bet_ratio)`.
  - `search_with_averaging` — runs `num_rounds` searches and averages returns
    per first-action-type before selecting (more robust than single-shot).
  - `_encode_action(action_type, bet_ratio)` — packs into the RSSM's 4-D
    action format `[action_type/3, bet_ratio, tta_placeholder, is_allin]`.

### 3.6 Agent (training + live play)

- [src/world_model/agent.py](src/world_model/agent.py)
  - `WorldModelAgent` — owns `RSSM`, `OpponentAdapter`, `PolicyHead`,
    `ValueHead`, `LatentLookAhead`.
    - `forward_train(observations, actions, opponent_actions, opponent_mask)` —
      teacher-forced rollout through `observe_step` over `T` time steps;
      returns prior/posterior logits, reward predictions, obs
      reconstructions, and the latent sequences for loss computation.
    - `imagine_trajectories(h_start, z_start, z_opp, horizon)` — actor-critic
      rollout in imagination used by the policy loss: samples actions from
      the policy, advances via `imagine_step`, records logits, log-probs,
      rewards, and values.
  - `LiveAgent` — stateful wrapper for play:
    - `new_hand` resets RSSM state.
    - `observe_opponent_action` pushes onto the adapter buffer.
    - `observe_and_act` — encodes the current observation, recomputes
      `z_opp`, advances RSSM with `observe_step` (a dummy action for the
      first call), then either runs `LatentLookAhead.search` (with a legal-
      action mask) or calls the policy directly (`use_search=False`).
    - `reset_opponent_model` — clears the adapter buffer between opponents.

### 3.7 Data pipeline (PokerBench)

- [src/world_model/data.py](src/world_model/data.py)
  - `encode_observation` — packs hole cards, board texture (flush draw,
    straight connectivity, paired, suited), pot odds, SPR, street, position,
    and aggression proxies into the 30-D feature vector.
  - `encode_action` — `[action_type/3, bet_ratio, log1p(tta), is_allin]` for
    RSSM input.
  - `encode_opponent_action` — 6-D feature vector for the adapter.
  - `classify_opponent_type` — heuristic VPIP / aggression-factor / TTA
    classifier into `{tight,loose} × {passive,aggressive}`; provides labels
    for the contrastive loss.
  - `HandRecord` / `TrainingSequence` — dataclasses for raw and processed
    hands.
  - `PokerBenchDataset._process_record` — replays a hand to produce
    `(observations, actions, rewards, opponent_actions, opponent_type)`,
    bookkeeping pot/stack/board/bet_facing along the way. Reward is non-zero
    only at terminal.
  - `_classify_sizing` — bucketizes bet/pot ratio into
    small/medium/large/overbet for the adapter's `sizing_pattern` channel.
  - `collate_sequences` — pads to `(B, T, ·)` and `(B, T_opp, ·)` with
    boolean masks.
  - `create_dataloader` — convenience constructor.

### 3.8 Losses

- [src/world_model/losses.py](src/world_model/losses.py)
  - `transition_loss` — `L_recon (MSE on obs) + L_reward (MSE) +
    kl_scale · L_kl`. KL uses categorical posteriors/priors with
    **KL balancing** (`0.8 · KL[q‖p] + 0.2 · KL[p‖q]`) to prevent posterior
    collapse while keeping the prior informative.
  - `contrastive_adapter_loss` — InfoNCE over `(anchor, positive,
    negatives)`: cosine-normalized embeddings, positive = same opponent type,
    negatives = different types. Encourages `z_opp` to cluster by play
    style.
  - `policy_loss` — actor-critic in imagination using λ-returns
    (`_compute_lambda_returns`, λ=0.95, computed in reverse). Advantage =
    `R_λ - V` (detached, normalized). Actor loss combines discrete
    log-prob and continuous bet log-prob with the advantage; critic loss is
    MSE against `R_λ`.

### 3.9 Training entry point

- [src/world_model/train.py](src/world_model/train.py) wires `data.py`,
  `agent.py`, and `losses.py` into a training loop using `WorldModelConfig`.

---

## 4. How the two approaches relate

| Aspect          | ISMCTS (Approach A)                       | World Model (Approach B)                                  |
|-----------------|-------------------------------------------|-----------------------------------------------------------|
| Hidden info     | Sampled per iteration (`belief.py`)       | Absorbed into stochastic latent `z_t` and `z_opp`         |
| Transition      | Real env (`environment.apply_action`)     | Learned RSSM (`rssm.imagine_step`)                        |
| Opponent model  | Implicit (uniform over hands)             | Explicit, online (`OpponentAdapter` + buffer)             |
| Search          | UCB1 over a shared tree, K iterations     | K parallel latent rollouts, horizon H (MPC)               |
| Leaf value      | Equity + pot-odds heuristic               | Learned `ValueHead`                                       |
| Adaptation      | None — fixed search                       | Real-time via `z_opp`; offline via training               |
| Cost / decision | ~50–200 ms (depends on iterations)        | `K × H` forward passes through small networks             |

The two are designed to be swappable behind the same observation/action
contract, so the ISMCTS agent serves as a baseline and a safety net while the
world-model agent is being trained and stress-tested.

---

## 5. Tests and scripts

- [tests/](tests/) — unit tests for the engine (`test_cards`, `test_actions`,
  `test_evaluator`, `test_environment`, `test_belief`, `test_value`,
  `test_ismcts`) and for each world-model component
  ([tests/world_model/](tests/world_model/): `test_rssm`, `test_heads`,
  `test_adapter`, `test_data`, `test_losses`, `test_search`, `test_agent`).
- [scripts/play_random.py](scripts/play_random.py) — self-play sanity check.
- [scripts/play_human.py](scripts/play_human.py) — interactive CLI vs. the
  agent.
