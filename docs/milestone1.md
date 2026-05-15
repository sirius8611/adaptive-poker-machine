# Adaptive Poker Machine — Milestone 1 Report

**Course / Project:** Adaptive Poker Machine for Heads-Up No-Limit Texas Hold'em
**Milestone:** 1 — Updated Proposal & Progress Report
**Date:** 2026-05-04

---

## 1. Introduction and Motivation

Heads-Up No-Limit Texas Hold'em (HUNL) is the canonical benchmark for
imperfect-information game AI: the state space is enormous, information is
hidden, and strong play requires reasoning about what the opponent *might*
hold and what they *tend* to do. Solver-based agents (Libratus, DeepStack,
Pluribus) achieve superhuman play, but they are static — they compute a
near-equilibrium policy offline and do not adapt online to a specific
opponent's tendencies.

The original proposal targeted a *different* slice of the problem:
**fast online adaptation against a human opponent**, while keeping a
deterministic search baseline as a safety net. Since the proposal we have
substantially refined the scope. Specifically, we now build two complementary
agents that share a single environment and observation/action contract:

1. A classical **Information Set Monte Carlo Tree Search (ISMCTS)** agent that
   plans by sampling possible worlds (determinizations) and searching each
   one. This is our deterministic baseline / fallback.
2. A learned **World Model** agent (Recurrent State-Space Model + Opponent
   Adapter) that plans in latent space and adapts online via a per-opponent
   embedding `z_opp` updated in real time, without retraining.

This is an expansion over the proposal in two ways. First, the proposal
described a single end-to-end model with adaptation only as a stretch goal;
we now treat adaptation as a first-class objective with an explicit
mechanism (`OpponentAdapter`, contrastive training). Second, we now
maintain both agents behind one contract so the ISMCTS agent is a
production-quality fallback, not just a stepping-stone.

---

## 2. Problem Statement

Given a HUNL hand played at fixed blinds and starting stack, the agent
receives a partial-information observation at every decision point:

- Its own two hole cards.
- The public state (board, pot, stacks, street, action history,
  amount-to-call, current player).

It must output a legal action drawn from a discrete space of size 10:
`fold, check, call`, six pot-fraction raises (0.33, 0.5, 0.75, 1.0, 1.5,
2.0×pot), and `all_in`.

We measure success along three axes:

- **Strength against a fixed baseline** (random and a tight-passive bot):
  expected mBB/hand over a long match.
- **Adaptation gain**: improvement in EV after `N` hands of observation
  versus the same agent with `z_opp` reset.
- **Decision latency**: target ≤ 200 ms per decision so the agent can play
  in real time.

The original proposal stated only the first axis. The second and third are
new constraints introduced to keep the system honest about its "adaptive,
real-time" claim.

---

## 3. Approach

The system has three layers: a shared poker engine, the ISMCTS agent, and
the World Model agent.

### 3.1 Shared poker engine

`src/poker/` implements a deterministic HUNL environment with an
OpenSpiel-style API. Cards are encoded as integers in `[0, 52)` via
`rank*4 + suit`. State transitions are pure functions: randomness is
resolved at deal time by pre-shuffling the deck, so `apply_action` never
samples. `HUNLEnvironment` exposes `new_initial_state`, `apply_action`,
`get_legal_actions`, `get_rewards`, and `get_observation`. Hand strength
is delegated to the `treys` evaluator. This shared substrate guarantees
that both agents see identical legality, reward, and observation
semantics — a precondition for a fair internal comparison.

### 3.2 Approach A — ISMCTS

The pipeline is:

```
Observation → Determinize → ISMCTS search → Most-visited action
```

- **Determinization** (`belief.resample_history`) samples a fully-specified
  state consistent with the observation: hero's cards and the board are
  fixed; the opponent's cards and the remaining deck are sampled uniformly
  from unseen cards.
- **Search** (`ismcts.ISMCTS.search`) runs `N` iterations (default 1000).
  Each iteration draws a new determinization and walks a *shared* tree
  using UCB1, but only along edges legal in the current sample. Untried
  legal actions are expanded; leaves are evaluated by exact reward
  (terminal), a heuristic value function (depth-limited), or a random
  rollout.
- **Value function** (`value.value_function`) blends Monte-Carlo equity
  (or exact equity at the river) with a pot-odds term:
  `0.7·equity + 0.3·(equity·pot/max_payoff)`.

The final action is the **most visited** root child, which is more robust
than mean value under noisy leaf estimates. This baseline was specified
in the proposal; the refinement since then is the constraint that the
ISMCTS agent must respect the *real* legal-action set at the root, even
when raise-bucket deduplication differs between the determinized and
real states (handled via the `real_legal_actions` argument in
`search`).

### 3.3 Approach B — World Model agent

The World Model agent replaces card sampling with a learned latent
transition model conditioned on a per-opponent embedding. `src/world_model/`
contains the full stack.

**Recurrent State-Space Model (`rssm.py`).** The latent state is
`s_t = (h_t, z_t)`: a deterministic GRU recurrence `h_t = f(h_{t-1},
z_{t-1}, a_{t-1}, z_opp)` plus a categorical stochastic latent
`z_t ∈ {C classes × K categories}`. The RSSM has both a prior
`p(z_t | h_t)` (used during imagination) and a posterior
`q(z_t | h_t, o_t)` (used during teacher-forced training). Categorical
sampling uses straight-through Gumbel-Softmax. Auxiliary heads predict
the next observation and a scalar EV reward.

**Opponent Adapter (`adapter.py`).** A causal Transformer encoder maps a
window of the opponent's last 32 actions to `z_opp ∈ ℝ^64`. Each action
carries six features:
`[action_type, bet_ratio, log1p(tta), sizing_pattern, street, position]`,
which is critical: it captures not just *what* the opponent did but
*how* (Time-to-Action, sizing tendency). At inference, an
`OnlineAdapterState` rolling buffer is updated after every opponent
action, so `z_opp` evolves in real time without touching model weights —
this is the **fast-path online adaptation** mechanism that distinguishes
this approach from a static solver.

**Policy and Value heads (`heads.py`).** The policy emits a 4-way
categorical over action types and a `TanhNormal` over bet ratios in
`[0, 1]`. The value head is an MLP critic over the RSSM features.

**Latent Look-Ahead search (`search.py`).** At decision time,
`LatentLookAhead.search` runs `K = 64` parallel rollouts of horizon
`H = 8` in latent space, holding `z_opp` fixed. Returns are accumulated
from the RSSM reward head with a value-bootstrap at the horizon, and the
first action of the highest-return trajectory is selected (with a
legal-action mask applied at `t=0`). This is structurally MPC: search
cost is `K·H` small forward passes, not `iterations × game_depth` calls
into the real environment.

**Training (`losses.py`, `train.py`).** Three objectives are jointly
minimized on PokerBench-style data:

- *Transition loss*: MSE on observation reconstruction + MSE on reward +
  KL between prior and posterior, with **KL balancing**
  (`0.8·KL[q‖p] + 0.2·KL[p‖q]`) to prevent posterior collapse.
- *Contrastive adapter loss*: InfoNCE that pulls embeddings of
  same-style opponents together and pushes embeddings of different
  styles apart, using VPIP / aggression-factor / TTA heuristics to
  produce labels (`tight/loose × passive/aggressive`).
- *Policy loss*: actor-critic in *imagination* with λ-returns
  (λ=0.95, γ=0.99) and a normalized-advantage policy gradient.

The data pipeline (`data.py`) replays hand histories into 30-D
observation vectors (hole cards, board texture features, pot odds, SPR,
street, position, aggression proxies) and produces aligned
`(observations, actions, rewards, opponent_actions, opponent_type)`
sequences with appropriate padding masks.

### 3.4 How the two agents relate

| Aspect          | ISMCTS                                | World Model                                  |
|-----------------|---------------------------------------|----------------------------------------------|
| Hidden info     | Sampled per iteration                 | Absorbed into `z_t`, `z_opp`                 |
| Transition      | Real environment                      | Learned RSSM                                 |
| Opponent model  | Implicit / uniform                    | Explicit, online via Transformer adapter     |
| Search          | UCB1 tree, K iterations               | K parallel latent rollouts (MPC)             |
| Leaf value      | Equity + pot-odds heuristic           | Learned `ValueHead`                          |
| Adaptation      | None                                  | Real-time via `z_opp`; offline via training  |

Both agents are swappable behind the same `Observation`/action contract.

---

## 4. Progress Since the Proposal

The proposal listed four phases. We are partway through Phase 2.

**Phase 1 — Baseline & test environment (complete).**
Implemented the full HUNL engine (`cards`, `actions`, `state`,
`environment`, `evaluator`) with deterministic transitions, plus a unit
test suite covering legal-action generation, betting bookkeeping (min
raise, all-in, side-pot-free terminal payoffs), and hand evaluation.
Built `play_random.py` and `play_human.py` scripts for self-play and
interactive testing.

**Phase 2 — Baseline model + world model (in progress).**
- Implemented and unit-tested the ISMCTS agent end-to-end
  (`belief.resample_history`, `ismcts.ISMCTS`, `value.value_function`).
  The agent runs at ~100 ms / decision at 1000 iterations.
- Implemented every component of the World Model stack: `RSSM`,
  `OpponentAdapter` + `OnlineAdapterState`, `PolicyHead` (with
  `TanhNormal`), `ValueHead`, `LatentLookAhead`, the three loss
  functions, and the PokerBench data loader. All have dedicated unit
  tests under `tests/world_model/`.
- Wired everything into `WorldModelAgent.forward_train` /
  `imagine_trajectories` and a stateful `LiveAgent` for play.
- Wrote architecture documentation (`docs/architecture.md`,
  `docs/ai_pipeline.md`).

**Phase 3 — UI / integration (not yet started).**
We will reuse `play_human.py` as a CLI front-end; a richer UI is
deferred until model quality is confirmed.

**Phase 4 — Stress test & optimize (not yet started).**

### 4.1 Changes from the original proposal

- *Adaptation moved from stretch goal to first-class objective.* The
  Opponent Adapter and contrastive loss were not in the proposal; they
  are now central.
- *Two-agent architecture.* The proposal described a single model. We
  now keep the ISMCTS baseline as a permanent fallback so the system
  has a known-good failure mode while the World Model is still
  training.
- *Decision-latency budget added.* Imposed to make "real-time
  adaptation" a measurable claim rather than a slogan.
- *PokerBench-style supervised pretraining.* Originally we planned
  pure self-play; we now pretrain the RSSM on solver-style data first
  to bootstrap the transition model, which is much more sample-
  efficient.

### 4.2 Open risks

- **Opponent-type labels are heuristic.** The contrastive loss depends
  on the VPIP/AF classifier in `data.classify_opponent_type`. If the
  heuristic mislabels, the adapter learns a noisy embedding. Mitigation:
  evaluate `z_opp` clustering quality directly and consider unsupervised
  contrastive variants.
- **Sim-to-real gap on TTA.** Training data has synthetic Time-to-Action
  values; real opponents will not match the distribution. Mitigation:
  test ablations with TTA zeroed-out, and collect a small live-play
  calibration set.
- **Latent rollout drift.** With `H=8` and a learned dynamics model,
  errors compound. Mitigation: keep horizon short, use the value
  bootstrap at the horizon, and consider `search_with_averaging` for
  more robust action selection.

---

## 5. Experimental Plan (next milestone)

- *E1 — Baseline strength.* ISMCTS vs random (target +200 mBB/hand) and
  ISMCTS vs tight-passive bot.
- *E2 — World Model strength.* Trained World Model vs random; vs ISMCTS
  baseline (target: ≥ ISMCTS within latency budget).
- *E3 — Adaptation gain.* Same World Model agent with `z_opp` reset
  vs `z_opp` warmed up over `N ∈ {10, 50, 200}` hands against a fixed
  exploitable opponent (loose-passive). Expect monotone improvement in
  EV.
- *E4 — Ablations.* Remove TTA channel; remove contrastive loss; remove
  `z_opp` (constant zero); pure-policy vs `LatentLookAhead`.
- *E5 — Latency.* Median and p95 decision time for both agents on the
  same hardware.

---

## 6. Team Contributions

> *Replace the placeholder names with the actual team members. Keep the
> contribution descriptions specific — list files, modules, or
> experiments owned, not generic verbs.*

- **[Member A] — Poker engine & ISMCTS baseline.**
  Implemented `src/poker/{cards,actions,state,environment,evaluator}.py`
  and the corresponding tests. Implemented `belief.resample_history`,
  the `ISMCTS` searcher, and the heuristic `value_function`. Authored
  the `play_random.py` and `play_human.py` scripts. Owner of the
  decision-latency benchmark.

- **[Member B] — World Model core (RSSM + heads).**
  Implemented `src/world_model/rssm.py` (deterministic GRU + categorical
  stochastic latent, prior/posterior nets, reward and decoder heads),
  `heads.py` (`TanhNormal`, `PolicyHead`, `ValueHead`), and the
  corresponding unit tests. Tuned the KL-balancing coefficient.

- **[Member C] — Opponent Adapter & contrastive learning.**
  Implemented `adapter.py` (`OpponentAdapter`, positional encoding,
  `OnlineAdapterState` rolling buffer) and the
  `contrastive_adapter_loss` in `losses.py`. Designed the 6-D opponent
  action feature (TTA, sizing pattern, position, street). Owner of the
  adaptation-gain experiment.

- **[Member D] — Search, training pipeline, and data.**
  Implemented `LatentLookAhead.search` and `search_with_averaging`,
  `transition_loss`, `policy_loss` with λ-returns, and the PokerBench
  data pipeline (`data.py`, `train.py`). Wrote
  `WorldModelAgent.forward_train` / `imagine_trajectories` and the
  `LiveAgent` wrapper.

- **[All members] — Documentation & review.**
  Co-authored `docs/architecture.md` and `docs/ai_pipeline.md`. Pair-
  reviewed every PR; ran each other's unit tests before merge.

---

## 7. Conclusion

We have a working ISMCTS baseline and a complete, unit-tested
implementation of every component of the World Model stack. The remaining
work for the next milestone is end-to-end training on PokerBench data,
the five planned experiments, and integration of the World Model agent
behind the same CLI used for the baseline. The architectural commitment
to a single observation/action contract for both agents lets us measure
progress directly: the World Model's adaptation gain is whatever EV it
adds *over* the static ISMCTS baseline on the same hands.
