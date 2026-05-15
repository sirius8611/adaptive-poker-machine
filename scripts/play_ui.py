"""Gradio poker UI: play HUNL against ISMCTS or the World Model agent.

Usage (local):
    python scripts/play_ui.py --ckpt path/to/world_model.pt

Usage (Colab):
    from play_ui import build_ui
    ui = build_ui(world_model_ckpt=CKPT, cfg=cfg, device=device)
    ui.launch(share=True)
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from poker import actions as A
from poker.cards import card_to_str
from poker.environment import HUNLEnvironment
from poker.ismcts import ISMCTS
from poker.state import FLOP, PREFLOP, RIVER, TURN

STREET_NAMES = {PREFLOP: "Preflop", FLOP: "Flop", TURN: "Turn", RIVER: "River"}

SUIT_HTML = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}
SUIT_COLOR = {"c": "#222", "d": "#c0392b", "h": "#c0392b", "s": "#222"}


def card_html(card_int: int) -> str:
    s = card_to_str(card_int)
    rank, suit = s[0], s[1]
    color = SUIT_COLOR[suit]
    return (
        f'<span style="display:inline-block;min-width:42px;padding:6px 8px;margin:2px;'
        f'border:1px solid #999;border-radius:6px;background:#fff;'
        f'font-family:monospace;font-size:20px;color:{color};text-align:center;">'
        f'{rank}{SUIT_HTML[suit]}</span>'
    )


def hidden_card_html() -> str:
    return (
        '<span style="display:inline-block;min-width:42px;padding:6px 8px;margin:2px;'
        'border:1px solid #999;border-radius:6px;background:#34495e;'
        'font-family:monospace;font-size:20px;color:#34495e;text-align:center;">??</span>'
    )


def cards_html(cards, hide=False) -> str:
    if not cards:
        return '<span style="color:#888">—</span>'
    if hide:
        return "".join(hidden_card_html() for _ in cards)
    return "".join(card_html(c) for c in cards)


# ---------------------------------------------------------------------------
# Agent wrappers — uniform interface: agent.act(env, state, ai_player_id) -> action
# ---------------------------------------------------------------------------


class ISMCTSWrapper:
    def __init__(self, env: HUNLEnvironment, num_iterations: int = 200, seed: int | None = None):
        self.env = env
        self.iterations = num_iterations
        self.rng = random.Random(seed)

    def act(self, state, ai_player: int) -> int:
        obs = self.env.get_observation(state, ai_player)
        legal = self.env.get_legal_actions(state)
        searcher = ISMCTS(
            self.env, player_id=ai_player,
            num_iterations=self.iterations,
            rng=random.Random(self.rng.randint(0, 2**32)),
        )
        return searcher.search(obs, real_legal_actions=legal)

    def observe_opponent_action(self, *a, **k):
        pass

    def new_hand(self):
        pass


class WorldModelWrapper:
    """Bridges LiveAgent into the same interface as ISMCTSWrapper."""

    def __init__(self, ckpt_path: str, cfg, device):
        import torch
        from world_model.agent import LiveAgent, WorldModelAgent

        agent = WorldModelAgent(cfg).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        agent.load_state_dict(ckpt["agent_state_dict"])
        agent.eval()
        self.live = LiveAgent(agent, cfg, device=device, use_search=True)
        self.live.new_hand()

    def new_hand(self):
        self.live.new_hand()

    def observe_opponent_action(self, action_type: int, bet_ratio: float, tta: float,
                                sizing_pattern: int, street: int, position: int):
        self.live.observe_opponent_action(action_type, bet_ratio, tta, sizing_pattern, street, position)

    def act(self, state, ai_player: int) -> int:
        # Build observation in the LiveAgent's format and ask it to act.
        cp = ai_player
        hero_cards = state.hole_cards[cp]
        board = list(state.board)
        stack_size = 10000  # centiBB
        result = self.live.observe_and_act(
            hero_cards=hero_cards,
            board=board,
            pot=state.pot / 100,
            stack=state.stacks[cp] / 100,
            street=state.street,
            position=cp,
            bet_facing=state.bet_to_call / 100,
            num_actions_street=state.num_actions_this_street,
            stack_size=stack_size / 100,
            legal_action_types=None,
        )
        # Map (action_type, bet_ratio) → concrete legal action ID
        legal = state  # placeholder, computed below
        from poker.environment import HUNLEnvironment as _Env  # avoid unused warning
        env = _Env()
        legal = env.get_legal_actions(state)
        atype = result["action_type"]
        bet_ratio = result["bet_ratio"]

        # 0=fold, 1=check, 2=call, 3=raise
        if atype == 0 and A.FOLD in legal:
            return A.FOLD
        if atype == 1 and A.CHECK in legal:
            return A.CHECK
        if atype == 2 and A.CALL in legal:
            return A.CALL
        if atype == 3:
            # Pick the raise bucket closest to bet_ratio * stack
            target_amt = bet_ratio * state.stacks[cp]
            best, best_diff = None, float("inf")
            for a in legal:
                if a == A.ALL_IN:
                    amt = state.stacks[cp]
                elif A.RAISE_33 <= a <= A.RAISE_200:
                    chips_after_call = state.stacks[cp] - min(state.bet_to_call, state.stacks[cp])
                    amt = A.raise_amount(a, state.pot, state.min_raise, chips_after_call)
                else:
                    continue
                d = abs(amt - target_amt)
                if d < best_diff:
                    best_diff, best = d, a
            if best is not None:
                return best
        # Fallback: pick check/call/fold in that order
        for fallback in (A.CHECK, A.CALL, A.FOLD):
            if fallback in legal:
                return fallback
        return legal[0]


# ---------------------------------------------------------------------------
# Session state — one per Gradio user
# ---------------------------------------------------------------------------


class Session:
    def __init__(self, agent_kind: str, ismcts_iterations: int,
                 world_model_ckpt: str | None, cfg, device):
        self.env = HUNLEnvironment()
        self.rng = random.Random()
        self.human_player = 1  # human is BB
        self.ai_player = 0
        self.state = None
        self.total_human = 0
        self.total_ai = 0
        self.hand_num = 0
        self.log: list[str] = []
        self.agent_kind = agent_kind
        self.ismcts_iterations = ismcts_iterations
        self.world_model_ckpt = world_model_ckpt
        self.cfg = cfg
        self.device = device
        self.agent = self._build_agent()
        self.start_new_hand()

    def _build_agent(self):
        if self.agent_kind == "world_model" and self.world_model_ckpt:
            try:
                return WorldModelWrapper(self.world_model_ckpt, self.cfg, self.device)
            except Exception as e:
                self.log.append(f"⚠️  World Model failed to load ({e}); falling back to ISMCTS.")
                self.agent_kind = "ismcts"
        return ISMCTSWrapper(self.env, num_iterations=self.ismcts_iterations)

    def start_new_hand(self):
        self.hand_num += 1
        self.state = self.env.new_initial_state(self.rng)
        self.agent.new_hand()
        self.log.append(f"--- Hand #{self.hand_num} dealt ---")
        # AI may need to act first preflop (SB acts first)
        self._let_ai_play_until_human_turn()

    def _action_label(self, action: int, cp: int) -> str:
        if A.RAISE_33 <= action <= A.RAISE_200:
            chips_after_call = self.state.stacks[cp] - min(self.state.bet_to_call, self.state.stacks[cp])
            amt = A.raise_amount(action, self.state.pot, self.state.min_raise, chips_after_call)
            return f"{A.action_to_str(action)} ({amt/100:.1f} BB)"
        if action == A.CALL:
            return f"call ({min(self.state.bet_to_call, self.state.stacks[cp])/100:.1f} BB)"
        if action == A.ALL_IN:
            return f"all-in ({self.state.stacks[cp]/100:.1f} BB)"
        return A.action_to_str(action)

    def _let_ai_play_until_human_turn(self):
        while (not self.env.is_terminal(self.state)
               and self.state.current_player == self.ai_player):
            action = self.agent.act(self.state, self.ai_player)
            label = self._action_label(action, self.ai_player)
            self.log.append(f"AI: {label}")
            self.state = self.env.apply_action(self.state, action)

    def human_act(self, action: int):
        if self.env.is_terminal(self.state):
            return
        legal = self.env.get_legal_actions(self.state)
        if action not in legal:
            self.log.append(f"⚠️  Illegal action attempted: {A.action_to_str(action)}")
            return
        label = self._action_label(action, self.human_player)
        self.log.append(f"You: {label}")

        # Encode for the agent's opponent adapter (if it has one)
        atype = 0 if action == A.FOLD else 1 if action == A.CHECK else 2 if action == A.CALL else 3
        if A.RAISE_33 <= action <= A.RAISE_200:
            chips_after_call = self.state.stacks[self.human_player] - min(
                self.state.bet_to_call, self.state.stacks[self.human_player]
            )
            amt = A.raise_amount(action, self.state.pot, self.state.min_raise, chips_after_call)
            bet_ratio = amt / max(self.state.stacks[self.human_player], 1)
        elif action == A.ALL_IN:
            bet_ratio = 1.0
        else:
            bet_ratio = 0.0
        sizing = 0 if bet_ratio < 0.2 else 1 if bet_ratio < 0.5 else 2 if bet_ratio < 1.0 else 3
        self.agent.observe_opponent_action(
            action_type=atype, bet_ratio=bet_ratio, tta=2.0,
            sizing_pattern=sizing, street=self.state.street, position=self.human_player,
        )

        self.state = self.env.apply_action(self.state, action)
        if not self.env.is_terminal(self.state):
            self._let_ai_play_until_human_turn()
        if self.env.is_terminal(self.state):
            self._handle_terminal()

    def _handle_terminal(self):
        r0, r1 = self.env.get_rewards(self.state)
        self.total_human += r1
        self.total_ai += r0
        if self.state.is_showdown:
            self.log.append(
                f"Showdown — You: {' '.join(card_to_str(c) for c in self.state.hole_cards[self.human_player])} "
                f"vs AI: {' '.join(card_to_str(c) for c in self.state.hole_cards[self.ai_player])}"
            )
        elif self.state.is_folded:
            who = "AI" if self.state.folding_player == self.ai_player else "You"
            self.log.append(f"{who} folded.")
        delta = r1 / 100
        if r1 > 0:
            self.log.append(f"🏆  You won +{delta:.1f} BB")
        elif r1 < 0:
            self.log.append(f"💀  You lost {delta:.1f} BB")
        else:
            self.log.append("Split pot.")


# ---------------------------------------------------------------------------
# View rendering
# ---------------------------------------------------------------------------


def render_view(s: Session) -> dict:
    st = s.state
    showdown_or_fold = s.env.is_terminal(st)
    show_ai_cards = showdown_or_fold and st.is_showdown

    ai_cards = cards_html(list(st.hole_cards[s.ai_player]), hide=not show_ai_cards)
    board = cards_html(list(st.board))
    your_cards = cards_html(list(st.hole_cards[s.human_player]))

    table_html = f"""
    <div style='font-family:system-ui;padding:14px;background:#1e8449;border-radius:14px;color:#fff;'>
        <div style='text-align:center;margin-bottom:10px;font-size:14px;opacity:0.85;'>
            Hand #{s.hand_num} · {STREET_NAMES[st.street]} · Pot {st.pot/100:.1f} BB
            · Score (you): {s.total_human/100:+.1f} BB
        </div>
        <div style='display:flex;justify-content:space-between;align-items:center;margin:12px 0;'>
            <div>
                <div style='font-size:13px;opacity:0.85;'>AI ({s.agent_kind}) · Stack {st.stacks[s.ai_player]/100:.1f} BB</div>
                <div>{ai_cards}</div>
            </div>
            <div style='font-size:13px;opacity:0.85;'>Bet to call: {st.bet_to_call/100:.1f} BB</div>
        </div>
        <div style='text-align:center;background:#196f3d;padding:10px;border-radius:10px;margin:6px 0;'>
            <div style='font-size:12px;opacity:0.8;margin-bottom:4px;'>Board</div>
            <div>{board}</div>
        </div>
        <div style='margin:12px 0;'>
            <div style='font-size:13px;opacity:0.85;'>You · Stack {st.stacks[s.human_player]/100:.1f} BB</div>
            <div>{your_cards}</div>
        </div>
    </div>
    """

    log_html = "<div style='font-family:monospace;font-size:13px;max-height:260px;overflow:auto;'>" + \
               "<br>".join(s.log[-40:]) + "</div>"

    legal = s.env.get_legal_actions(st) if not showdown_or_fold and st.current_player == s.human_player else []
    button_updates = {}
    button_map = {
        "fold": A.FOLD, "check": A.CHECK, "call": A.CALL,
        "raise_33": A.RAISE_33, "raise_50": A.RAISE_50, "raise_75": A.RAISE_75,
        "raise_100": A.RAISE_100, "raise_150": A.RAISE_150, "raise_200": A.RAISE_200,
        "all_in": A.ALL_IN,
    }
    for name, code in button_map.items():
        active = code in legal
        label = name.replace("_", " ")
        if active and A.RAISE_33 <= code <= A.RAISE_200:
            chips_after_call = st.stacks[s.human_player] - min(st.bet_to_call, st.stacks[s.human_player])
            amt = A.raise_amount(code, st.pot, st.min_raise, chips_after_call)
            label = f"{name.replace('_', ' ')} ({amt/100:.1f} BB)"
        elif active and code == A.CALL:
            label = f"call ({min(st.bet_to_call, st.stacks[s.human_player])/100:.1f} BB)"
        elif active and code == A.ALL_IN:
            label = f"all-in ({st.stacks[s.human_player]/100:.1f} BB)"
        button_updates[name] = gr.update(interactive=active, value=label)

    next_active = showdown_or_fold
    button_updates["next"] = gr.update(interactive=next_active)
    return table_html, log_html, button_updates


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------


def build_ui(world_model_ckpt: str | None = None, cfg=None, device=None,
             ismcts_iterations: int = 200) -> gr.Blocks:
    import torch
    if cfg is None:
        from world_model.config import WorldModelConfig
        cfg = WorldModelConfig()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with gr.Blocks(title="Adaptive Poker Machine") as ui:
        gr.Markdown("# 🃏 Adaptive Poker Machine — HUNL\nPlay heads-up no-limit hold'em against either agent.")

        with gr.Row():
            agent_kind = gr.Radio(
                ["ismcts", "world_model"],
                value="world_model" if world_model_ckpt else "ismcts",
                label="Opponent",
            )
            new_session_btn = gr.Button("Start new session", variant="primary")

        session_state = gr.State(None)

        table_view = gr.HTML()
        with gr.Row():
            fold_b = gr.Button("fold", interactive=False)
            check_b = gr.Button("check", interactive=False)
            call_b = gr.Button("call", interactive=False)
            allin_b = gr.Button("all in", interactive=False)
        with gr.Row():
            r33_b = gr.Button("raise 33", interactive=False)
            r50_b = gr.Button("raise 50", interactive=False)
            r75_b = gr.Button("raise 75", interactive=False)
            r100_b = gr.Button("raise 100", interactive=False)
            r150_b = gr.Button("raise 150", interactive=False)
            r200_b = gr.Button("raise 200", interactive=False)
        next_b = gr.Button("Next hand →", interactive=False)
        log_view = gr.HTML()

        outputs = [table_view, log_view, fold_b, check_b, call_b,
                   r33_b, r50_b, r75_b, r100_b, r150_b, r200_b, allin_b, next_b]
        button_order = ["fold", "check", "call",
                        "raise_33", "raise_50", "raise_75",
                        "raise_100", "raise_150", "raise_200", "all_in", "next"]

        def _emit(s: Session):
            table_html, log_html, btns = render_view(s)
            return [s, table_html, log_html] + [btns[k] for k in button_order]

        def start_new_session(kind):
            s = Session(kind, ismcts_iterations, world_model_ckpt, cfg, device)
            return _emit(s)

        def play(s: Session, action: int):
            s.human_act(action)
            return _emit(s)

        def next_hand(s: Session):
            s.start_new_hand()
            return _emit(s)

        out_all = [session_state] + outputs

        new_session_btn.click(start_new_session, [agent_kind], out_all)
        fold_b.click(lambda s: play(s, A.FOLD), [session_state], out_all)
        check_b.click(lambda s: play(s, A.CHECK), [session_state], out_all)
        call_b.click(lambda s: play(s, A.CALL), [session_state], out_all)
        r33_b.click(lambda s: play(s, A.RAISE_33), [session_state], out_all)
        r50_b.click(lambda s: play(s, A.RAISE_50), [session_state], out_all)
        r75_b.click(lambda s: play(s, A.RAISE_75), [session_state], out_all)
        r100_b.click(lambda s: play(s, A.RAISE_100), [session_state], out_all)
        r150_b.click(lambda s: play(s, A.RAISE_150), [session_state], out_all)
        r200_b.click(lambda s: play(s, A.RAISE_200), [session_state], out_all)
        allin_b.click(lambda s: play(s, A.ALL_IN), [session_state], out_all)
        next_b.click(next_hand, [session_state], out_all)

    return ui


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="World Model checkpoint path")
    parser.add_argument("--iterations", type=int, default=200, help="ISMCTS iterations")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    args = parser.parse_args()
    ui = build_ui(world_model_ckpt=args.ckpt, ismcts_iterations=args.iterations)
    ui.launch(share=args.share)


if __name__ == "__main__":
    main()
