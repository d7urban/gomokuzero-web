#!/usr/bin/env python3
"""Gomoku - Play with a Web UI.

Feature-equivalent to play_qt.py: Human-vs-AI, Human-vs-Human, continuous
analysis heatmap, undo, save/load, difficulty and side selection.

Requires: pip install flask
"""

import os
import sys
import threading
import time

import numpy as np

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

try:
    from flask import Flask, jsonify, render_template, request
except ImportError:
    print("Flask not found. Install with:  pip install flask", file=sys.stderr)
    sys.exit(1)

from gomoku import (
    BOARD_SIZE,
    EMPTY,
    PLAYER1,
    PLAYER2,
    WIN_LENGTH,
    GomokuGame,
    mcts_begin,
    mcts_expand_root,
    mcts_process_results,
    mcts_select_leaves,
)
from entrypoint_shared import (
    AIPlayer,
    AI_MCTS_BATCH,
    AI_SIMULATIONS,
    load_model_and_predict_fn,
    resolve_difficulty,
)

WEIGHTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "gomoku_best.weights.h5"
)

ANALYSIS_MAX_SIMS = 1_000_000
ANALYSIS_EMIT_INTERVAL_SEC = 0.2

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Analysis worker (background MCTS pondering)
# ---------------------------------------------------------------------------

class AnalysisWorker(threading.Thread):
    """Background MCTS pondering thread with double-buffer GPU pipeline."""

    def __init__(self, session_id, predict_fn, game,
                 batch_size=AI_MCTS_BATCH, c_puct=1.5,
                 max_sims=ANALYSIS_MAX_SIMS):
        super().__init__(daemon=True)
        self.session_id = session_id
        self.predict_fn = predict_fn
        self.game = game.copy()
        self.batch_size = batch_size
        self.c_puct = c_puct
        self.max_sims = max_sims
        self._running = True
        self._lock = threading.Lock()
        self.policy = None
        self.q_vals = None
        self.root_q = 0.0
        self.sims_done = 0

    def stop(self):
        self._running = False

    def get_snapshot(self):
        with self._lock:
            if self.policy is None:
                return None, None, 0.0, 0
            return (self.policy.copy(), self.q_vals.copy(),
                    self.root_q, self.sims_done)

    def _update_snapshot(self, ctx):
        root = ctx["root"]
        counts = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
        q_vals = np.full(BOARD_SIZE * BOARD_SIZE, np.nan, dtype=np.float32)
        for (r, c), child in root.children.items():
            counts[r * BOARD_SIZE + c] = child.visit_count
            if child.visit_count:
                q_vals[r * BOARD_SIZE + c] = -child.q_value
        with self._lock:
            self.policy = counts
            self.q_vals = q_vals
            self.root_q = float(root.q_value)
            self.sims_done = int(ctx["sims_done"])

    def run(self):
        try:
            ctx, root_state = mcts_begin(
                self.game, num_simulations=self.max_sims,
                batch_size=self.batch_size, c_puct=self.c_puct,
                add_noise=False,
            )
            root_batch = np.array([root_state], dtype=np.float32)
            logits_np, values_np = self.predict_fn(root_batch)
            mcts_expand_root(ctx, logits_np[0], values_np.ravel()[0])

            _raw = getattr(self.predict_fn, "_raw", None)
            last_emit = 0.0

            if _raw is not None:
                self._run_double_buffer(_raw, ctx, last_emit)
            else:
                self._run_sync(ctx, last_emit)

            self._update_snapshot(ctx)
        except Exception as e:
            print(f"Analysis worker error: {e}", file=sys.stderr)

    def _run_double_buffer(self, _raw, ctx, last_emit):
        slot_has_data = False
        slot_pending = slot_eval_list = slot_n_batch = None
        slot_tf_logits = slot_tf_values = None

        while self._running and ctx["sims_done"] < ctx["sims_target"]:
            leaf_states = mcts_select_leaves(ctx)
            cur_pending = ctx["pending"]
            cur_eval_list = ctx["eval_list"]
            cur_n_batch = ctx["n_batch"]
            if leaf_states:
                cur_tf_logits, cur_tf_values = _raw(
                    np.array(leaf_states, dtype=np.float32))
            else:
                cur_tf_logits = cur_tf_values = None

            if slot_has_data:
                ctx["pending"] = slot_pending
                ctx["eval_list"] = slot_eval_list
                ctx["n_batch"] = slot_n_batch
                if slot_tf_logits is not None:
                    mcts_process_results(
                        ctx, slot_tf_logits.numpy(),
                        slot_tf_values.numpy().ravel())
                else:
                    mcts_process_results(ctx)
                now = time.time()
                if now - last_emit >= ANALYSIS_EMIT_INTERVAL_SEC:
                    self._update_snapshot(ctx)
                    last_emit = now

            slot_pending = cur_pending
            slot_eval_list = cur_eval_list
            slot_n_batch = cur_n_batch
            slot_tf_logits = cur_tf_logits
            slot_tf_values = cur_tf_values
            slot_has_data = True

        # Drain last batch.
        if slot_has_data:
            ctx["pending"] = slot_pending
            ctx["eval_list"] = slot_eval_list
            ctx["n_batch"] = slot_n_batch
            if slot_tf_logits is not None:
                mcts_process_results(
                    ctx, slot_tf_logits.numpy(),
                    slot_tf_values.numpy().ravel())
            else:
                mcts_process_results(ctx)

    def _run_sync(self, ctx, last_emit):
        while self._running and ctx["sims_done"] < ctx["sims_target"]:
            leaf_states = mcts_select_leaves(ctx)
            if leaf_states:
                batch = np.array(leaf_states, dtype=np.float32)
                l_np, v_np = self.predict_fn(batch)
                mcts_process_results(ctx, l_np, v_np.ravel())
            else:
                mcts_process_results(ctx)
            now = time.time()
            if now - last_emit >= ANALYSIS_EMIT_INTERVAL_SEC:
                self._update_snapshot(ctx)
                last_emit = now


# ---------------------------------------------------------------------------
# Game server (single-session state + thread management)
# ---------------------------------------------------------------------------

class GameServer:
    def __init__(self):
        self.lock = threading.Lock()
        self.game = GomokuGame()

        print(f"  Loading weights: {WEIGHTS_FILE}")
        self.model, self.predict_fn = load_model_and_predict_fn(WEIGHTS_FILE)
        self.ai = AIPlayer(
            self.predict_fn,
            simulations=AI_SIMULATIONS,
            difficulty="medium",
        )

        self.human_player = PLAYER1
        self.human_only_mode = False
        self.game_over = False
        self.human_turn = True
        self.message = "Your turn (Black)."
        self.difficulty_label = "medium"
        self.difficulty_sims = AI_SIMULATIONS

        self.analysis_enabled = False
        self.analysis_worker = None
        self.analysis_session_id = 0
        self.analysis_started_at = 0.0

        self.ai_thinking = False
        self._ai_turn_gen = 0

    # -- State snapshot for the client --

    def get_state(self):
        with self.lock:
            last_move = None
            if self.game.move_history:
                lm = self.game.move_history[-1]
                last_move = [int(lm[0]), int(lm[1])]

            winning = self._winning_line_cells()

            out = {
                "board": self.game.board.tolist(),
                "board_size": int(BOARD_SIZE),
                "current_player": int(self.game.current_player),
                "human_player": int(self.human_player),
                "human_turn": bool(self.human_turn),
                "game_over": bool(self.game_over),
                "human_only_mode": bool(self.human_only_mode),
                "message": self.message,
                "move_count": len(self.game.move_history),
                "last_move": last_move,
                "winning_cells": [list(c) for c in winning],
                "difficulty": f"{self.difficulty_label} ({self.difficulty_sims} sims)",
                "ai_thinking": bool(self.ai_thinking),
                "analysis_enabled": bool(self.analysis_enabled),
                "analysis": None,
            }

            if self.analysis_enabled and self.analysis_worker is not None:
                policy, q_vals, root_q, sims_done = \
                    self.analysis_worker.get_snapshot()
                if policy is not None:
                    elapsed = max(1e-9, time.time() - self.analysis_started_at)
                    sps = sims_done / elapsed if sims_done > 0 else 0
                    q_list = [
                        None if np.isnan(v) else round(float(v), 4)
                        for v in q_vals
                    ]
                    out["analysis"] = {
                        "policy": [round(float(v), 1) for v in policy],
                        "q_vals": q_list,
                        "root_q": round(root_q, 4),
                        "sims_done": sims_done,
                        "sims_per_sec": round(sps, 1),
                    }
            return out

    def _winning_line_cells(self):
        if not self.game_over or not self.game.move_history:
            return []
        row, col, player = self.game.move_history[-1]
        row, col, player = int(row), int(col), int(player)
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return []
        if int(self.game.board[row, col]) != player:
            return []
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            cells = [(row, col)]
            rr, cc = row + dr, col + dc
            while (0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE
                   and int(self.game.board[rr, cc]) == player):
                cells.append((rr, cc))
                rr += dr
                cc += dc
            rr, cc = row - dr, col - dc
            while (0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE
                   and int(self.game.board[rr, cc]) == player):
                cells.insert(0, (rr, cc))
                rr -= dr
                cc -= dc
            if len(cells) >= WIN_LENGTH:
                return cells
        return []

    # -- Game actions --

    def new_game(self, side=None, mode=None):
        with self.lock:
            self._stop_analysis()
            self._ai_turn_gen += 1
            self.ai_thinking = False

            if mode == "human":
                self.human_only_mode = True
            elif mode == "ai":
                self.human_only_mode = False

            self.game = GomokuGame()
            self.game_over = False

            if self.human_only_mode:
                self.human_player = PLAYER1
                self.human_turn = True
                self.message = "New game (Human vs Human). Black plays first."
            else:
                self.human_player = (
                    PLAYER1 if side != "ai_first" else PLAYER2
                )
                self.human_turn = (
                    self.game.current_player == self.human_player
                )
                if not self.human_turn:
                    self.message = "New game. AI goes first..."
                    self._begin_ai_turn()
                    return {"ok": True}
                self.message = "New game. Your turn (Black)."

            self._restart_analysis()
            return {"ok": True}

    def make_move(self, row, col):
        with self.lock:
            if self.game_over:
                return {"error": "Game is over."}
            if self.ai_thinking:
                return {"error": "Wait for AI to finish."}
            if not self.human_only_mode and not self.human_turn:
                return {"error": "Not your turn."}
            if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
                return {"error": "Out of bounds."}
            if self.game.board[row, col] != EMPTY:
                return {"error": "Cell is occupied."}

            self._stop_analysis()
            reward, done = self.game.make_move(row, col)

            if done:
                self.game_over = True
                if reward == 1:
                    if self.human_only_mode:
                        w = ("Black" if self.game.current_player == PLAYER1
                             else "White")
                        self.message = f"{w} wins!"
                    else:
                        self.message = "You win!"
                else:
                    self.message = "Draw!"
                return {"ok": True}

            if self.human_only_mode:
                side = ("Black" if self.game.current_player == PLAYER1
                        else "White")
                self.message = f"{side} to play."
                self._restart_analysis()
                return {"ok": True}

            self.human_turn = False
            self.message = f"You played ({row},{col}). AI thinking..."
            self._begin_ai_turn()
            return {"ok": True}

    def undo(self):
        with self.lock:
            if self.ai_thinking:
                return {"error": "Wait for AI to finish."}

            self._stop_analysis()

            if self.human_only_mode:
                if self.game.undo_move():
                    self.game_over = False
                    self.message = "Move undone."
                else:
                    self.message = "Nothing to undo."
                self._restart_analysis()
                return {"ok": True}

            n = 1 if not self.human_turn else 2
            if len(self.game.move_history) < n:
                self.message = "Nothing to undo."
                self._restart_analysis()
                return {"ok": True}

            for _ in range(n):
                self.game.undo_move()
            self.game_over = False
            self.human_turn = (
                self.game.current_player == self.human_player
            )
            self.message = "Move undone."

            if not self.human_turn:
                self._begin_ai_turn()
            else:
                self._restart_analysis()
            return {"ok": True}

    def set_difficulty(self, difficulty, custom_sims=None):
        with self.lock:
            try:
                if difficulty == "custom" and custom_sims is not None:
                    label, sims = resolve_difficulty(str(int(custom_sims)))
                else:
                    label, sims = resolve_difficulty(difficulty)
            except ValueError as e:
                return {"error": str(e)}
            self.difficulty_label = label
            self.difficulty_sims = sims
            if self.ai is not None:
                self.ai.difficulty = label
                self.ai.sims = sims
            self.message = f"Difficulty: {label} ({sims} sims)."
            return {"ok": True}

    def toggle_analysis(self, enabled):
        with self.lock:
            self.analysis_enabled = bool(enabled)
            if enabled:
                self._restart_analysis()
            else:
                self._stop_analysis()
            return {"ok": True}

    def toggle_mode(self):
        with self.lock:
            if self.ai_thinking:
                return {"error": "Wait for AI to finish."}
            if self.human_only_mode:
                self.human_only_mode = False
                self.human_player = -self.game.current_player
                self.human_turn = False
                self._stop_analysis()
                if not self.game_over:
                    self.message = "Switched to Human vs AI. AI thinking..."
                    self._begin_ai_turn()
                else:
                    self.message = "Switched to Human vs AI."
            else:
                self.human_only_mode = True
                self.human_turn = True
                self.message = "Switched to Human vs Human."
                self._restart_analysis()
            return {"ok": True}

    def save_game(self):
        with self.lock:
            return {
                "format": "gomokuzero-web-save",
                "version": 1,
                "board_size": int(BOARD_SIZE),
                "saved_at": float(time.time()),
                "human_only_mode": bool(self.human_only_mode),
                "human_player": int(self.human_player),
                "game_over": bool(self.game_over),
                "move_history": [
                    [int(r), int(c), int(p)]
                    for r, c, p in self.game.move_history
                ],
                "difficulty_label": self.difficulty_label,
                "difficulty_sims": self.difficulty_sims,
                "analysis_enabled": self.analysis_enabled,
            }

    def load_game(self, payload):
        with self.lock:
            if self.ai_thinking:
                return {"error": "Wait for AI to finish."}
            try:
                if not isinstance(payload, dict):
                    raise ValueError("Invalid save format.")
                fmt = payload.get("format", "")
                if fmt not in ("gomokuzero-web-save", "gomokuzero-qt-save"):
                    raise ValueError(f"Unknown format: {fmt}")
                bs = int(payload.get("board_size", 0))
                if bs != BOARD_SIZE:
                    raise ValueError(
                        f"Board size {bs} != {BOARD_SIZE}.")

                game, game_over = self._restore_game(
                    payload.get("move_history", []))

                self._stop_analysis()
                self._ai_turn_gen += 1
                self.ai_thinking = False
                self.game = game
                self.game_over = game_over
                self.human_only_mode = bool(
                    payload.get("human_only_mode", False))
                self.human_player = int(
                    payload.get("human_player", PLAYER1))

                if "difficulty_label" in payload:
                    self.difficulty_label = str(payload["difficulty_label"])
                if "difficulty_sims" in payload:
                    self.difficulty_sims = int(payload["difficulty_sims"])
                    if self.ai:
                        self.ai.difficulty = self.difficulty_label
                        self.ai.sims = self.difficulty_sims

                self.analysis_enabled = bool(
                    payload.get("analysis_enabled", False))

                if self.human_only_mode:
                    self.human_turn = True
                else:
                    self.human_turn = (
                        self.game.current_player == self.human_player
                    )

                if (not self.human_only_mode and not self.game_over
                        and not self.human_turn):
                    self.message = "Game loaded. AI thinking..."
                    self._begin_ai_turn()
                else:
                    self.message = "Game loaded."
                    self._restart_analysis()

                return {"ok": True}
            except (ValueError, KeyError, TypeError) as e:
                return {"error": str(e)}

    @staticmethod
    def _restore_game(raw_history):
        if not isinstance(raw_history, list):
            raise ValueError("Invalid move_history.")
        game = GomokuGame()
        game_over = False
        for i, move in enumerate(raw_history):
            if not isinstance(move, (list, tuple)) or len(move) < 2:
                raise ValueError("Invalid move entry.")
            row, col = int(move[0]), int(move[1])
            if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
                raise ValueError("Move out of bounds.")
            reward, done = game.make_move(row, col)
            if reward == -1:
                raise ValueError("Illegal move in history.")
            if done:
                game_over = True
                if i != len(raw_history) - 1:
                    raise ValueError("Moves after game end.")
        return game, game_over

    # -- AI turn (background thread) --

    def _begin_ai_turn(self):
        """Start AI computation. Caller must hold self.lock."""
        if self.ai is None or self.game_over:
            return
        self._ai_turn_gen += 1
        gen = self._ai_turn_gen
        self.ai_thinking = True
        self.human_turn = False
        t = threading.Thread(
            target=self._run_ai_turn, args=(gen,), daemon=True)
        t.start()

    def _run_ai_turn(self, gen):
        try:
            with self.lock:
                if gen != self._ai_turn_gen:
                    return
                game_copy = self.game.copy()
                ai = self.ai
                if ai is None:
                    self.ai_thinking = False
                    return

            row, col, val = ai.get_move(game_copy)

            with self.lock:
                if gen != self._ai_turn_gen:
                    return
                if self.game.board[row, col] != EMPTY:
                    self.message = f"AI illegal move ({row},{col})."
                    self.ai_thinking = False
                    self.human_turn = True
                    return
                reward, done = self.game.make_move(row, col)
                self.ai_thinking = False
                if done:
                    self.game_over = True
                    tag = "AI wins!" if reward == 1 else "Draw!"
                    self.message = (
                        f"{tag} AI played ({row},{col}) "
                        f"eval {val:+.2f}"
                    )
                else:
                    self.human_turn = True
                    self.message = (
                        f"AI played ({row},{col}) eval {val:+.2f}. "
                        f"Your turn."
                    )
                self._restart_analysis()
        except Exception as e:
            with self.lock:
                if gen == self._ai_turn_gen:
                    self.message = f"AI error: {e}"
                    self.ai_thinking = False
                    self.human_turn = True

    # -- Analysis management --

    def _stop_analysis(self):
        """Stop worker. Caller must hold self.lock."""
        w = self.analysis_worker
        if w is not None:
            w.stop()
            self.analysis_worker = None
        self.analysis_session_id += 1

    def _restart_analysis(self):
        """Restart analysis if appropriate. Caller must hold self.lock."""
        self._stop_analysis()
        if not self.analysis_enabled:
            return
        if self.game_over:
            return
        if not self.human_turn and not self.human_only_mode:
            return
        self.analysis_session_id += 1
        self.analysis_started_at = time.time()
        worker = AnalysisWorker(
            session_id=self.analysis_session_id,
            predict_fn=self.predict_fn,
            game=self.game,
            batch_size=AI_MCTS_BATCH,
            c_puct=1.5,
        )
        self.analysis_worker = worker
        worker.start()


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

server = GameServer()


@app.route("/")
def index():
    return render_template("play_web.html")


@app.route("/api/state")
def api_state():
    return jsonify(server.get_state())


@app.route("/api/new_game", methods=["POST"])
def api_new_game():
    data = request.get_json(force=True)
    return jsonify(server.new_game(
        side=data.get("side"),
        mode=data.get("mode"),
    ))


@app.route("/api/move", methods=["POST"])
def api_move():
    data = request.get_json(force=True)
    return jsonify(server.make_move(int(data["row"]), int(data["col"])))


@app.route("/api/undo", methods=["POST"])
def api_undo():
    return jsonify(server.undo())


@app.route("/api/set_difficulty", methods=["POST"])
def api_set_difficulty():
    data = request.get_json(force=True)
    return jsonify(server.set_difficulty(
        data.get("difficulty", "medium"),
        data.get("custom_sims"),
    ))


@app.route("/api/toggle_analysis", methods=["POST"])
def api_toggle_analysis():
    data = request.get_json(force=True)
    return jsonify(server.toggle_analysis(data.get("enabled", False)))


@app.route("/api/toggle_mode", methods=["POST"])
def api_toggle_mode():
    return jsonify(server.toggle_mode())


@app.route("/api/save_game", methods=["POST"])
def api_save_game():
    return jsonify(server.save_game())


@app.route("/api/load_game", methods=["POST"])
def api_load_game():
    data = request.get_json(force=True)
    return jsonify(server.load_game(data))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="GomokuZero Web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"\n  GomokuZero Web UI: http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
