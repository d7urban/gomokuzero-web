#!/usr/bin/env python3
"""GomokuZero Web — stateless API for Vercel deployment.

All game state lives on the client.  The server provides two compute
endpoints (AI move and analysis) plus serves the static HTML.
"""

import os
import sys
import threading

import numpy as np

try:
    from flask import Flask, jsonify, render_template, request
except ImportError:
    print("Flask not found. Install with:  pip install flask", file=sys.stderr)
    sys.exit(1)

from gomoku import (
    BOARD_SIZE,
    GomokuGame,
    mcts_search_batched,
    mcts_policy,
)

# ---------------------------------------------------------------------------
# TFLite model (loaded once per cold start)
# ---------------------------------------------------------------------------

MODEL_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "gomoku_best.tflite"
)
MCTS_BATCH = 32
MAX_SIMS = 5000


def _load_model():
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_path=MODEL_FILE)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    outs = interp.get_output_details()

    if outs[0]["shape"][-1] == BOARD_SIZE * BOARD_SIZE:
        logits_idx, value_idx = 0, 1
    else:
        logits_idx, value_idx = 1, 0

    lock = threading.Lock()

    def predict_fn(batch):
        batch = np.asarray(batch, dtype=np.float32)
        with lock:
            interp.resize_tensor_input(inp["index"], batch.shape)
            interp.allocate_tensors()
            interp.set_tensor(inp["index"], batch)
            interp.invoke()
            logits = interp.get_tensor(outs[logits_idx]["index"])
            values = interp.get_tensor(outs[value_idx]["index"])
        return logits.copy(), values.copy()

    from gomoku import NUM_INPUT_PLANES
    predict_fn(
        np.zeros((1, BOARD_SIZE, BOARD_SIZE, NUM_INPUT_PLANES),
                 dtype=np.float32)
    )
    return predict_fn


predict_fn = _load_model()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reconstruct_game(move_history):
    """Replay a move list [[r,c], ...] into a GomokuGame."""
    game = GomokuGame()
    for move in move_history:
        r, c = int(move[0]), int(move[1])
        reward, done = game.make_move(r, c)
        if reward == -1:
            raise ValueError(f"Illegal move ({r},{c}) in history")
    return game


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("play_web.html")


@app.route("/api/ai_move", methods=["POST"])
def api_ai_move():
    """Compute AI's next move.

    Request:  {move_history: [[r,c],...], sims: 500}
    Response: {row, col, eval}
    """
    data = request.get_json(force=True)
    try:
        game = _reconstruct_game(data.get("move_history", []))
    except ValueError as e:
        return jsonify({"error": str(e)})

    sims = max(1, min(int(data.get("sims", 500)), MAX_SIMS))

    root = mcts_search_batched(
        game, predict_fn,
        num_simulations=sims,
        batch_size=MCTS_BATCH,
        c_puct=1.5,
        add_noise=False,
    )
    pi = mcts_policy(root, temperature=0.05)
    idx = int(np.argmax(pi))
    row, col = divmod(idx, BOARD_SIZE)

    return jsonify({
        "row": row,
        "col": col,
        "eval": round(float(root.q_value), 4),
    })


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Run MCTS analysis on the current position.

    Request:  {move_history: [[r,c],...], sims: 500}
    Response: {policy, q_vals, root_q, sims_done}
    """
    data = request.get_json(force=True)
    try:
        game = _reconstruct_game(data.get("move_history", []))
    except ValueError as e:
        return jsonify({"error": str(e)})

    sims = max(1, min(int(data.get("sims", 500)), MAX_SIMS))

    root = mcts_search_batched(
        game, predict_fn,
        num_simulations=sims,
        batch_size=MCTS_BATCH,
        c_puct=1.5,
        add_noise=False,
    )

    counts = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
    q_vals = np.full(BOARD_SIZE * BOARD_SIZE, np.nan, dtype=np.float32)
    for (r, c), child in root.children.items():
        counts[r * BOARD_SIZE + c] = child.visit_count
        if child.visit_count:
            q_vals[r * BOARD_SIZE + c] = -child.q_value

    return jsonify({
        "policy": [round(float(v), 1) for v in counts],
        "q_vals": [
            None if np.isnan(v) else round(float(v), 4)
            for v in q_vals
        ],
        "root_q": round(float(root.q_value), 4),
        "sims_done": int(sims),
    })


# ---------------------------------------------------------------------------
# Local dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GomokuZero Web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(f"\n  GomokuZero Web UI: http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
