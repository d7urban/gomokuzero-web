#!/usr/bin/env python3
"""
Shared helpers for interactive entry point scripts.

This module is intentionally focused on behavior used by both play.py and
play_qt.py so UI-specific logic can stay local to each front end.
"""

import glob
import os

import numpy as np

from gomoku import (
    BOARD_SIZE,
    NUM_INPUT_PLANES,
    create_model,
    make_predict_fn,
    mcts_policy,
    mcts_search_batched,
)

# Common play defaults.
AI_SIMULATIONS = 500
AI_MCTS_BATCH = 32
DIFFICULTY_SIMS = {
    "easy": 250,
    "medium": 500,
    "hard": 2000,
}

BEST_WEIGHTS = "weights/gomoku_best.weights.h5"
LATEST_WEIGHTS = "weights/gomoku_weights.weights.h5"
MODEL_ARCH_CANDIDATES = (
    (10, 128),
    (6, 128),
)


class AIPlayer:
    """AI that selects moves via MCTS backed by a trained network."""

    def __init__(self, predict_fn, simulations=AI_SIMULATIONS, difficulty="medium"):
        self.predict_fn = predict_fn
        self.sims = simulations
        self.difficulty = difficulty

    def get_move(self, game):
        root = mcts_search_batched(
            game,
            self.predict_fn,
            num_simulations=self.sims,
            batch_size=AI_MCTS_BATCH,
            c_puct=1.5,
            add_noise=False,  # no exploration noise during play
        )
        pi = mcts_policy(root, temperature=0.05)  # near-greedy
        idx = int(np.argmax(pi))
        row, col = divmod(idx, BOARD_SIZE)
        return row, col, root.q_value


def resolve_difficulty(value):
    """Resolve a difficulty setting to (label, simulations)."""
    key = (value or "").strip().lower()
    if key in DIFFICULTY_SIMS:
        return key, DIFFICULTY_SIMS[key]
    try:
        sims = int(value)
    except (TypeError, ValueError) as err:
        raise ValueError(
            "Invalid --difficulty. Use easy|medium|hard or a positive integer sims count."
        ) from err
    if sims <= 0:
        raise ValueError("Custom --difficulty sims must be a positive integer.")
    return "Custom", sims


def select_weights(mode="best", explicit_path=""):
    """Return (weights_path, label) from requested selection mode.

    Mode may be:
      - "best" (default): prefer best, then latest, then newest checkpoint
      - "latest": prefer latest, then newest checkpoint
      - "file": use explicit_path (validated)
    """
    mode = (mode or "").strip().lower()

    if mode == "file":
        wf = os.path.expanduser((explicit_path or "").strip())
        if not wf:
            raise ValueError("Select a .h5 file for 'Specific file' mode.")
        if not wf.endswith(".h5"):
            raise ValueError("Invalid weight file: expected a .h5 file.")
        if not os.path.isfile(wf):
            raise ValueError(f"Specified weight file does not exist: {wf}")
        return wf, "explicit"

    if mode == "latest":
        if os.path.exists(LATEST_WEIGHTS):
            return LATEST_WEIGHTS, "latest"
    else:
        if os.path.exists(BEST_WEIGHTS):
            return BEST_WEIGHTS, "best"
        if os.path.exists(LATEST_WEIGHTS):
            return LATEST_WEIGHTS, "latest (no best yet)"

    # Last resort: newest checkpoint file.
    files = glob.glob("weights/gomoku_*.weights.h5")
    if files:
        files.sort(key=os.path.getmtime, reverse=True)
        return files[0], "checkpoint"

    return None, None


def load_model_and_predict_fn(weight_file):
    """Create model, load weights, build predict_fn, and run one warmup call.

    Supports both legacy 6x128 and newer 10x128 checkpoint architectures.
    """
    errors = []
    model = None

    for num_res_blocks, num_filters in MODEL_ARCH_CANDIDATES:
        try:
            candidate = create_model(
                num_res_blocks=num_res_blocks,
                num_filters=num_filters,
            )
            candidate.load_weights(weight_file)
            model = candidate
            break
        except Exception as e:
            reason = str(e).splitlines()[0] if str(e) else type(e).__name__
            errors.append(
                f"{num_res_blocks}x{num_filters}: {reason}"
            )

    if model is None:
        details = "; ".join(errors) if errors else "no loader attempts"
        raise ValueError(
            f"Unable to load weights '{weight_file}'. Tried architectures: {details}"
        )

    predict_fn = make_predict_fn(model)
    predict_fn(
        np.zeros(
            (1, BOARD_SIZE, BOARD_SIZE, NUM_INPUT_PLANES),
            dtype=np.float32,
        )
    )
    return model, predict_fn
