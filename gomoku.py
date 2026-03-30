#!/usr/bin/env python3
"""
Gomoku (Five In A Row) - Shared module
Game logic, AlphaZero-style residual CNN, and MCTS implementation.
Includes both single-eval and batched MCTS (virtual-loss) variants.

TensorFlow is imported lazily (inside create_model) so that
multiprocessing workers can set CUDA_VISIBLE_DEVICES before TF
initialises the GPU.  All other code in this module uses only NumPy.
"""

import numpy as np
import warnings
from math import sqrt as _sqrt

# ── Optional Cython acceleration ───────────────────────────────────────────
# Build with:  python setup_accel.py build_ext --inplace
try:
    import mcts_accel as _accel
    _USE_ACCEL = True
except ImportError:
    _accel = None
    _USE_ACCEL = False

# select_child in older mcts_accel builds used the wrong value sign and
# deterministic index-order tie-breaking. Keep other accel paths enabled,
# but use Python select_child unless extension advertises both fixes.
_USE_ACCEL_SELECT = _USE_ACCEL and bool(
    getattr(_accel, "SELECT_CHILD_PARENT_VIEW", 0)
) and bool(
    getattr(_accel, "SELECT_CHILD_TIEBREAK_PRIOR", 0)
)
if _USE_ACCEL and not _USE_ACCEL_SELECT:
    warnings.warn(
        "mcts_accel missing SELECT_CHILD_PARENT_VIEW and/or "
        "SELECT_CHILD_TIEBREAK_PRIOR; using Python select_child. "
        "Rebuild with setup_accel.py.",
        RuntimeWarning,
        stacklevel=2,
    )

# ── Optional Numba acceleration for threat planes ─────────────────────────
try:
    from numba import njit as _njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

# ── Game constants ──────────────────────────────────────────────────────────
BOARD_SIZE = 15
WIN_LENGTH = 5
EMPTY = 0
PLAYER1 = 1

# Input encoding: 2 stone planes + 4 threat planes
NUM_INPUT_PLANES = 6
PLAYER2 = -1

# Prior floor: small blend to bootstrap exploration. With random networks,
# both prior and value are noise — MCTS needs SOME forced exploration so the
# value head can learn from diverse positions. 0.05 gives a tiny floor
# (0.05/30 ≈ 0.17% per move) without flattening a peaked learned prior.
# 0.25 was too much (destroyed signal); 0.0 was too little (no bootstrapping).
PRIOR_UNIFORM_BLEND = 0.05


# ── Game ────────────────────────────────────────────────────────────────────
class GomokuGame:
    """Five In A Row game logic."""

    _FRONTIER_DIST = 2  # distance-2 neighbours for broader tactical coverage

    def __init__(self, size=BOARD_SIZE):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)
        self.current_player = PLAYER1
        self.move_history = []
        # Frontier: count of occupied neighbours within distance=2.
        # Maintained incrementally so get_candidate_moves is a cheap np.where.
        self._frontier = np.zeros((size, size), dtype=np.int16)

    def _update_frontier(self, row, col, delta):
        """Add or remove frontier contribution for one stone. O(49)."""
        d = self._FRONTIER_DIST
        s = self.size
        r0 = max(0, row - d); r1 = min(s, row + d + 1)
        c0 = max(0, col - d); c1 = min(s, col + d + 1)
        self._frontier[r0:r1, c0:c1] += delta

    def copy(self):
        g = GomokuGame.__new__(GomokuGame)
        g.size = self.size
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.move_history = list(self.move_history)
        g._frontier = self._frontier.copy()
        return g

    def reset(self):
        self.board[:] = 0
        self._frontier[:] = 0
        self.current_player = PLAYER1
        self.move_history.clear()

    def get_state(self):
        """Board from current player's perspective (+1 = mine, -1 = opponent)."""
        return self.board * self.current_player

    def get_valid_moves(self):
        return list(zip(*np.where(self.board == EMPTY), strict=False))

    def make_move(self, row, col):
        """Returns (reward, done). reward=1 means current player wins."""
        if self.board[row, col] != EMPTY:
            return -1, True
        self.board[row, col] = self.current_player
        self._update_frontier(row, col, 1)
        self.move_history.append((row, col, self.current_player))
        if self._check_win(row, col):
            return 1, True
        if len(self.move_history) == self.size * self.size:
            return 0, True   # board full — draw
        self.current_player *= -1
        return 0, False

    def undo_move(self):
        if not self.move_history:
            return False
        row, col, player = self.move_history.pop()
        self.board[row, col] = EMPTY
        self._update_frontier(row, col, -1)
        self.current_player = player
        return True

    def _check_win(self, row, col):
        player = self.board[row, col]
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1
            r, c = row + dr, col + dc
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == player:
                count += 1; r += dr; c += dc
            r, c = row - dr, col - dc
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == player:
                count += 1; r -= dr; c -= dc
            if count >= WIN_LENGTH:
                return True
        return False


# ── State encoding ──────────────────────────────────────────────────────────
def encode_state(game):
    """Encode board as 6 planes from the current player's perspective.

    Planes 0-1: stone positions (my stones, opponent stones)
    Planes 2-3: four-threats (cells in a line-of-5 window with 4 friendly, 0 enemy)
    Planes 4-5: three-threats (cells in a line-of-5 window with 3 friendly, 0 enemy)

    Threat planes give the network direct tactical awareness of open-fours
    and open-threes without requiring it to rediscover these patterns from
    sparse win/loss signal.
    """
    state = game.board * game.current_player
    out = np.zeros((game.size, game.size, NUM_INPUT_PLANES), dtype=np.float32)
    my = (state == 1)
    opp = (state == -1)
    out[:, :, 0] = my
    out[:, :, 1] = opp

    # Dispatch: Cython (~3μs) > numba (~3μs) > numpy (~150μs)
    if _USE_ACCEL:
        _accel.compute_threat_planes(my.view(np.int8), opp.view(np.int8),
                                     out, game.size)
    elif _HAS_NUMBA:
        _compute_threat_planes_numba(my.view(np.int8), opp.view(np.int8),
                                     out, game.size)
    else:
        _compute_threat_planes(my, opp, out, game.size)
    return out


def _compute_threat_planes(my, opp, out, size):
    """Fill threat planes (channels 2-5) by scanning all line-of-5 windows.

    Channels: 2=my fours, 3=opp fours, 4=my threes, 5=opp threes.
    A "four" window has exactly 4 friendly + 0 enemy stones.
    A "three" window has exactly 3 friendly + 0 enemy stones.

    Pure current-board, no lookahead.  ~150μs on 15×15 via numpy slicing.
    """
    n = size - 4  # windows per line

    def _scan_dir(my_slices, opp_slices, scatter_fn):
        """Count stones in all windows, mark matching cells."""
        my_c = np.zeros_like(my_slices[0], dtype=np.int8)
        op_c = np.zeros_like(my_slices[0], dtype=np.int8)
        for ms, os_ in zip(my_slices, opp_slices, strict=False):
            my_c += ms
            op_c += os_

        # Both players from one set of counts
        for count, ch4, ch3 in [(my_c, 2, 4), (op_c, 3, 5)]:
            other = op_c if count is my_c else my_c
            m4 = (count == 4) & (other == 0)
            m3 = (count == 3) & (other == 0)
            if m4.any():
                scatter_fn(m4, ch4)
            if m3.any():
                scatter_fn(m3, ch3)

    # ── Horizontal (0, +1) ──
    h_slices_my = [my[:, k:k + n] for k in range(5)]
    h_slices_op = [opp[:, k:k + n] for k in range(5)]
    def scatter_h(mask, ch):
        for k in range(5):
            out[:, k:k + n, ch] += mask
    _scan_dir(h_slices_my, h_slices_op, scatter_h)

    # ── Vertical (+1, 0) ──
    v_slices_my = [my[k:k + n, :] for k in range(5)]
    v_slices_op = [opp[k:k + n, :] for k in range(5)]
    def scatter_v(mask, ch):
        for k in range(5):
            out[k:k + n, :, ch] += mask
    _scan_dir(v_slices_my, v_slices_op, scatter_v)

    # ── Diagonal (+1, +1) ──
    d_slices_my = [my[k:k + n, k:k + n] for k in range(5)]
    d_slices_op = [opp[k:k + n, k:k + n] for k in range(5)]
    def scatter_d(mask, ch):
        for k in range(5):
            out[k:k + n, k:k + n, ch] += mask
    _scan_dir(d_slices_my, d_slices_op, scatter_d)

    # ── Anti-diagonal (+1, -1) ──
    a_slices_my = [my[k:k + n, (4 - k):(size - k)] for k in range(5)]
    a_slices_op = [opp[k:k + n, (4 - k):(size - k)] for k in range(5)]
    def scatter_a(mask, ch):
        for k in range(5):
            out[k:k + n, (4 - k):(size - k), ch] += mask
    _scan_dir(a_slices_my, a_slices_op, scatter_a)

    # Clamp to binary (a cell can be in multiple windows)
    np.clip(out[:, :, 2:], 0, 1, out=out[:, :, 2:])


# ── Numba-accelerated threat plane computation (~50× faster) ──────────────
if _HAS_NUMBA:
    @_njit
    def _compute_threat_planes_numba(my, opp, out, size):
        """JIT-compiled threat plane scanner.  ~3μs on 15×15 vs ~150μs numpy."""
        n = size - 4
        # Horizontal
        for r in range(size):
            for c in range(n):
                mc = my[r,c]+my[r,c+1]+my[r,c+2]+my[r,c+3]+my[r,c+4]
                oc = opp[r,c]+opp[r,c+1]+opp[r,c+2]+opp[r,c+3]+opp[r,c+4]
                if mc == 4 and oc == 0:
                    for k in range(5): out[r, c+k, 2] = 1.0
                elif mc == 3 and oc == 0:
                    for k in range(5): out[r, c+k, 4] = 1.0
                if oc == 4 and mc == 0:
                    for k in range(5): out[r, c+k, 3] = 1.0
                elif oc == 3 and mc == 0:
                    for k in range(5): out[r, c+k, 5] = 1.0
        # Vertical
        for r in range(n):
            for c in range(size):
                mc = my[r,c]+my[r+1,c]+my[r+2,c]+my[r+3,c]+my[r+4,c]
                oc = opp[r,c]+opp[r+1,c]+opp[r+2,c]+opp[r+3,c]+opp[r+4,c]
                if mc == 4 and oc == 0:
                    for k in range(5): out[r+k, c, 2] = 1.0
                elif mc == 3 and oc == 0:
                    for k in range(5): out[r+k, c, 4] = 1.0
                if oc == 4 and mc == 0:
                    for k in range(5): out[r+k, c, 3] = 1.0
                elif oc == 3 and mc == 0:
                    for k in range(5): out[r+k, c, 5] = 1.0
        # Diagonal and anti-diagonal
        for r in range(n):
            for c in range(n):
                mc = my[r,c]+my[r+1,c+1]+my[r+2,c+2]+my[r+3,c+3]+my[r+4,c+4]
                oc = opp[r,c]+opp[r+1,c+1]+opp[r+2,c+2]+opp[r+3,c+3]+opp[r+4,c+4]
                if mc == 4 and oc == 0:
                    for k in range(5): out[r+k, c+k, 2] = 1.0
                elif mc == 3 and oc == 0:
                    for k in range(5): out[r+k, c+k, 4] = 1.0
                if oc == 4 and mc == 0:
                    for k in range(5): out[r+k, c+k, 3] = 1.0
                elif oc == 3 and mc == 0:
                    for k in range(5): out[r+k, c+k, 5] = 1.0
                # Anti-diagonal: (r, c+4) to (r+4, c)
                mc = my[r,c+4]+my[r+1,c+3]+my[r+2,c+2]+my[r+3,c+1]+my[r+4,c]
                oc = opp[r,c+4]+opp[r+1,c+3]+opp[r+2,c+2]+opp[r+3,c+1]+opp[r+4,c]
                if mc == 4 and oc == 0:
                    for k in range(5): out[r+k, c+4-k, 2] = 1.0
                elif mc == 3 and oc == 0:
                    for k in range(5): out[r+k, c+4-k, 4] = 1.0
                if oc == 4 and mc == 0:
                    for k in range(5): out[r+k, c+4-k, 3] = 1.0
                elif oc == 3 and mc == 0:
                    for k in range(5): out[r+k, c+4-k, 5] = 1.0


# ── Model ───────────────────────────────────────────────────────────────────
def create_model(board_size=BOARD_SIZE, num_res_blocks=10, num_filters=128,
                 se_ratio=4):
    """AlphaZero-style residual CNN with SE global pooling.

    Every other residual block includes a Squeeze-and-Excitation layer
    that lets the network condition on global board context (e.g. which
    side has more threats overall, board-wide stone balance).

    Policy head outputs raw *logits* (no softmax) — masking and softmax
    are applied externally during MCTS or training.

    TF/Keras are imported here (lazily) so that worker processes can
    configure GPU visibility before the first import.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    tf.get_logger().setLevel("ERROR")
    warnings.filterwarnings("ignore", category=UserWarning)

    inputs = keras.Input(shape=(board_size, board_size, NUM_INPUT_PLANES))

    # Initial convolution
    x = layers.Conv2D(num_filters, 3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Residual tower with optional SE blocks
    se_channels = max(num_filters // se_ratio, 1)
    for i in range(num_res_blocks):
        residual = x
        x = layers.Conv2D(num_filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(num_filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)

        # SE block on every other residual block (1, 3, 5, ...)
        if i % 2 == 1:
            se = layers.GlobalAveragePooling2D()(x)
            se = layers.Dense(se_channels, activation="relu")(se)
            se = layers.Dense(num_filters, activation="sigmoid")(se)
            se = layers.Reshape((1, 1, num_filters))(se)
            x = layers.Multiply()([x, se])

        x = layers.Add()([x, residual])
        x = layers.ReLU()(x)

    # Policy head → logits
    p = layers.Conv2D(2, 1, use_bias=False)(x)
    p = layers.BatchNormalization()(p)
    p = layers.ReLU()(p)
    p = layers.Flatten()(p)
    p = layers.Dense(board_size * board_size, name="policy")(p)

    # Value head → tanh
    v = layers.Conv2D(1, 1, use_bias=False)(x)
    v = layers.BatchNormalization()(v)
    v = layers.ReLU()(v)
    v = layers.Flatten()(v)
    v = layers.Dense(128, activation="relu")(v)
    v = layers.Dense(1, activation="tanh", name="value")(v)

    return keras.Model(inputs=inputs, outputs=[p, v])


def make_predict_fn(model):
    """Create a @tf.function-compiled prediction function for the model.

    Uses input_signature with a dynamic batch dimension so tf.function
    traces exactly once and reuses that graph for all batch sizes.
    This eliminates the per-call retracing that dominates with
    reduce_retracing=True and variable-size coordinator batches.

    Returns:
        predict_np(x) → (logits_np, values_np) as numpy arrays
    """
    import tensorflow as tf

    @tf.function(input_signature=[
        tf.TensorSpec([None, BOARD_SIZE, BOARD_SIZE, NUM_INPUT_PLANES], tf.float32)
    ])
    def _predict(x):
        return model(x, training=False)

    def predict_np(x):
        logits, values = _predict(x)
        return logits.numpy(), values.numpy()

    # Expose the raw tf.function so callers can launch GPU work asynchronously.
    # Calling _predict(x) enqueues GPU ops and returns TF tensors immediately
    # (before the GPU finishes).  Calling .numpy() on those tensors blocks until
    # the result is ready.  This lets the CPU do other work (e.g. MCTS tree
    # traversal) while the GPU computes, overlapping the two phases.
    predict_np._raw = _predict

    return predict_np


# ── Nearby-moves optimisation ──────────────────────────────────────────────
def get_candidate_moves(board, distance=2, density_threshold=2, frontier=None):
    """Return candidate empty squares for MCTS expansion.

    When fewer than `density_threshold` stones are on the board, returns ALL
    empty squares.  Once denser, restricts to squares within `distance` of
    any occupied square. With distance=2 and typical positions, this gives
    a broader tactical candidate set for MCTS expansion.

    If `frontier` is provided (incremental count array from GomokuGame),
    uses a fast np.where instead of dilation.
    """
    n_occupied = int(np.count_nonzero(board))

    if n_occupied < density_threshold:
        return list(zip(*np.where(board == EMPTY), strict=False))

    # Fast path: use pre-computed frontier from GomokuGame
    if frontier is not None:
        return list(zip(*np.where((board == EMPTY) & (frontier > 0)), strict=False))

    # Fallback: dilation (used when frontier not available)
    if _USE_ACCEL:
        return _accel.get_candidate_moves(board, distance, density_threshold)

    size = board.shape[0]
    nearby = np.zeros_like(board, dtype=np.bool_)
    for dr in range(-distance, distance + 1):
        for dc in range(-distance, distance + 1):
            src_r0 = max(0, -dr);  src_r1 = min(size, size - dr)
            src_c0 = max(0, -dc);  src_c1 = min(size, size - dc)
            dst_r0 = src_r0 + dr;  dst_r1 = src_r1 + dr
            dst_c0 = src_c0 + dc;  dst_c1 = src_c1 + dc
            nearby[dst_r0:dst_r1, dst_c0:dst_c1] |= (board[src_r0:src_r1, src_c0:src_c1] != EMPTY)

    nearby &= (board == EMPTY)
    return list(zip(*np.where(nearby), strict=False))


# ── Shared helpers ──────────────────────────────────────────────────────────
def _masked_softmax(logits, moves, board_size):
    """Apply masked softmax over legal moves. Returns full probability vector."""
    if _USE_ACCEL:
        return _accel.masked_softmax(logits, moves, board_size)

    # Vectorized mask: convert move list to flat indices once
    indices = np.array([r * board_size + c for r, c in moves], dtype=np.intp)
    mask = np.full(len(logits), -1e9, dtype=logits.dtype)
    mask[indices] = 0.0
    logits = logits + mask
    logits -= logits.max()
    probs = np.exp(logits)
    s = probs.sum()
    if s <= 0 or not np.isfinite(s):
        probs[:] = 0.0
        probs[indices] = 1.0
        probs /= probs.sum()
    else:
        probs /= s
    return probs


# ── MCTS ────────────────────────────────────────────────────────────────────
class MCTSNode:
    __slots__ = ("visit_count", "value_sum", "prior", "children",
                 "_moves", "_priors")

    def __init__(self, prior: float):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children: dict[tuple[int, int], "MCTSNode"] = {}
        self._moves = None    # list of (r,c) — set on expansion
        self._priors = None   # float32 array aligned with _moves

    @property
    def q_value(self):
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    @property
    def expanded(self):
        return self._moves is not None


def _call_predict(fn, batch):
    """Call a predict function, handling both compiled and raw model results.

    Compiled predict_fn (from make_predict_fn): takes batch, returns numpy.
    Raw keras model: needs training=False, returns tensors → converted.
    """
    if hasattr(fn, 'trainable_variables'):
        # Raw keras model
        result = fn(batch, training=False)
        return result[0].numpy(), result[1].numpy()
    # Compiled predict_fn — already returns numpy
    return fn(batch)


def _expand(node, game, predict_fn):
    """Expand a leaf: run the network, store priors for lazy child creation.

    predict_fn: either make_predict_fn() result or a raw keras model.
    """
    moves = get_candidate_moves(game.board, frontier=game._frontier)
    if not moves:
        return 0.0

    state = encode_state(game)[np.newaxis, ...]
    logits, value = _call_predict(predict_fn, state)

    probs = _masked_softmax(logits.ravel(), moves, game.size)
    indices = np.array([r * game.size + c for r, c in moves], dtype=np.intp)
    raw_priors = probs[indices].astype(np.float32)
    # Blend with uniform to prevent prior starvation
    n = len(moves)
    node._moves = moves
    node._priors = ((1 - PRIOR_UNIFORM_BLEND) * raw_priors
                    + PRIOR_UNIFORM_BLEND / n).astype(np.float32)

    return float(value.ravel()[0])


def _expand_from_output(node, moves, logits, value, board_size):
    """Expand a leaf using pre-computed network outputs (for batched MCTS).

    Stores moves and priors for lazy child creation — children are only
    instantiated when _select_child first picks them.  This avoids creating
    ~120 MCTSNode objects per leaf when only ~30 ever get visited.
    """
    if _USE_ACCEL:
        value = _accel.expand_from_output(node, moves, logits, value, board_size)
        # Apply prior floor to Cython-computed priors too
        n = len(moves)
        if PRIOR_UNIFORM_BLEND > 0 and n > 0:
            node._priors = ((1 - PRIOR_UNIFORM_BLEND) * node._priors
                            + PRIOR_UNIFORM_BLEND / n).astype(np.float32)
        return value

    probs = _masked_softmax(logits, moves, board_size)
    indices = np.array([r * board_size + c for r, c in moves], dtype=np.intp)
    raw_priors = probs[indices].astype(np.float32)
    n = len(moves)
    node._moves = moves
    node._priors = ((1 - PRIOR_UNIFORM_BLEND) * raw_priors
                    + PRIOR_UNIFORM_BLEND / n).astype(np.float32)
    return value


def _select_child(node, c_puct):
    """Select best child by UCB.  Creates the child node lazily on first visit.

    Iterates over node._moves / node._priors (set at expansion time).
    Only the selected child gets a MCTSNode allocated — unvisited moves
    are evaluated purely from their prior, avoiding ~90% of node allocations.
    """
    if _USE_ACCEL_SELECT:
        return _accel.select_child(node, c_puct)

    sqrt_n = _sqrt(node.visit_count) if node.visit_count else 0.0
    best_score = -1e18
    best_idx = -1
    best_child = None
    best_prior = -1.0
    children = node.children
    moves = node._moves
    priors = node._priors

    eps = 1e-12
    for i in range(len(moves)):
        child = children.get(moves[i])
        if child is not None:
            vc = child.visit_count
            # child.q_value is from child-player perspective; flip sign so
            # parent selects moves maximizing parent value.
            q = -(child.value_sum / vc) if vc else 0.0
        else:
            vc = 0
            q = 0.0
        prior_i = float(priors[i])
        ucb = q + c_puct * prior_i * sqrt_n / (1 + vc)
        if (ucb > best_score + eps
                or (best_idx < 0)
                or (abs(ucb - best_score) <= eps and prior_i > best_prior + eps)):
            best_score = ucb
            best_idx = i
            best_child = child
            best_prior = prior_i

    action = moves[best_idx]
    if best_child is None:
        best_child = MCTSNode(prior=float(priors[best_idx]))
        children[action] = best_child
    return action, best_child


def _add_dirichlet_noise(root, alpha, noise_frac):
    """Mix Dirichlet noise into root priors for exploration."""
    if root._moves is not None and len(root._moves) > 0:
        noise = np.random.dirichlet([alpha] * len(root._moves))
        root._priors = ((1 - noise_frac) * root._priors
                        + noise_frac * noise).astype(np.float32)
        # Update any already-created children to match
        for i, action in enumerate(root._moves):
            child = root.children.get(action)
            if child is not None:
                child.prior = float(root._priors[i])


# ── Single-eval MCTS (original, used by play.py) ───────────────────────────
def mcts_search(game, model, num_simulations=200, c_puct=1.5,
                add_noise=True, dirichlet_alpha=0.15, noise_frac=0.25):
    """Run MCTS from `game`'s current state and return the root node.

    Each simulation evaluates one leaf via a single network forward pass.
    For training self-play, prefer mcts_search_batched() which amortises
    forward-pass cost across multiple leaves per call.
    """
    root = MCTSNode(prior=0.0)
    _expand(root, game, model)

    if add_noise:
        _add_dirichlet_noise(root, dirichlet_alpha, noise_frac)

    scratch = game.copy()   # single scratch game, reused every simulation

    for _ in range(num_simulations):
        node = root
        path = [node]
        depth = 0

        while node.expanded:
            action, node = _select_child(node, c_puct)
            path.append(node)
            reward, done = scratch.make_move(*action)
            depth += 1
            if done:
                value = -reward
                break
        else:
            value = _expand(node, scratch, model)

        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += value
            value = -value

        # Undo moves to return scratch to root state
        for _ in range(depth):
            scratch.undo_move()

    return root


# ── Batched MCTS (virtual loss) ────────────────────────────────────────────
def _backprop_batch(pending, eval_list, logits_np, values_np, game_size):
    """Expand leaf nodes from NN output and backpropagate values.

    Shared by both the synchronous and pipelined paths of mcts_search_batched.
    logits_np / values_np may be None when eval_list is empty (all-terminal round).
    """
    VIRTUAL_LOSS = 1.0
    if eval_list and logits_np is not None:
        values_flat = values_np.ravel()
        for j, e in enumerate(eval_list):
            logits_j = logits_np[j].ravel()
            value_j = float(values_flat[j])
            if e["moves"]:
                _expand_from_output(e["node"], e["moves"], logits_j, value_j, game_size)
            else:
                value_j = 0.0
            for idx in e["indices"]:
                pending[idx]["value"] = value_j

    for p in pending:
        path = p["path"]
        value = p["value"] if p["value"] is not None else 0.0
        for n in path[1:]:
            n.visit_count -= 1
            n.value_sum -= VIRTUAL_LOSS
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += value
            value = -value


def mcts_search_batched(game, model, num_simulations=200, batch_size=8,
                        c_puct=1.5, add_noise=True, dirichlet_alpha=0.15,
                        noise_frac=0.25):
    """MCTS with batched neural-network evaluation via virtual loss.

    Instead of one forward pass per simulation, we select `batch_size`
    paths in parallel (using virtual loss to encourage diversity), collect
    their leaf states, run a single batched forward pass, then expand and
    back-propagate all paths at once.

    Uses a single scratch game with make/undo instead of copying the game
    for each path, which is much cheaper on CPU.

    When model exposes a ._raw tf.function (set by make_predict_fn), uses
    double-buffering: the GPU evaluates batch N while the CPU selects leaves
    for batch N+1, hiding GPU latency behind CPU tree-traversal work.
    """
    VIRTUAL_LOSS = 1.0

    root = MCTSNode(prior=0.0)
    _expand(root, game, model)

    if add_noise:
        _add_dirichlet_noise(root, dirichlet_alpha, noise_frac)

    scratch = game.copy()   # single scratch game, reused for all paths
    _raw = getattr(model, '_raw', None)

    sims_done = 0
    # Pipeline slot: stash previous batch while GPU runs, process it next iter.
    slot_pending = slot_eval_list = None
    slot_tf_logits = slot_tf_values = None
    slot_n_batch = 0
    slot_has_data = False

    while sims_done < num_simulations:
        n_batch = min(batch_size, num_simulations - sims_done)

        # ── Phase 1: select paths with virtual loss ─────────────────
        # (GPU is computing the previous batch in the background here)
        pending = []
        seen_leaves = {}  # id(node) → index into pending (first occurrence)

        for _ in range(n_batch):
            node = root
            path = [node]
            depth = 0
            terminal_value = None

            while node.expanded:
                action, child = _select_child(node, c_puct)
                # Virtual loss: make this child look worse so later paths
                # in the same batch are pushed toward different leaves.
                child.visit_count += 1
                child.value_sum += VIRTUAL_LOSS
                path.append(child)
                reward, done = scratch.make_move(*action)
                depth += 1
                node = child
                if done:
                    terminal_value = -reward
                    break

            # Only encode leaf state once per unique node
            leaf_state = None
            leaf_moves = None
            if terminal_value is None:
                nid = id(node)
                if nid not in seen_leaves:
                    leaf_state = encode_state(scratch)
                    leaf_moves = get_candidate_moves(scratch.board, frontier=scratch._frontier)
                    seen_leaves[nid] = len(pending)

            # Undo moves to return scratch to root state
            for _ in range(depth):
                scratch.undo_move()

            pending.append({
                "path": path,
                "terminal": terminal_value,
                "node": node,
                "value": terminal_value,
                "leaf_state": leaf_state,
                "leaf_moves": leaf_moves,
            })

        # ── Build unique eval set ────────────────────────────────────
        unique_evals = {}   # id(node) → {state, moves, node, indices}
        for i, p in enumerate(pending):
            if p["terminal"] is not None:
                continue
            nid = id(p["node"])
            if nid not in unique_evals:
                first_idx = seen_leaves[nid]
                unique_evals[nid] = {
                    "state": pending[first_idx]["leaf_state"],
                    "moves": pending[first_idx]["leaf_moves"],
                    "node": p["node"],
                    "indices": [i],
                }
            else:
                unique_evals[nid]["indices"].append(i)

        eval_list = list(unique_evals.values()) if unique_evals else []

        if _raw is not None:
            # ── Async path: launch GPU for current batch ─────────────
            if eval_list:
                batch_states = np.array([e["state"] for e in eval_list])
                cur_tf_logits, cur_tf_values = _raw(batch_states)
            else:
                cur_tf_logits = cur_tf_values = None

            # Process previous slot: .numpy() syncs the GPU result, then
            # expand + backprop (GPU has been running during Phase 1 above).
            if slot_has_data:
                if slot_tf_logits is not None:
                    logits_np = slot_tf_logits.numpy()
                    values_np = slot_tf_values.numpy()
                else:
                    logits_np = values_np = None
                _backprop_batch(slot_pending, slot_eval_list,
                                logits_np, values_np, game.size)
                sims_done += slot_n_batch

            slot_pending, slot_eval_list = pending, eval_list
            slot_tf_logits, slot_tf_values = cur_tf_logits, cur_tf_values
            slot_n_batch = n_batch
            slot_has_data = True

        else:
            # ── Sync path: original behaviour (no ._raw available) ───
            if eval_list:
                batch_states = np.array([e["state"] for e in eval_list])
                logits_np, values_np = _call_predict(model, batch_states)
            else:
                logits_np = values_np = None
            _backprop_batch(pending, eval_list, logits_np, values_np, game.size)
            sims_done += n_batch

    # ── Drain: process the last outstanding slot ─────────────────────────
    if slot_has_data:
        if slot_tf_logits is not None:
            logits_np = slot_tf_logits.numpy()
            values_np = slot_tf_values.numpy()
        else:
            logits_np = values_np = None
        _backprop_batch(slot_pending, slot_eval_list,
                        logits_np, values_np, game.size)

    return root


# ── Phased MCTS (for multi-game interleaving) ────────────────────────────
# These split the MCTS loop into stages so an external coordinator can
# combine leaf evaluations from many games into one GPU batch.

def mcts_begin(game, num_simulations=100, batch_size=8, c_puct=1.5,
               add_noise=True, dirichlet_alpha=0.15, noise_frac=0.25):
    """Create an MCTS context.  Returns (ctx, root_state).

    root_state is the encoded board needing one NN evaluation.
    Call mcts_expand_root(ctx, logits, value) with the result,
    then alternate mcts_select_leaves / mcts_process_results.
    """
    root = MCTSNode(prior=0.0)
    scratch = game.copy()
    root_state = encode_state(game)
    root_moves = get_candidate_moves(game.board, frontier=game._frontier)

    ctx = {
        "root": root,
        "game": game,
        "scratch": scratch,
        "sims_done": 0,
        "sims_target": num_simulations,
        "batch_size": batch_size,
        "c_puct": c_puct,
        "add_noise": add_noise,
        "dir_alpha": dirichlet_alpha,
        "noise_frac": noise_frac,
        "root_moves": root_moves,
        "root_expanded": False,
        "pending": None,
        "eval_list": None,
        "n_batch": 0,
    }
    return ctx, root_state


def mcts_expand_root(ctx, logits, value):
    """Expand root node with NN output, optionally add Dirichlet noise."""
    root = ctx["root"]
    moves = ctx["root_moves"]
    board_size = ctx["game"].size

    logits = np.asarray(logits).ravel()
    value = float(np.asarray(value).ravel()[0]) if not isinstance(value, float) else value

    if moves:
        _expand_from_output(root, moves, logits, value, board_size)

    if ctx["add_noise"]:
        _add_dirichlet_noise(root, ctx["dir_alpha"], ctx["noise_frac"])

    ctx["root_expanded"] = True


def mcts_select_leaves(ctx):
    """Phase 1: walk tree with virtual loss, return leaf states needing eval.

    Returns a list of encoded state arrays (may be empty if all paths
    hit terminal nodes).  Stores internal state for mcts_process_results.
    """
    root = ctx["root"]
    assert ctx["root_expanded"], "mcts_select_leaves called before mcts_expand_root"
    scratch = ctx["scratch"]
    c_puct = ctx["c_puct"]
    sims_left = ctx["sims_target"] - ctx["sims_done"]
    n_batch = min(ctx["batch_size"], sims_left)

    VIRTUAL_LOSS = 1.0
    pending = []
    seen_leaves = {}

    for _ in range(n_batch):
        node = root
        path = [node]
        depth = 0
        terminal_value = None

        while node.expanded:
            action, child = _select_child(node, c_puct)
            child.visit_count += 1
            child.value_sum += VIRTUAL_LOSS
            path.append(child)
            reward, done = scratch.make_move(*action)
            depth += 1
            node = child
            if done:
                terminal_value = -reward
                break

        leaf_state = None
        leaf_moves = None
        if terminal_value is None:
            nid = id(node)
            if nid not in seen_leaves:
                leaf_state = encode_state(scratch)
                leaf_moves = get_candidate_moves(scratch.board, frontier=scratch._frontier)
                seen_leaves[nid] = len(pending)

        for _ in range(depth):
            scratch.undo_move()

        pending.append({
            "path": path,
            "terminal": terminal_value,
            "node": node,
            "value": terminal_value,
            "leaf_state": leaf_state,
            "leaf_moves": leaf_moves,
        })

    # Build unique eval set
    unique_evals = {}
    for i, p in enumerate(pending):
        if p["terminal"] is not None:
            continue
        nid = id(p["node"])
        if nid not in unique_evals:
            first_idx = seen_leaves[nid]
            unique_evals[nid] = {
                "state": pending[first_idx]["leaf_state"],
                "moves": pending[first_idx]["leaf_moves"],
                "node": p["node"],
                "indices": [i],
            }
        else:
            unique_evals[nid]["indices"].append(i)

    eval_list = list(unique_evals.values()) if unique_evals else []
    ctx["pending"] = pending
    ctx["eval_list"] = eval_list
    ctx["n_batch"] = n_batch

    return [e["state"] for e in eval_list]


def mcts_process_results(ctx, batch_logits=None, batch_values=None):
    """Phase 2+3: expand leaves with NN output and backpropagate.

    batch_logits / batch_values must correspond 1:1 to the states
    returned by the previous mcts_select_leaves call.  Pass None if
    that call returned an empty list (all-terminal round).
    """
    pending = ctx["pending"]
    eval_list = ctx["eval_list"]
    game_size = ctx["game"].size
    VIRTUAL_LOSS = 1.0

    if eval_list:
        assert batch_logits is not None and batch_values is not None, \
            f"eval_list has {len(eval_list)} entries but no batch results provided"
        assert len(batch_logits) == len(eval_list), \
            f"logits count {len(batch_logits)} != eval_list {len(eval_list)}"
        assert len(batch_values) == len(eval_list), \
            f"values count {len(batch_values)} != eval_list {len(eval_list)}"
        for j, e in enumerate(eval_list):
            logits_j = batch_logits[j].ravel()
            value_j = float(batch_values[j])
            node = e["node"]
            moves = e["moves"]

            if moves:
                _expand_from_output(node, moves, logits_j, value_j, game_size)
            else:
                value_j = 0.0

            for idx in e["indices"]:
                pending[idx]["value"] = value_j

    # Undo virtual loss, then standard backup
    for p in pending:
        path = p["path"]
        value = p["value"] if p["value"] is not None else 0.0

        for n in path[1:]:
            n.visit_count -= 1
            n.value_sum -= VIRTUAL_LOSS

        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += value
            value = -value

    ctx["sims_done"] += ctx["n_batch"]
    ctx["pending"] = None
    ctx["eval_list"] = None
    ctx["n_batch"] = 0


# ── Policy extraction & action sampling ─────────────────────────────────────
def mcts_policy(root, board_size=BOARD_SIZE, temperature=1.0):
    """Convert root visit counts into a probability vector over all squares."""
    counts = np.zeros(board_size * board_size, dtype=np.float64)
    for (r, c), child in root.children.items():
        counts[r * board_size + c] = child.visit_count

    # Tie-breaker signal for equal visit counts: prefer higher root prior.
    # This avoids deterministic index-order artifacts such as always choosing
    # (0,0) when many moves share identical visit counts at low sim budgets.
    prior = np.zeros(board_size * board_size, dtype=np.float64)
    if root._moves is not None and root._priors is not None:
        for i, (r, c) in enumerate(root._moves):
            idx = r * board_size + c
            if counts[idx] > 0.0:
                prior[idx] = float(root._priors[i])

    if temperature < 0.01:
        max_count = counts.max()
        if max_count <= 0:
            best = int(np.argmax(prior))
        else:
            best_idxs = np.flatnonzero(counts == max_count)
            if len(best_idxs) > 1:
                best = int(best_idxs[np.argmax(prior[best_idxs])])
            else:
                best = int(best_idxs[0])
        policy = np.zeros_like(counts)
        policy[best] = 1.0
    else:
        if np.any(prior > 0.0):
            counts = counts + (counts > 0.0) * (1e-6 * prior)
        counts = counts ** (1.0 / temperature)
        total = counts.sum()
        policy = counts / total if total > 0 else counts

    return policy.astype(np.float32)


def select_action(policy):
    """Sample an action index from a policy vector, return (row, col)."""
    idx = np.random.choice(len(policy), p=policy)
    return divmod(idx, BOARD_SIZE)
