"""Neural-guided minimax agent.

Runs alpha-beta search to a given depth and uses a DQN network to evaluate
leaf positions (instead of a hand-coded heuristic).

Terminal wins/losses always beat any DQN-estimated value, so the agent
never ignores a forced outcome in favor of a high Q-value.

Performance strategy:
    1. First pass (`_collect_leaves`): enumerate every non-terminal leaf
       position reachable at `depth` plies. NO pruning — we need all of them.
    2. Batch DQN call on all leaves in a single forward pass (GPU-friendly).
    3. Second pass (`_negamax`): full alpha-beta search, but leaf values now
       come from the transposition table (filled in step 2). No more DQN calls.

This gives up some alpha-beta pruning in the first pass in exchange for
GPU batching, which is a big net win — the GPU is otherwise idle between
sequential single-state forward passes.
"""

import random as _rand
import numpy as np
import torch

from envs.connect4 import ROWS, COLS, WIN_LENGTH


TERMINAL_WIN = 10_000.0
PLY_BONUS = 1.0  # prefer faster wins / slower losses

# Column ordering for alpha-beta — center columns prune best
MOVE_ORDER = [3, 2, 4, 1, 5, 0, 6]


class NeuralMinimaxAgent:
    """Alpha-beta minimax guided by a DQN value estimator.

    Args:
        rl_agent: A DQNAgent instance (must expose `q_net` and `device`).
        depth:    Search depth in plies (default 3).
    """

    def __init__(self, rl_agent, depth=3):
        self.rl_agent = rl_agent
        self.depth = depth
        self._device = rl_agent.device  # use whatever device the model is on
        self._q_net = rl_agent.q_net
        self._q_net.eval()
        # Transposition table: (board_bytes, player) -> value
        self._tt = {}

    # ---- public API (matches HybridAgent / DQNAgent) ----

    def select_action(self, env, greedy=True, allowed_actions=None):
        board = env.board.copy()
        player = int(env.current_player)
        legal = allowed_actions if allowed_actions is not None else env.get_legal_actions()
        if not legal:
            return 0

        # Clear TT for this move
        self._tt = {}

        # --- Pass 1: enumerate all leaves (no DQN calls yet) ---
        leaf_boards = []
        leaf_players = []
        leaf_keys = []
        self._collect_leaves(board, player, self.depth, leaf_boards, leaf_players, leaf_keys)

        # --- Batch DQN evaluation ---
        if leaf_boards:
            self._batch_eval(leaf_boards, leaf_players, leaf_keys)

        # --- Pass 2: alpha-beta search, leaf values come from TT ---
        best_score = -float("inf")
        best_actions = []
        alpha = -float("inf")
        beta = float("inf")

        for col in MOVE_ORDER:
            if col not in legal:
                continue

            row = self._drop_row(board, col)
            board[row, col] = player

            if self._check_win(board, row, col):
                board[row, col] = 0
                score = TERMINAL_WIN + PLY_BONUS * self.depth
            elif self._is_full(board):
                board[row, col] = 0
                score = 0.0
            else:
                score = -self._negamax(board, 3 - player, self.depth - 1, -beta, -alpha)
                board[row, col] = 0

            if score > best_score:
                best_score = score
                best_actions = [col]
            elif score == best_score:
                best_actions.append(col)

            alpha = max(alpha, score)
            # Don't prune the root — we want to find all tied best moves

        return _rand.choice(best_actions) if best_actions else legal[0]

    def load(self, path):
        self.rl_agent.load(path)

    # ---- Pass 1: leaf enumeration ----

    def _collect_leaves(self, board, player, depth, leaf_boards, leaf_players, leaf_keys):
        """Recursively enumerate all non-terminal leaf boards at `depth` plies.

        Terminal positions (win/draw) never need DQN evaluation, so they are
        NOT added. Duplicates (transpositions) are deduplicated via the TT.
        """
        if depth == 0:
            key = (board.tobytes(), player)
            if key not in self._tt:
                self._tt[key] = None  # placeholder until batch eval
                leaf_boards.append(board.copy())
                leaf_players.append(player)
                leaf_keys.append(key)
            return

        legal = [c for c in range(COLS) if board[0, c] == 0]
        if not legal:
            return  # draw — no DQN eval needed

        for col in legal:
            row = self._drop_row(board, col)
            board[row, col] = player

            if self._check_win(board, row, col):
                # Terminal win — no DQN eval, _negamax handles it directly
                pass
            elif self._is_full(board):
                pass  # terminal draw
            else:
                self._collect_leaves(
                    board, 3 - player, depth - 1,
                    leaf_boards, leaf_players, leaf_keys,
                )
            board[row, col] = 0

    def _batch_eval(self, leaf_boards, leaf_players, leaf_keys):
        """One big DQN forward pass over all leaf positions."""
        n = len(leaf_boards)
        states = np.zeros((n, 2, ROWS, COLS), dtype=np.float32)
        for i in range(n):
            b = leaf_boards[i]
            p = leaf_players[i]
            states[i, 0] = (b == p)
            states[i, 1] = (b == (3 - p))

        states_t = torch.from_numpy(states).to(self._device)
        with torch.no_grad():
            q_values = self._q_net(states_t).cpu().numpy()

        # Cache per-leaf value (max over legal actions)
        for i in range(n):
            board = leaf_boards[i]
            legal = [c for c in range(COLS) if board[0, c] == 0]
            value = float(max(q_values[i, c] for c in legal))
            self._tt[leaf_keys[i]] = value

    # ---- Pass 2: alpha-beta search with cached leaf values ----

    def _negamax(self, board, player, depth, alpha, beta):
        """Alpha-beta search. Leaf values come from the TT (pre-computed)."""
        if depth == 0:
            # Should always be in TT from pass 1 (unless pruned out, which
            # can't happen during pass 1 since it doesn't prune)
            key = (board.tobytes(), player)
            v = self._tt.get(key)
            return v if v is not None else 0.0

        legal = [c for c in range(COLS) if board[0, c] == 0]
        if not legal:
            return 0.0

        best = -float("inf")
        for col in MOVE_ORDER:
            if col not in legal:
                continue

            row = self._drop_row(board, col)
            board[row, col] = player

            if self._check_win(board, row, col):
                board[row, col] = 0
                return TERMINAL_WIN + PLY_BONUS * depth

            if self._is_full(board):
                board[row, col] = 0
                value = 0.0
            else:
                value = -self._negamax(board, 3 - player, depth - 1, -beta, -alpha)
                board[row, col] = 0

            if value > best:
                best = value
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break

        return best

    # ---- board helpers ----

    @staticmethod
    def _drop_row(board, col):
        for r in range(ROWS - 1, -1, -1):
            if board[r, col] == 0:
                return r
        raise ValueError(f"Column {col} is full")

    @staticmethod
    def _is_full(board):
        return bool(np.all(board[0, :] != 0))

    @staticmethod
    def _check_win(board, row, col):
        player = board[row, col]
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1
            for sign in (1, -1):
                for step in range(1, WIN_LENGTH):
                    r = row + dr * step * sign
                    c = col + dc * step * sign
                    if 0 <= r < ROWS and 0 <= c < COLS and board[r, c] == player:
                        count += 1
                    else:
                        break
            if count >= WIN_LENGTH:
                return True
        return False
