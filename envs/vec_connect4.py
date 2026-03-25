"""Vectorized Connect 4: run N games in parallel with numpy."""

import random as _rand

import numpy as np
from .connect4 import ROWS, COLS, WIN_LENGTH
from opponents.minimax_opponent import MinimaxOpponent
from opponents.self_play_opponent import SelfPlayOpponent


class VecConnect4Env:
    """N independent Connect 4 games running in lockstep.

    All games are always at the agent's turn. After the agent steps,
    the opponent automatically responds on games that didn't end.
    Finished games auto-reset.
    """

    def __init__(self, n_envs, opponent):
        self.n_envs = n_envs
        self.opponent = opponent
        self._batch_minimax = isinstance(opponent, MinimaxOpponent)
        self._batch_selfplay = isinstance(opponent, SelfPlayOpponent)
        self._can_batch = self._batch_minimax or self._batch_selfplay
        # boards[i] is a (6,7) int8 array: 0=empty, 1=P1, 2=P2
        self.boards = np.zeros((n_envs, ROWS, COLS), dtype=np.int8)
        # which player the agent controls in each env (1 or 2)
        self.agent_player = np.ones(n_envs, dtype=np.int8)
        self.current_player = np.ones(n_envs, dtype=np.int8)
        self.dones = np.zeros(n_envs, dtype=bool)

    def reset_all(self):
        """Reset all envs, randomize sides, let opponent go first if needed."""
        self.boards[:] = 0
        # Half play as P1, half as P2
        self.agent_player = np.array(
            [1 if i % 2 == 0 else 2 for i in range(self.n_envs)], dtype=np.int8
        )
        self.current_player[:] = 1
        self.dones[:] = False

        # Opponent goes first where agent is P2
        opp_first = [i for i in range(self.n_envs) if self.agent_player[i] == 2]
        if opp_first and self._can_batch:
            self.current_player[opp_first] = 1  # P1 goes first
            actions = self._batch_opponent_actions(opp_first)
            for j, i in enumerate(opp_first):
                self._drop(i, int(actions[j]), 1)
        else:
            for i in opp_first:
                self._opponent_move(i)

        return self.get_states()

    def _reset_env(self, i):
        """Reset a single env after a game ends."""
        self.boards[i] = 0
        self.agent_player[i] = 3 - self.agent_player[i]  # swap sides
        self.current_player[i] = 1
        self.dones[i] = False

        # If opponent goes first in this new game
        if self.agent_player[i] == 2:
            self._opponent_move(i)

    def get_states(self):
        """Return (N, 6, 7, 2) float32 from each agent's perspective."""
        states = np.zeros((self.n_envs, ROWS, COLS, 2), dtype=np.float32)
        for i in range(self.n_envs):
            ap = self.agent_player[i]
            states[i, :, :, 0] = (self.boards[i] == ap)
            states[i, :, :, 1] = (self.boards[i] == (3 - ap))
        return states

    def get_legal_actions_batch(self):
        """Return list of legal action lists for each env."""
        return [
            [c for c in range(COLS) if self.boards[i, 0, c] == 0]
            for i in range(self.n_envs)
        ]

    def step(self, actions):
        """Agent plays in all envs. Opponent auto-responds.

        Args:
            actions: array of N column indices

        Returns:
            next_states:  (N, 6, 7, 2) — states after agent+opponent both moved
            rewards:      (N,) float — from agent's perspective
            dones:        (N,) bool — whether episode ended
            next_legals:  list of legal action lists for non-terminal next states
        """
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=bool)
        next_legals = [[] for _ in range(self.n_envs)]

        # --- Agent moves (all envs) ---
        needs_opp = []  # indices that need opponent response
        for i in range(self.n_envs):
            col = actions[i]
            ap = self.agent_player[i]
            row = self._drop(i, col, ap)
            if self._check_win(i, row, col):
                rewards[i] = 1.0
                dones[i] = True
            elif self._is_full(i):
                rewards[i] = 0.0
                dones[i] = True
            else:
                self.current_player[i] = 3 - ap
                needs_opp.append(i)

        # --- Opponent moves (batch if possible, sequential otherwise) ---
        if needs_opp:
            if self._batch_minimax:
                # Batch get_scores: parallel minimax on all boards
                boards = np.stack([self.boards[i] for i in needs_opp])
                players = np.array([self.current_player[i] for i in needs_opp], dtype=np.int8)
                all_scores, solved_flags = self.opponent.get_scores_batch(boards, players)

                for j, i in enumerate(needs_opp):
                    scores = all_scores[j]
                    legal = [c for c in range(COLS) if self.boards[i, 0, c] == 0]
                    best_score = max(scores[c] for c in legal)

                    if solved_flags[j] and best_score >= 100:
                        # Opponent has a forced win — end early
                        rewards[i] = -1.0
                        dones[i] = True
                        continue

                    # Pick randomly among best moves (non-deterministic)
                    best_actions = [c for c in legal if scores[c] == best_score]
                    col = _rand.choice(best_actions)
                    row = self._drop(i, col, self.current_player[i])
                    if self._check_win(i, row, col):
                        rewards[i] = -1.0
                        dones[i] = True
                    elif self._is_full(i):
                        rewards[i] = 0.0
                        dones[i] = True
                    else:
                        self.current_player[i] = self.agent_player[i]
                        next_legals[i] = [c for c in range(COLS) if self.boards[i, 0, c] == 0]
            elif self._batch_selfplay:
                opp_actions = self._batch_opponent_actions(needs_opp)
                for j, i in enumerate(needs_opp):
                    col = int(opp_actions[j])
                    row = self._drop(i, col, self.current_player[i])
                    if self._check_win(i, row, col):
                        rewards[i] = -1.0
                        dones[i] = True
                    elif self._is_full(i):
                        rewards[i] = 0.0
                        dones[i] = True
                    else:
                        self.current_player[i] = self.agent_player[i]
                        next_legals[i] = [c for c in range(COLS) if self.boards[i, 0, c] == 0]
            else:
                # Sequential opponent moves (random, heuristic, etc.)
                for i in needs_opp:
                    col = self._opponent_move(i)
                    row = self._last_drop_row
                    if self._check_win(i, row, col):
                        rewards[i] = -1.0
                        dones[i] = True
                    elif self._is_full(i):
                        rewards[i] = 0.0
                        dones[i] = True
                    else:
                        self.current_player[i] = self.agent_player[i]
                        next_legals[i] = [c for c in range(COLS) if self.boards[i, 0, c] == 0]

        # Get next states before resetting
        next_states = self.get_states()

        # Auto-reset finished games
        reset_indices = [i for i in range(self.n_envs) if dones[i]]
        opp_first_indices = []
        for i in reset_indices:
            self.boards[i] = 0
            self.agent_player[i] = 3 - self.agent_player[i]
            self.current_player[i] = 1
            self.dones[i] = False
            if self.agent_player[i] == 2:
                opp_first_indices.append(i)

        # Batch opponent-first moves for reset envs
        if opp_first_indices:
            if self._can_batch:
                actions = self._batch_opponent_actions(opp_first_indices)
                for j, i in enumerate(opp_first_indices):
                    self._drop(i, int(actions[j]), 1)
            else:
                for i in opp_first_indices:
                    self._opponent_move(i)

        return next_states, rewards, dones, next_legals

    # ---- internal helpers ----

    _last_drop_row = 0  # stash for opponent's drop row

    def _drop(self, env_idx, col, player):
        """Drop a piece, return the row it landed on."""
        board = self.boards[env_idx]
        for row in range(ROWS - 1, -1, -1):
            if board[row, col] == 0:
                board[row, col] = player
                return row
        raise ValueError(f"Column {col} is full in env {env_idx}")

    def _batch_opponent_actions(self, indices):
        """Get opponent actions for all envs in indices via a single batched call."""
        if self._batch_minimax:
            boards = np.stack([self.boards[i] for i in indices])
            players = np.array([self.current_player[i] for i in indices], dtype=np.int8)
            return self.opponent.select_actions_batch(boards, players)
        else:
            # Self-play: build states from opponent's perspective and legal actions
            states = np.zeros((len(indices), ROWS, COLS, 2), dtype=np.float32)
            legals = []
            for j, i in enumerate(indices):
                opp = self.current_player[i]
                states[j, :, :, 0] = (self.boards[i] == opp)
                states[j, :, :, 1] = (self.boards[i] == (3 - opp))
                legals.append([c for c in range(COLS) if self.boards[i, 0, c] == 0])
            return self.opponent.select_actions_batch(states, legals)

    def _opponent_move(self, env_idx):
        """Let the opponent make a move. Returns the column chosen."""
        # Build a lightweight env-like object for the opponent
        proxy = _EnvProxy(self.boards[env_idx], self.current_player[env_idx])
        col = self.opponent.select_action(proxy)
        row = self._drop(env_idx, col, self.current_player[env_idx])
        self._last_drop_row = row
        return col

    def _check_win(self, env_idx, row, col):
        """Check if the piece at (row, col) wins."""
        board = self.boards[env_idx]
        player = board[row, col]
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1
            for sign in [1, -1]:
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

    def _is_full(self, env_idx):
        return np.all(self.boards[env_idx, 0, :] != 0)


class _EnvProxy:
    """Minimal env-like object so opponents can call select_action(env)."""

    def __init__(self, board, current_player):
        self.board = board
        self.current_player = current_player

    def get_legal_actions(self):
        return [c for c in range(COLS) if self.board[0, c] == 0]

    def get_state(self):
        state = np.zeros((ROWS, COLS, 2), dtype=np.float32)
        state[:, :, 0] = (self.board == self.current_player)
        state[:, :, 1] = (self.board == (3 - self.current_player))
        return state

    def _check_win(self, row, col):
        """Needed by HeuristicOpponent._would_win."""
        player = self.board[row, col]
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1
            for sign in [1, -1]:
                for step in range(1, 4):
                    r = row + dr * step * sign
                    c = col + dc * step * sign
                    if 0 <= r < ROWS and 0 <= c < COLS and self.board[r, c] == player:
                        count += 1
                    else:
                        break
            if count >= 4:
                return True
        return False
