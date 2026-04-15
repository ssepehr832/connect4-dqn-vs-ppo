"""Vectorized Connect 4: run N games in parallel with numpy."""

import random as _rand

import numpy as np
from .connect4 import ROWS, COLS, WIN_LENGTH
from opponents.minimax_opponent import MinimaxOpponent
from opponents.self_play_opponent import SelfPlayOpponent

# Reward constants — tweak these to shape agent behavior
REWARD_WIN = 1.0
REWARD_WIN_BONUS = 0.3  # max bonus for fast wins (linearly scaled)
REWARD_DRAW = 0.2
REWARD_LOSS = -1.1


def win_reward(board):
    """Win reward with bonus for faster wins. Range: REWARD_WIN to REWARD_WIN + REWARD_WIN_BONUS."""
    pieces = int(np.count_nonzero(board))
    # pieces ranges from 7 (earliest win) to 41 (latest win)
    # bonus = REWARD_WIN_BONUS * (1 - (pieces - 7) / (41 - 7))
    speed = max(0.0, 1.0 - (pieces - 7) / 34.0)
    return REWARD_WIN + REWARD_WIN_BONUS * speed


class VecConnect4Env:
    """N independent Connect 4 games running in lockstep.

    All games are always at the agent's turn. After the agent steps,
    the opponent automatically responds on games that didn't end.
    Finished games auto-reset.
    """

    def __init__(self, n_envs, opponent, arbiter=None, arbiter_min_pieces=12):
        """
        Args:
            n_envs: number of parallel games
            opponent: opponent object
            arbiter: optional MinimaxOpponent used to end games early when
                     the position is solved (forced win/loss/draw). Works
                     with any opponent type (designed for self-play).
            arbiter_min_pieces: minimum pieces on board before arbiter checks
                                kick in (default 12 = ~6 moves per player)
        """
        self.n_envs = n_envs
        self.opponent = opponent
        self._batch_minimax = isinstance(opponent, MinimaxOpponent)
        self._batch_selfplay = isinstance(opponent, SelfPlayOpponent)
        self._can_batch = self._batch_minimax or self._batch_selfplay
        self.arbiter = arbiter
        self.arbiter_min_pieces = arbiter_min_pieces
        # boards[i] is a (6,7) int8 array: 0=empty, 1=P1, 2=P2
        self.boards = np.zeros((n_envs, ROWS, COLS), dtype=np.int8)
        # which player the agent controls in each env (1 or 2)
        self.agent_player = np.ones(n_envs, dtype=np.int8)
        self.current_player = np.ones(n_envs, dtype=np.int8)
        self.dones = np.zeros(n_envs, dtype=bool)
        # Move tracking for game uniqueness detection
        self._move_seqs = [[] for _ in range(n_envs)]
        self._finished_hashes = []  # hashes of completed games since last drain
        self._finished_records = []  # structured episode summaries since last drain

    def reset_all(self):
        """Reset all envs, randomize sides, let opponent go first if needed."""
        self.boards[:] = 0
        # Half play as P1, half as P2
        self.agent_player = np.array(
            [1 if i % 2 == 0 else 2 for i in range(self.n_envs)], dtype=np.int8
        )
        self.current_player[:] = 1
        self.dones[:] = False
        self._move_seqs = [[] for _ in range(self.n_envs)]
        self._finished_hashes = []
        self._finished_records = []

        # Opponent goes first where agent is P2
        opp_first = [i for i in range(self.n_envs) if self.agent_player[i] == 2]
        if opp_first and self._can_batch:
            self.current_player[opp_first] = 1  # P1 goes first
            actions = self._batch_opponent_actions(opp_first)
            for j, i in enumerate(opp_first):
                col = int(actions[j])
                self._drop(i, col, 1)
                self._move_seqs[i].append(col)
        else:
            for i in opp_first:
                col = self._opponent_move(i)
                self._move_seqs[i].append(col)

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
        outcomes = [None for _ in range(self.n_envs)]
        termination_sources = [None for _ in range(self.n_envs)]

        # --- Agent moves (all envs) ---
        needs_opp = []  # indices that need opponent response
        for i in range(self.n_envs):
            col = actions[i]
            self._move_seqs[i].append(int(col))
            ap = self.agent_player[i]
            row = self._drop(i, col, ap)
            if self._check_win(i, row, col):
                rewards[i] = win_reward(self.boards[i])
                dones[i] = True
                outcomes[i] = "win"
                termination_sources[i] = "natural"
            elif self._is_full(i):
                rewards[i] = REWARD_DRAW
                dones[i] = True
                outcomes[i] = "draw"
                termination_sources[i] = "natural"
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

                    if solved_flags[j]:
                        if best_score >= 10000:
                            # Opponent has a forced win — agent loss
                            rewards[i] = REWARD_LOSS
                            dones[i] = True
                            outcomes[i] = "loss"
                            termination_sources[i] = "solved"
                            continue
                        if best_score <= -10000:
                            # Opponent has a forced loss — agent win
                            rewards[i] = win_reward(self.boards[i])
                            dones[i] = True
                            outcomes[i] = "win"
                            termination_sources[i] = "solved"
                            continue

                    # Pick randomly among best moves (non-deterministic)
                    best_actions = [c for c in legal if scores[c] == best_score]
                    col = _rand.choice(best_actions)
                    self._move_seqs[i].append(col)
                    row = self._drop(i, col, self.current_player[i])
                    if self._check_win(i, row, col):
                        rewards[i] = REWARD_LOSS
                        dones[i] = True
                        outcomes[i] = "loss"
                        termination_sources[i] = "natural"
                    elif self._is_full(i):
                        rewards[i] = REWARD_DRAW
                        dones[i] = True
                        outcomes[i] = "draw"
                        termination_sources[i] = "natural"
                    else:
                        self.current_player[i] = self.agent_player[i]
                        next_legals[i] = [c for c in range(COLS) if self.boards[i, 0, c] == 0]
            elif self._batch_selfplay:
                opp_actions = self._batch_opponent_actions(needs_opp)
                for j, i in enumerate(needs_opp):
                    col = int(opp_actions[j])
                    self._move_seqs[i].append(col)
                    row = self._drop(i, col, self.current_player[i])
                    if self._check_win(i, row, col):
                        rewards[i] = REWARD_LOSS
                        dones[i] = True
                        outcomes[i] = "loss"
                        termination_sources[i] = "natural"
                    elif self._is_full(i):
                        rewards[i] = REWARD_DRAW
                        dones[i] = True
                        outcomes[i] = "draw"
                        termination_sources[i] = "natural"
                    else:
                        self.current_player[i] = self.agent_player[i]
                        next_legals[i] = [c for c in range(COLS) if self.boards[i, 0, c] == 0]
            else:
                # Sequential opponent moves (random, heuristic, etc.)
                for i in needs_opp:
                    col = self._opponent_move(i)
                    self._move_seqs[i].append(col)
                    row = self._last_drop_row
                    if self._check_win(i, row, col):
                        rewards[i] = REWARD_LOSS
                        dones[i] = True
                        outcomes[i] = "loss"
                        termination_sources[i] = "natural"
                    elif self._is_full(i):
                        rewards[i] = REWARD_DRAW
                        dones[i] = True
                        outcomes[i] = "draw"
                        termination_sources[i] = "natural"
                    else:
                        self.current_player[i] = self.agent_player[i]
                        next_legals[i] = [c for c in range(COLS) if self.boards[i, 0, c] == 0]

        # --- Arbiter: use minimax to end games early if position is solved ---
        # Skip when opponent is minimax (already handled both-ways above)
        if self.arbiter is not None and not self._batch_minimax:
            # Find non-done envs with enough pieces on the board
            arbiter_candidates = [
                i for i in range(self.n_envs)
                if not dones[i] and np.count_nonzero(self.boards[i]) >= self.arbiter_min_pieces
            ]
            if arbiter_candidates:
                arb_boards = np.stack([self.boards[i] for i in arbiter_candidates])
                # Check from agent's perspective (it's agent's turn now)
                arb_players = np.array(
                    [self.agent_player[i] for i in arbiter_candidates], dtype=np.int8
                )
                all_scores, solved_flags = self.arbiter.get_scores_batch(arb_boards, arb_players)

                for j, i in enumerate(arbiter_candidates):
                    if not solved_flags[j]:
                        continue
                    legal = [c for c in range(COLS) if self.boards[i, 0, c] == 0]
                    best_score = max(all_scores[j][c] for c in legal)
                    if best_score >= 10000:
                        # Agent has a forced win
                        rewards[i] = win_reward(self.boards[i])
                        dones[i] = True
                        outcomes[i] = "win"
                        termination_sources[i] = "solved"
                    elif best_score <= -10000:
                        # Agent has a forced loss
                        rewards[i] = REWARD_LOSS
                        dones[i] = True
                        outcomes[i] = "loss"
                        termination_sources[i] = "solved"

        # Get next states before resetting
        next_states = self.get_states()

        # Auto-reset finished games
        reset_indices = [i for i in range(self.n_envs) if dones[i]]
        opp_first_indices = []
        for i in reset_indices:
            # Hash finished game for uniqueness tracking
            game_hash = hash(tuple(self._move_seqs[i]))
            self._finished_hashes.append(game_hash)
            outcome = outcomes[i]
            if outcome is None:
                if abs(float(rewards[i]) - REWARD_DRAW) < 1e-6:
                    outcome = "draw"
                elif rewards[i] > 0:
                    outcome = "win"
                else:
                    outcome = "loss"
            self._finished_records.append({
                "outcome": outcome,
                "reward": float(rewards[i]),
                "game_length": len(self._move_seqs[i]),
                "termination_source": termination_sources[i] or "natural",
                "game_hash": game_hash,
                "agent_player": int(self.agent_player[i]),
            })
            self._move_seqs[i] = []
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
                    col = int(actions[j])
                    self._drop(i, col, 1)
                    self._move_seqs[i].append(col)
            else:
                for i in opp_first_indices:
                    col = self._opponent_move(i)
                    self._move_seqs[i].append(col)

        return next_states, rewards, dones, next_legals

    def drain_game_hashes(self):
        """Return and clear the list of finished game hashes since last call."""
        hashes = self._finished_hashes
        self._finished_hashes = []
        return hashes

    def drain_game_records(self):
        """Return and clear structured episode summaries since last call."""
        records = self._finished_records
        self._finished_records = []
        return records

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
