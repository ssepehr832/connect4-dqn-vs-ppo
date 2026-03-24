import numpy as np


ROWS = 6
COLS = 7
WIN_LENGTH = 4


class Connect4Env:
    """Connect 4 environment with a Gym-like API.

    Board encoding:
        0 = empty, 1 = player 1, 2 = player 2

    State representation for RL (get_state):
        6x7x2 binary tensor — channel 0 is current player's pieces,
        channel 1 is opponent's pieces.
    """

    def __init__(self):
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.current_player = 1  # 1 or 2
        self.done = False
        self.winner = None  # None, 1, 2, or 0 (draw)

    def reset(self):
        """Reset the board and return the initial state."""
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.get_state()

    def get_legal_actions(self):
        """Return list of columns that are not full."""
        return [c for c in range(COLS) if self.board[0, c] == 0]

    def get_state(self):
        """Return a 6x7x2 binary array from current player's perspective."""
        state = np.zeros((ROWS, COLS, 2), dtype=np.float32)
        state[:, :, 0] = (self.board == self.current_player).astype(np.float32)
        state[:, :, 1] = (self.board == self._other_player()).astype(np.float32)
        return state

    def step(self, action):
        """Play a move in the given column.

        Returns:
            state: next state (from the *next* player's perspective)
            reward: +1 if current player wins, -1 if loss, 0 otherwise
                    (reward is from the perspective of the player who just moved)
            done: whether the game is over
            info: dict with 'winner' key
        """
        if self.done:
            raise ValueError("Game is already over.")
        if action not in self.get_legal_actions():
            raise ValueError(f"Column {action} is not a legal move.")

        # Drop the piece
        row = self._get_drop_row(action)
        self.board[row, action] = self.current_player

        # Check for win
        if self._check_win(row, action):
            self.done = True
            self.winner = self.current_player
            reward = 1.0
        elif len(self.get_legal_actions()) == 0:
            self.done = True
            self.winner = 0  # draw
            reward = 0.0
        else:
            reward = 0.0

        # Switch player
        self.current_player = self._other_player()

        state = self.get_state()
        info = {"winner": self.winner}
        return state, reward, self.done, info

    def clone(self):
        """Return a deep copy of the environment."""
        env = Connect4Env()
        env.board = self.board.copy()
        env.current_player = self.current_player
        env.done = self.done
        env.winner = self.winner
        return env

    def render(self):
        """Print the board to stdout."""
        symbols = {0: ".", 1: "X", 2: "O"}
        print()
        for row in range(ROWS):
            print(" ".join(symbols[self.board[row, c]] for c in range(COLS)))
        print(" ".join(str(c) for c in range(COLS)))
        print()

    # ---- internal helpers ----

    def _other_player(self):
        return 3 - self.current_player

    def _get_drop_row(self, col):
        """Return the lowest empty row in the given column."""
        for row in range(ROWS - 1, -1, -1):
            if self.board[row, col] == 0:
                return row
        raise ValueError(f"Column {col} is full.")

    def _check_win(self, row, col):
        """Check if the last move at (row, col) resulted in a win."""
        player = self.board[row, col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horiz, vert, diag, anti-diag
        for dr, dc in directions:
            count = 1
            # Check in positive direction
            for i in range(1, WIN_LENGTH):
                r, c = row + dr * i, col + dc * i
                if 0 <= r < ROWS and 0 <= c < COLS and self.board[r, c] == player:
                    count += 1
                else:
                    break
            # Check in negative direction
            for i in range(1, WIN_LENGTH):
                r, c = row - dr * i, col - dc * i
                if 0 <= r < ROWS and 0 <= c < COLS and self.board[r, c] == player:
                    count += 1
                else:
                    break
            if count >= WIN_LENGTH:
                return True
        return False
