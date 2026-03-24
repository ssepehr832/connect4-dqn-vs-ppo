import random
from envs.connect4 import ROWS, COLS


class HeuristicOpponent:
    """Rule-based opponent: win if possible, block opponent win, prefer center."""

    def select_action(self, env):
        legal = env.get_legal_actions()
        me = env.current_player
        opp = 3 - me

        # 1. Win if possible
        for col in legal:
            if self._would_win(env, col, me):
                return col

        # 2. Block opponent win
        for col in legal:
            if self._would_win(env, col, opp):
                return col

        # 3. Prefer center columns (3, 2, 4, 1, 5, 0, 6)
        center_priority = sorted(legal, key=lambda c: abs(c - COLS // 2))
        return center_priority[0]

    def _would_win(self, env, col, player):
        """Check if dropping a piece in col would win for player."""
        row = self._get_drop_row(env.board, col)
        if row is None:
            return False
        # Temporarily place the piece
        env.board[row, col] = player
        win = env._check_win(row, col)
        env.board[row, col] = 0  # undo
        return win

    def _get_drop_row(self, board, col):
        for row in range(ROWS - 1, -1, -1):
            if board[row, col] == 0:
                return row
        return None
