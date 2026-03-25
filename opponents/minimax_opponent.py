import ctypes
import os
import platform
import subprocess
import numpy as np


_LIB = None


def _get_lib_path():
    """Return the path to the compiled shared library."""
    base = os.path.dirname(os.path.abspath(__file__))
    system = platform.system()
    if system == "Darwin":
        return os.path.join(base, "minimax_engine.dylib")
    elif system == "Windows":
        return os.path.join(base, "minimax_engine.dll")
    else:
        return os.path.join(base, "minimax_engine.so")


def _compile_engine():
    """Compile the C engine if the shared library doesn't exist."""
    lib_path = _get_lib_path()
    src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "minimax_engine.c")

    if os.path.exists(lib_path):
        # Recompile only if source is newer than library
        if os.path.getmtime(src_path) <= os.path.getmtime(lib_path):
            return lib_path

    system = platform.system()
    if system == "Darwin":
        cmd = ["gcc", "-O3", "-shared", "-lpthread", "-o", lib_path, src_path]
    elif system == "Windows":
        cmd = ["gcc", "-O3", "-shared", "-lpthread", "-o", lib_path, src_path]
    else:
        cmd = ["gcc", "-O3", "-shared", "-fPIC", "-lpthread", "-o", lib_path, src_path]

    print(f"Compiling minimax engine: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(
            "C compiler (gcc) not found. Please install it:\n"
            "  macOS:   xcode-select --install\n"
            "  Ubuntu:  sudo apt install build-essential\n"
            "  Windows: install MinGW-w64 and add gcc to PATH\n"
            "Or run 'bash build.sh' manually after installing a compiler."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to compile minimax engine:\n{e.stderr}"
        )
    return lib_path


def _load_lib():
    """Load (and auto-compile if needed) the C minimax library."""
    global _LIB
    if _LIB is not None:
        return _LIB

    lib_path = _compile_engine()
    _LIB = ctypes.CDLL(lib_path)

    # int minimax_best_action(int *board_flat, int depth, int player)
    _LIB.minimax_best_action.argtypes = [
        ctypes.POINTER(ctypes.c_int),  # board_flat
        ctypes.c_int,                   # depth
        ctypes.c_int,                   # player
    ]
    _LIB.minimax_best_action.restype = ctypes.c_int

    # void minimax_batch(int *boards, int *depths, int *players, int *results, int n)
    _LIB.minimax_batch.argtypes = [
        ctypes.POINTER(ctypes.c_int),  # boards_flat (N*42)
        ctypes.POINTER(ctypes.c_int),  # depths (N)
        ctypes.POINTER(ctypes.c_int),  # players (N)
        ctypes.POINTER(ctypes.c_int),  # results (N)
        ctypes.c_int,                   # n
    ]
    _LIB.minimax_batch.restype = None

    # int minimax_get_scores(int *board_flat, int depth, int player, int *scores)
    _LIB.minimax_get_scores.argtypes = [
        ctypes.POINTER(ctypes.c_int),  # board_flat
        ctypes.c_int,                   # depth
        ctypes.c_int,                   # player
        ctypes.POINTER(ctypes.c_int),  # scores (output, 7 ints)
    ]
    _LIB.minimax_get_scores.restype = ctypes.c_int  # 1 if solved, 0 otherwise

    # void minimax_batch_scores(int *boards, int *depths, int *players,
    #                           int *all_scores, int *solved_flags, int n)
    _LIB.minimax_batch_scores.argtypes = [
        ctypes.POINTER(ctypes.c_int),  # boards_flat (N*42)
        ctypes.POINTER(ctypes.c_int),  # depths (N)
        ctypes.POINTER(ctypes.c_int),  # players (N)
        ctypes.POINTER(ctypes.c_int),  # all_scores (N*7)
        ctypes.POINTER(ctypes.c_int),  # solved_flags (N)
        ctypes.c_int,                   # n
    ]
    _LIB.minimax_batch_scores.restype = None

    return _LIB


class MinimaxOpponent:
    """Alpha-beta pruning minimax opponent (C-accelerated)."""

    def __init__(self, depth=5):
        self.depth = depth
        self._lib = _load_lib()

    def select_action(self, env):
        # Convert board to flat C-compatible int array
        board_flat = env.board.astype(np.intc).flatten()
        board_ptr = board_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        return self._lib.minimax_best_action(board_ptr, self.depth, env.current_player)

    def select_actions_batch(self, boards, players):
        """Compute best actions for N boards in parallel using pthreads.

        Args:
            boards: (N, 6, 7) int array of board states
            players: (N,) int array of current players

        Returns:
            (N,) int array of best columns
        """
        n = len(boards)
        boards_flat = np.ascontiguousarray(boards.reshape(n, -1), dtype=np.intc)
        depths = np.full(n, self.depth, dtype=np.intc)
        players_arr = np.ascontiguousarray(players, dtype=np.intc)
        results = np.zeros(n, dtype=np.intc)

        self._lib.minimax_batch(
            boards_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            depths.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            players_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            results.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(n),
        )
        return results

    def get_scores(self, env):
        """Return (is_solved, scores) where scores is a length-7 array.

        is_solved: True if any column leads to a forced win/loss
        scores[c]: minimax score for column c (very negative = illegal)
        """
        board_flat = env.board.astype(np.intc).flatten()
        board_ptr = board_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        scores = (ctypes.c_int * 7)()
        is_solved = self._lib.minimax_get_scores(
            board_ptr, self.depth, env.current_player, scores
        )
        return bool(is_solved), [scores[i] for i in range(7)]

    def get_scores_batch(self, boards, players):
        """Compute per-column scores for N boards in parallel.

        Args:
            boards: (N, 6, 7) int array of board states
            players: (N,) int array of current players

        Returns:
            all_scores:   (N, 7) int array — minimax score per column
            solved_flags: (N,) bool array — True if position is solved
        """
        n = len(boards)
        boards_flat = np.ascontiguousarray(boards.reshape(n, -1), dtype=np.intc)
        depths = np.full(n, self.depth, dtype=np.intc)
        players_arr = np.ascontiguousarray(players, dtype=np.intc)
        all_scores = np.zeros((n, 7), dtype=np.intc)
        solved_flags = np.zeros(n, dtype=np.intc)

        self._lib.minimax_batch_scores(
            boards_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            depths.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            players_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            all_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            solved_flags.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(n),
        )
        return all_scores, solved_flags.astype(bool)
