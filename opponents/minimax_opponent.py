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
        cmd = ["gcc", "-O2", "-shared", "-o", lib_path, src_path]
    elif system == "Windows":
        cmd = ["gcc", "-O2", "-shared", "-o", lib_path, src_path]
    else:
        cmd = ["gcc", "-O2", "-shared", "-fPIC", "-o", lib_path, src_path]

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
