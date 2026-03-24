#!/usr/bin/env bash
# Build the C minimax engine shared library.
# Usage: bash build.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$SCRIPT_DIR/opponents/minimax_engine.c"

case "$(uname -s)" in
    Darwin)
        OUT="$SCRIPT_DIR/opponents/minimax_engine.dylib"
        gcc -O2 -shared -o "$OUT" "$SRC"
        ;;
    MINGW*|MSYS*|CYGWIN*)
        OUT="$SCRIPT_DIR/opponents/minimax_engine.dll"
        gcc -O2 -shared -o "$OUT" "$SRC"
        ;;
    *)
        OUT="$SCRIPT_DIR/opponents/minimax_engine.so"
        gcc -O2 -shared -fPIC -o "$OUT" "$SRC"
        ;;
esac

echo "Built: $OUT"
