"""Interactive Connect 4 — play against any agent or opponent."""

import argparse
import sys
import os

from envs.connect4 import Connect4Env, ROWS, COLS


# ── Pretty board rendering ──────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
DIM = "\033[2m"


def render_board(env, last_action=None):
    """Print a coloured board to the terminal."""
    symbols = {0: "·", 1: f"{RED}●{RESET}", 2: f"{YELLOW}●{RESET}"}
    print()
    # Column headers
    header = "  ".join(
        f"{BOLD}{CYAN}{c}{RESET}" if c in env.get_legal_actions() else f"{DIM}{c}{RESET}"
        for c in range(COLS)
    )
    print(f"  {header}")
    print(f"  {'─' * (COLS * 3 - 1)}")
    for row in range(ROWS):
        cells = []
        for col in range(COLS):
            sym = symbols[env.board[row, col]]
            # Highlight last-placed piece
            if last_action == col and env.board[row, col] != 0:
                # Check if this is the topmost piece in this column
                is_top = (row == 0) or (env.board[row - 1, col] == 0)
                if is_top:
                    sym = f"\033[4m{sym}\033[24m"  # underline
            cells.append(sym)
        print(f"  {'  '.join(cells)}")
    print(f"  {'─' * (COLS * 3 - 1)}")
    print()


# ── Player types ─────────────────────────────────────────────────────

def human_player(env):
    """Prompt the human for a column choice."""
    legal = env.get_legal_actions()
    while True:
        try:
            raw = input(f"  Your move (columns {', '.join(map(str, legal))}): ").strip()
            if raw.lower() in ("q", "quit", "exit"):
                print("\nGoodbye!")
                sys.exit(0)
            col = int(raw)
            if col in legal:
                return col
            print(f"  Column {col} is not a legal move.")
        except ValueError:
            print("  Please enter a column number.")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            sys.exit(0)


def make_opponent(name, model_path=None):
    """Return (select_action_fn, display_name) for the given opponent type."""
    name = name.lower()

    if name == "human":
        return human_player, "Human"

    elif name == "random":
        from opponents.random_opponent import RandomOpponent
        opp = RandomOpponent()
        return opp.select_action, "Random"

    elif name == "heuristic":
        from opponents.heuristic_opponent import HeuristicOpponent
        opp = HeuristicOpponent()
        return opp.select_action, "Heuristic"

    elif name == "minimax":
        from opponents.minimax_opponent import MinimaxOpponent
        opp = MinimaxOpponent(depth=4)
        return opp.select_action, "Minimax (depth 4)"

    elif name == "dqn":
        path = model_path or _find_model("dqn")
        if path is None:
            print("Error: No DQN model found. Train one first.")
            sys.exit(1)
        agent = load_agent("dqn", model_path=path)
        print(f"  Loaded DQN model from {path}")
        return lambda env: agent.select_action(env, greedy=True), f"DQN ({os.path.basename(path)})"

    elif name == "dqn-hybrid":
        path = model_path or _find_model("dqn")
        if path is None:
            print("Error: No DQN model found. Train one first.")
            sys.exit(1)
        agent = load_agent("dqn-hybrid", model_path=path)
        print(f"  Loaded DQN-Hybrid from {path}")
        return lambda env: agent.select_action(env, greedy=True), f"DQN-Hybrid ({os.path.basename(path)})"

    else:
        print(f"Unknown player type: {name}")
        print("Choose from: human, random, heuristic, minimax, dqn, dqn-hybrid")
        sys.exit(1)


def _find_model(prefix):
    """Auto-detect a saved model in models/<prefix>/ (or fall back to results/)."""
    base = os.path.dirname(os.path.abspath(__file__))

    # First: check models/<prefix>/latest.pt
    latest = os.path.join(base, "models", prefix, "latest.pt")
    if os.path.isfile(latest):
        return latest

    # Second: any .pt in models/<prefix>/
    model_dir = os.path.join(base, "models", prefix)
    if os.path.isdir(model_dir):
        candidates = [
            os.path.join(model_dir, f)
            for f in os.listdir(model_dir)
            if f.endswith(".pt")
        ]
        if candidates:
            return max(candidates, key=os.path.getmtime)

    # Fallback: legacy results/ directory
    results_dir = os.path.join(base, "results")
    if os.path.isdir(results_dir):
        candidates = [
            os.path.join(results_dir, f)
            for f in os.listdir(results_dir)
            if f.startswith(prefix) and f.endswith(".pt")
        ]
        if candidates:
            return max(candidates, key=os.path.getmtime)

    return None


# ── Game loop ────────────────────────────────────────────────────────

def play_game(p1_fn, p2_fn, p1_name, p2_name):
    """Play a single game and return the winner (1, 2, or 0 for draw)."""
    env = Connect4Env()
    env.reset()

    players = {1: (p1_fn, p1_name), 2: (p2_fn, p2_name)}
    last_action = None

    print(f"\n{'=' * 40}")
    print(f"  {RED}● Player 1{RESET}: {p1_name}")
    print(f"  {YELLOW}● Player 2{RESET}: {p2_name}")
    print(f"{'=' * 40}")

    render_board(env)

    while not env.done:
        fn, name = players[env.current_player]
        color = RED if env.current_player == 1 else YELLOW
        print(f"  {color}● {name}'s turn{RESET}")

        action = fn(env)
        _, reward, done, info = env.step(action)
        last_action = action

        render_board(env, last_action=action)

        if done:
            winner = info["winner"]
            if winner == 0:
                print(f"  {BOLD}It's a draw!{RESET}\n")
            else:
                wname = players[winner][1]
                wcolor = RED if winner == 1 else YELLOW
                print(f"  {BOLD}{wcolor}● {wname} wins!{RESET}\n")
            return winner

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Play Connect 4 against any agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Player types:
  human       You play (interactive input)
  random      Uniform random legal moves
  heuristic   Rule-based: win/block/center
  minimax     Alpha-beta search (depth 4)
  dqn         Trained DQN agent

Examples:
  python play.py                          # You vs Heuristic
  python play.py -p1 human -p2 dqn       # You vs DQN
  python play.py -p1 dqn -p2 minimax     # Watch DQN vs Minimax
  python play.py -p1 human -p2 human     # Local 2-player
""",
    )
    parser.add_argument("-p1", "--player1", default="human", help="Player 1 type (default: human)")
    parser.add_argument("-p2", "--player2", default="heuristic", help="Player 2 type (default: heuristic)")
    parser.add_argument("--p1-model", default=None, help="Path to player 1 model checkpoint (.pt)")
    parser.add_argument("--p2-model", default=None, help="Path to player 2 model checkpoint (.pt)")
    parser.add_argument("-n", "--games", type=int, default=1, help="Number of games to play")
    parser.add_argument("--swap", action="store_true", help="Alternate who goes first each game")

    args = parser.parse_args()

    p1_fn, p1_name = make_opponent(args.player1, args.p1_model)
    p2_fn, p2_name = make_opponent(args.player2, args.p2_model)

    results = {1: 0, 2: 0, 0: 0}

    for game_num in range(1, args.games + 1):
        if args.games > 1:
            print(f"\n{'━' * 40}")
            print(f"  Game {game_num}/{args.games}")
            print(f"{'━' * 40}")

        # Alternate first player if --swap
        if args.swap and game_num % 2 == 0:
            winner = play_game(p2_fn, p1_fn, p2_name, p1_name)
            # Map winner back to original player identity
            if winner == 1:
                results[2] += 1
            elif winner == 2:
                results[1] += 1
            else:
                results[0] += 1
        else:
            winner = play_game(p1_fn, p2_fn, p1_name, p2_name)
            results[winner] += 1

    if args.games > 1:
        print(f"\n{'=' * 40}")
        print(f"  Results over {args.games} games:")
        print(f"  {p1_name}: {results[1]} wins")
        print(f"  {p2_name}: {results[2]} wins")
        print(f"  Draws: {results[0]}")
        print(f"{'=' * 40}\n")


if __name__ == "__main__":
    main()
