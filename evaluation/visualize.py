"""Visualize DQN agent's play style and strategy.

Usage:
    python -m evaluation.visualize --agent dqn --opponent all
    python -m evaluation.visualize --agent dqn --opponent minimax --games 200
    python -m evaluation.visualize --agent dqn --opponent heuristic --save results/dqn_analysis.png
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

from envs.connect4 import Connect4Env, ROWS, COLS
from opponents import RandomOpponent, HeuristicOpponent, MinimaxOpponent


def load_agent(agent_type):
    if agent_type == "dqn":
        from agents.dqn.agent import DQNAgent
        agent = DQNAgent()
        path = "models/dqn/latest.pt"
        if not os.path.exists(path):
            print(f"Error: No saved model at {path}. Train first.")
            sys.exit(1)
        agent.load(path)
        agent.epsilon_start = 0.0
        agent.epsilon_end = 0.0
        return agent
    elif agent_type == "ppo":
        from agents.ppo.agent import PPOAgent
        agent = PPOAgent()
        path = "models/ppo/latest.pt"
        if not os.path.exists(path):
            print(f"Error: No saved model at {path}. Train first.")
            sys.exit(1)
        agent.load(path)
        return agent
    elif agent_type == "dqn-hybrid":
        from agents.dqn.agent import DQNAgent
        from agents.hybrid import HybridAgent
        rl_agent = DQNAgent()
        path = "models/dqn/latest.pt"
        if not os.path.exists(path):
            print(f"Error: No saved model at {path}. Train first.")
            sys.exit(1)
        rl_agent.load(path)
        rl_agent.epsilon_start = 0.0
        rl_agent.epsilon_end = 0.0
        return HybridAgent(rl_agent)
    elif agent_type == "ppo-hybrid":
        from agents.ppo.agent import PPOAgent
        from agents.hybrid import HybridAgent
        rl_agent = PPOAgent()
        path = "models/ppo/latest.pt"
        if not os.path.exists(path):
            print(f"Error: No saved model at {path}. Train first.")
            sys.exit(1)
        rl_agent.load(path)
        return HybridAgent(rl_agent)
    else:
        print(f"Unknown agent type: {agent_type}")
        sys.exit(1)


def make_opponent(name, depth=4, agent=None):
    if name == "random":
        return RandomOpponent()
    elif name == "heuristic":
        return HeuristicOpponent()
    elif name == "minimax":
        return MinimaxOpponent(depth=depth)
    elif name == "self":
        if agent is None:
            raise ValueError("Self-play requires an agent to copy")
        from evaluation.evaluate import EvalSelfPlayOpponent
        return EvalSelfPlayOpponent(agent, epsilon=0.05)
    else:
        raise ValueError(f"Unknown opponent: {name}")


def collect_game_data(agent, opponent, games=100):
    """Play games and collect detailed move data.

    Returns a list of game records, each containing:
        - moves: list of (player, col) tuples ('agent' or 'opponent')
        - result: 'win', 'loss', or 'draw'
        - length: number of total moves
        - agent_player: 1 or 2
        - boards_at_agent_moves: list of board snapshots when agent moved
        - q_values_at_agent_moves: list of Q-value arrays (DQN only)
    """
    env = Connect4Env()
    game_records = []
    # Get the underlying DQN agent for Q-value extraction
    dqn_inner = None
    if hasattr(agent, 'q_net'):
        dqn_inner = agent
    elif hasattr(agent, 'rl_agent') and hasattr(agent.rl_agent, 'q_net'):
        dqn_inner = agent.rl_agent

    for game_idx in range(games):
        env.reset()
        agent_player = 1 if game_idx % 2 == 0 else 2
        record = {
            'moves': [],
            'result': None,
            'length': 0,
            'agent_player': agent_player,
            'boards_at_agent_moves': [],
            'q_values_at_agent_moves': [],
            'agent_move_columns': [],
        }

        # If opponent goes first
        if agent_player == 2:
            col = opponent.select_action(env)
            env.step(col)
            record['moves'].append(('opponent', col))

        while not env.done:
            # Agent's turn
            board_snap = env.board.copy()
            record['boards_at_agent_moves'].append(board_snap)

            # Get Q-values for visualization
            if dqn_inner is not None:
                state = env.get_state()
                with torch.no_grad():
                    state_t = torch.from_numpy(state).float().permute(2, 0, 1).unsqueeze(0).to(dqn_inner.device)
                    q_vals = dqn_inner.q_net(state_t).squeeze(0).cpu().numpy()
                record['q_values_at_agent_moves'].append(q_vals)

            col = agent.select_action(env, greedy=True)
            record['agent_move_columns'].append(col)
            record['moves'].append(('agent', col))
            _, reward, done, _ = env.step(col)

            if done:
                if reward > 0:
                    record['result'] = 'win'
                elif reward == 0:
                    record['result'] = 'draw'
                else:
                    record['result'] = 'loss'
                break

            # Opponent's turn
            col = opponent.select_action(env)
            record['moves'].append(('opponent', col))
            _, reward, done, _ = env.step(col)

            if done:
                if reward > 0:
                    # Opponent won (reward is from opponent's perspective after step)
                    record['result'] = 'loss'
                elif reward == 0:
                    record['result'] = 'draw'
                else:
                    record['result'] = 'win'
                break

        record['length'] = len(record['moves'])
        game_records.append(record)

    return game_records


def plot_board_heatmap(records, ax_win, ax_loss, ax_all):
    """Plot heatmaps of where the agent places pieces."""
    heatmap_win = np.zeros((ROWS, COLS), dtype=np.float64)
    heatmap_loss = np.zeros((ROWS, COLS), dtype=np.float64)
    heatmap_all = np.zeros((ROWS, COLS), dtype=np.float64)
    n_win, n_loss = 0, 0

    for rec in records:
        # Reconstruct board for agent's pieces
        env = Connect4Env()
        env.reset()
        agent_player = rec['agent_player']

        for who, col in rec['moves']:
            row = env._get_drop_row(col)
            player = agent_player if who == 'agent' else (3 - agent_player)
            env.board[row, col] = player
            if who == 'agent':
                heatmap_all[row, col] += 1
                if rec['result'] == 'win':
                    heatmap_win[row, col] += 1
                elif rec['result'] == 'loss':
                    heatmap_loss[row, col] += 1

        if rec['result'] == 'win':
            n_win += 1
        elif rec['result'] == 'loss':
            n_loss += 1

    for ax, hmap, title, count in [
        (ax_win, heatmap_win, 'Wins', n_win),
        (ax_loss, heatmap_loss, 'Losses', n_loss),
        (ax_all, heatmap_all, 'All Games', len(records)),
    ]:
        # Normalize
        if hmap.sum() > 0:
            hmap_norm = hmap / hmap.sum()
        else:
            hmap_norm = hmap

        im = ax.imshow(hmap_norm, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        ax.set_title(f'{title} (n={count})', fontsize=11, fontweight='bold')
        ax.set_xticks(range(COLS))
        ax.set_yticks(range(ROWS))
        ax.set_xticklabels([str(c) for c in range(COLS)])
        ax.set_yticklabels([str(r) for r in range(ROWS)])
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')

        # Annotate cells with counts
        for r in range(ROWS):
            for c in range(COLS):
                if hmap[r, c] > 0:
                    ax.text(c, r, f'{int(hmap[r, c])}', ha='center', va='center',
                            fontsize=8, color='black' if hmap_norm[r, c] < 0.08 else 'white')

    return im


def plot_win_timing(records, ax):
    """Plot histogram of game lengths for wins vs losses."""
    win_lengths = [r['length'] for r in records if r['result'] == 'win']
    loss_lengths = [r['length'] for r in records if r['result'] == 'loss']
    draw_lengths = [r['length'] for r in records if r['result'] == 'draw']

    max_len = max([r['length'] for r in records]) if records else 42
    bins = range(1, max_len + 2)

    ax.hist(win_lengths, bins=bins, alpha=0.7, color='#2ecc71', label=f'Wins ({len(win_lengths)})', edgecolor='white')
    ax.hist(loss_lengths, bins=bins, alpha=0.7, color='#e74c3c', label=f'Losses ({len(loss_lengths)})', edgecolor='white')
    if draw_lengths:
        ax.hist(draw_lengths, bins=bins, alpha=0.7, color='#95a5a6', label=f'Draws ({len(draw_lengths)})', edgecolor='white')

    ax.set_xlabel('Game Length (total moves)')
    ax.set_ylabel('Count')
    ax.set_title('Win/Loss Timing', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)


def plot_opening_moves(records, ax, depth=4):
    """Plot agent's opening column preferences."""
    # Collect first N agent moves
    first_moves = [[] for _ in range(depth)]

    for rec in records:
        agent_cols = rec['agent_move_columns']
        for i in range(min(depth, len(agent_cols))):
            first_moves[i].append(agent_cols[i])

    x = np.arange(COLS)
    width = 0.8 / depth
    colors = ['#3498db', '#e67e22', '#9b59b6', '#1abc9c']

    for i in range(depth):
        if not first_moves[i]:
            continue
        counts = np.zeros(COLS)
        for col in first_moves[i]:
            counts[col] += 1
        counts = counts / counts.sum()  # normalize to percentage
        offset = (i - depth / 2 + 0.5) * width
        ax.bar(x + offset, counts, width, label=f'Move {i+1}', color=colors[i % len(colors)], alpha=0.85)

    ax.set_xlabel('Column')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Opening Preferences (first {depth} agent moves)', fontsize=11, fontweight='bold')
    ax.set_xticks(range(COLS))
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)


def plot_q_values_game(records, ax):
    """Plot Q-values across moves for a sample winning game."""
    # Find a winning game with Q-values
    win_records = [r for r in records if r['result'] == 'win' and r['q_values_at_agent_moves']]
    if not win_records:
        ax.text(0.5, 0.5, 'No winning games\nwith Q-values', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Q-Values (sample win)', fontsize=11, fontweight='bold')
        return

    # Pick the median-length winning game
    win_records.sort(key=lambda r: r['length'])
    rec = win_records[len(win_records) // 2]

    q_vals = np.array(rec['q_values_at_agent_moves'])
    chosen_cols = rec['agent_move_columns']
    n_moves = len(chosen_cols)
    move_indices = np.arange(n_moves)

    # Plot Q-value of chosen action vs max of other actions
    chosen_q = [q_vals[i][chosen_cols[i]] for i in range(n_moves)]
    max_other_q = []
    for i in range(n_moves):
        other_q = [q_vals[i][c] for c in range(COLS) if c != chosen_cols[i] and q_vals[i][c] > -1e6]
        max_other_q.append(max(other_q) if other_q else chosen_q[i])

    ax.plot(move_indices, chosen_q, 'o-', color='#2ecc71', label='Chosen action Q', markersize=5)
    ax.plot(move_indices, max_other_q, 's--', color='#95a5a6', label='Best alternative Q', markersize=4, alpha=0.7)
    ax.fill_between(move_indices, chosen_q, max_other_q, alpha=0.15, color='#2ecc71')

    ax.set_xlabel('Agent Move #')
    ax.set_ylabel('Q-Value')
    ax.set_title('Q-Values Over a Sample Win', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


def plot_column_preference_by_result(records, ax):
    """Stacked bar: column preference split by win/loss."""
    win_cols = np.zeros(COLS)
    loss_cols = np.zeros(COLS)
    draw_cols = np.zeros(COLS)

    for rec in records:
        for col in rec['agent_move_columns']:
            if rec['result'] == 'win':
                win_cols[col] += 1
            elif rec['result'] == 'loss':
                loss_cols[col] += 1
            else:
                draw_cols[col] += 1

    x = np.arange(COLS)
    ax.bar(x, win_cols, color='#2ecc71', label='Wins', edgecolor='white')
    ax.bar(x, draw_cols, bottom=win_cols, color='#95a5a6', label='Draws', edgecolor='white')
    ax.bar(x, loss_cols, bottom=win_cols + draw_cols, color='#e74c3c', label='Losses', edgecolor='white')

    ax.set_xlabel('Column')
    ax.set_ylabel('Total Moves')
    ax.set_title('Column Usage by Result', fontsize=11, fontweight='bold')
    ax.set_xticks(range(COLS))
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)


def visualize_opponent(agent, agent_name, opp_name, games, depth, save_path=None):
    """Generate full visualization for one opponent."""
    opponent = make_opponent(opp_name, depth=depth, agent=agent)
    print(f"  Collecting {games} games vs {opp_name}...", end=' ', flush=True)
    records = collect_game_data(agent, opponent, games)

    wins = sum(1 for r in records if r['result'] == 'win')
    losses = sum(1 for r in records if r['result'] == 'loss')
    draws = sum(1 for r in records if r['result'] == 'draw')
    print(f"W={wins} D={draws} L={losses}")

    # Create figure
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f'{agent_name.upper()} vs {opp_name.upper()}  —  W:{wins} D:{draws} L:{losses} ({games} games)',
                 fontsize=14, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3, top=0.93, bottom=0.05, left=0.06, right=0.96)

    # Row 1: Board heatmaps (wins, losses, all)
    ax_hw = fig.add_subplot(gs[0, 0])
    ax_hl = fig.add_subplot(gs[0, 1])
    ax_ha = fig.add_subplot(gs[0, 2])
    plot_board_heatmap(records, ax_hw, ax_hl, ax_ha)

    # Row 2: Win timing, Opening moves, Column preference
    ax_timing = fig.add_subplot(gs[1, 0])
    plot_win_timing(records, ax_timing)

    ax_opening = fig.add_subplot(gs[1, 1:])
    plot_opening_moves(records, ax_opening)

    # Row 3: Q-values sample game, Column by result
    has_dqn = hasattr(agent, 'q_net') or (hasattr(agent, 'rl_agent') and hasattr(agent.rl_agent, 'q_net'))
    ax_qvals = fig.add_subplot(gs[2, 0:2])
    if has_dqn:
        plot_q_values_game(records, ax_qvals)
    else:
        ax_qvals.text(0.5, 0.5, 'Q-value plot\n(DQN only)', ha='center', va='center',
                      transform=ax_qvals.transAxes, fontsize=12)
        ax_qvals.set_title('Q-Values', fontsize=11, fontweight='bold')

    ax_colpref = fig.add_subplot(gs[2, 2])
    plot_column_preference_by_result(records, ax_colpref)

    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize agent play style")
    parser.add_argument("--agent", default="dqn", choices=["dqn", "ppo", "dqn-hybrid", "ppo-hybrid"])
    parser.add_argument("--opponent", default="all",
                        choices=["all", "random", "heuristic", "minimax", "self"])
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--save", type=str, default=None,
                        help="Save to file instead of showing (use 'auto' for auto-naming)")
    args = parser.parse_args()

    agent = load_agent(args.agent)

    if args.opponent == "all":
        opponents = ["random", "heuristic", "minimax", "self"]
    else:
        opponents = [args.opponent]

    print(f"\n{'='*60}")
    print(f"Visualizing {args.agent.upper()} ({args.games} games per opponent)")
    print(f"{'='*60}")

    figs = []
    for opp_name in opponents:
        fig = visualize_opponent(agent, args.agent, opp_name, args.games, args.depth)
        figs.append((opp_name, fig))

    if args.save:
        os.makedirs("results", exist_ok=True)
        for opp_name, fig in figs:
            if args.save == "auto":
                path = f"results/{args.agent}_vs_{opp_name}.png"
            else:
                base, ext = os.path.splitext(args.save)
                path = f"{base}_{opp_name}{ext}" if len(figs) > 1 else args.save
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {path}")
    else:
        plt.show()

    print()


if __name__ == "__main__":
    main()
