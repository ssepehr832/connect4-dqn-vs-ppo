"""Evaluate trained agents against opponents and emit structured summaries."""

from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.hybrid import HybridAgent
from envs.vec_connect4 import VecConnect4Env
from opponents import HeuristicOpponent, MinimaxOpponent, RandomOpponent
from training.artifacts import append_csv_rows, now_iso


def default_model_path(agent_type):
    """Resolve the default model path for an agent family."""
    if agent_type.startswith("dqn"):
        return "models/dqn/latest.pt"
    raise ValueError(f"Unknown agent type: {agent_type}")


def load_agent(agent_type, model_path=None, hybrid_depth=4):
    """Load a trained agent from a checkpoint path."""
    path = model_path or default_model_path(agent_type)
    if not os.path.exists(path):
        print(f"Error: No saved model at {path}. Train first.")
        sys.exit(1)

    if agent_type == "dqn":
        from agents.dqn.agent import DQNAgent

        agent = DQNAgent()
        agent.load(path)
        agent.epsilon_start = 0.0
        agent.epsilon_end = 0.0
        return agent

    if agent_type == "dqn-hybrid":
        from agents.dqn.agent import DQNAgent

        rl_agent = DQNAgent()
        rl_agent.load(path)
        rl_agent.epsilon_start = 0.0
        rl_agent.epsilon_end = 0.0
        return HybridAgent(rl_agent, minimax_depth=hybrid_depth)

    if agent_type == "dqn-neural-minimax":
        from agents.dqn.agent import DQNAgent
        from agents.neural_minimax import NeuralMinimaxAgent

        rl_agent = DQNAgent()
        rl_agent.load(path)
        rl_agent.epsilon_start = 0.0
        rl_agent.epsilon_end = 0.0
        return NeuralMinimaxAgent(rl_agent, depth=hybrid_depth)

    print(f"Unknown agent type: {agent_type}")
    sys.exit(1)


class EvalSelfPlayOpponent:
    """Self-play opponent for evaluation: plays near-greedy with small epsilon."""

    def __init__(self, agent, epsilon=0.05):
        if isinstance(agent, HybridAgent):
            rl_copy = copy.deepcopy(agent.rl_agent)
            self._agent = HybridAgent(rl_copy, minimax_depth=agent.minimax.depth)
        else:
            self._agent = copy.deepcopy(agent)
        self._epsilon = epsilon

    def select_action(self, env):
        import random as _rand

        if _rand.random() < self._epsilon:
            return _rand.choice(env.get_legal_actions())
        return self._agent.select_action(env, greedy=True)


def make_opponent(name, depth=6, agent=None):
    """Construct an opponent by name."""
    if name == "random":
        return RandomOpponent()
    if name == "heuristic":
        return HeuristicOpponent()
    if name == "minimax":
        return MinimaxOpponent(depth=depth)
    if name == "self":
        if agent is None:
            raise ValueError("Self-play requires an agent to copy")
        return EvalSelfPlayOpponent(agent, epsilon=0.05)
    if name == "hybrid":
        return _AgentAsOpponent(load_agent("dqn-hybrid"))
    if name == "neural-minimax":
        return _AgentAsOpponent(load_agent("dqn-neural-minimax"))
    raise ValueError(f"Unknown opponent: {name}")


class _AgentAsOpponent:
    """Adapter so an agent (DQNHybrid / NeuralMinimax) can be used as an opponent."""

    def __init__(self, agent):
        self._agent = agent

    def select_action(self, env):
        return self._agent.select_action(env, greedy=True)


def evaluate(agent, opponent, games=100, n_envs=None):
    """Play games with alternating first-player using parallel envs.

    Returns (wins, draws, losses, p1_stats, p2_stats) where p*_stats is
    (wins, draws, losses) when the agent played as that player.
    """
    if n_envs is None:
        n_envs = min(games, 128)

    vec_env = VecConnect4Env(n_envs, opponent)
    wins, draws, losses = 0, 0, 0
    p1_w = p1_d = p1_l = 0
    p2_w = p2_d = p2_l = 0
    ep_count = 0

    states = vec_env.reset_all()

    while ep_count < games:
        legal_batch = vec_env.get_legal_actions_batch()
        actions = np.empty(n_envs, dtype=np.int64)
        for i in range(n_envs):
            proxy = _EvalProxy(
                vec_env.boards[i], vec_env.agent_player[i], states[i], legal_batch[i]
            )
            actions[i] = agent.select_action(proxy, greedy=True)

        vec_env.step(actions)
        records = vec_env.drain_game_records()

        for record in records:
            ep_count += 1
            outcome = record["outcome"]
            side = record["agent_player"]

            if outcome == "draw":
                draws += 1
                if side == 1: p1_d += 1
                else: p2_d += 1
            elif outcome == "win":
                wins += 1
                if side == 1: p1_w += 1
                else: p2_w += 1
            else:
                losses += 1
                if side == 1: p1_l += 1
                else: p2_l += 1

            if ep_count >= games:
                break

        states = vec_env.get_states()

    return wins, draws, losses, (p1_w, p1_d, p1_l), (p2_w, p2_d, p2_l)


class _EvalProxy:
    """Minimal env proxy for agent.select_action during evaluation."""

    def __init__(self, board, current_player, state, legal_actions):
        self.board = board
        self.current_player = current_player
        self._state = state
        self._legal = legal_actions

    def get_legal_actions(self):
        return self._legal

    def get_state(self):
        return self._state


def build_evaluation_row(
    agent_name,
    opponent_name,
    wins,
    draws,
    losses,
    games,
    *,
    p1_stats=None,
    p2_stats=None,
    model_path=None,
    checkpoint_label=None,
    eval_tier="manual",
    minimax_depth=None,
    hybrid_depth=4,
    extra=None,
):
    """Build a structured summary row for one matchup."""
    total = wins + draws + losses
    total = total or games
    row = {
        "timestamp": now_iso(),
        "agent": agent_name,
        "opponent": opponent_name,
        "games": total,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": round(wins / total, 6),
        "draw_rate": round(draws / total, 6),
        "loss_rate": round(losses / total, 6),
        "model_path": os.path.abspath(model_path) if model_path else "",
        "checkpoint_label": checkpoint_label or "",
        "eval_tier": eval_tier,
        "minimax_depth": minimax_depth if opponent_name == "minimax" else "",
        "official_hybrid_depth": hybrid_depth,
    }
    if p1_stats:
        p1w, p1d, p1l = p1_stats
        p1_total = p1w + p1d + p1l
        row["p1_win_rate"] = round(p1w / p1_total, 6) if p1_total else 0.0
    if p2_stats:
        p2w, p2d, p2l = p2_stats
        p2_total = p2w + p2d + p2l
        row["p2_win_rate"] = round(p2w / p2_total, 6) if p2_total else 0.0

    if extra:
        row.update(extra)
    return row


def run_evaluation_suite(
    agent_name,
    opponents,
    *,
    games=100,
    depth=4,
    model_path=None,
    hybrid_depth=4,
    save_csv=None,
    checkpoint_label=None,
    eval_tier="manual",
    extra=None,
):
    """Evaluate one agent against multiple opponents and optionally save rows."""
    agent = load_agent(agent_name, model_path=model_path, hybrid_depth=hybrid_depth)
    rows = []
    for opp_name in opponents:
        opp = make_opponent(opp_name, depth=depth, agent=agent)
        wins, draws, losses, p1, p2 = evaluate(agent, opp, games)
        rows.append(build_evaluation_row(
            agent_name,
            opp_name,
            wins,
            draws,
            losses,
            games,
            p1_stats=p1,
            p2_stats=p2,
            model_path=model_path,
            checkpoint_label=checkpoint_label,
            eval_tier=eval_tier,
            minimax_depth=depth,
            hybrid_depth=hybrid_depth,
            extra=extra,
        ))

    if save_csv:
        append_csv_rows(save_csv, rows)

    return rows


def print_results(row):
    """Pretty-print one evaluation row."""
    total = row["games"]
    print(
        f"  vs {row['opponent']:>10s}:  "
        f"W {row['wins']:>4d} ({100*row['win_rate']:5.1f}%)  "
        f"D {row['draws']:>4d} ({100*row['draw_rate']:5.1f}%)  "
        f"L {row['losses']:>4d} ({100*row['loss_rate']:5.1f}%)  "
        f"[{total} games, alternating first-player]"
    )
    if "p1_win_rate" in row:
        print(f"                as P1: {100*row['p1_win_rate']:.0f}% win")
    if "p2_win_rate" in row:
        print(f"                as P2: {100*row['p2_win_rate']:.0f}% win")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained agent")
    parser.add_argument("--agent", required=True,
                        choices=["dqn", "dqn-hybrid", "dqn-neural-minimax"])
    parser.add_argument("--opponent", default="all",
                        choices=["all", "random", "heuristic", "minimax", "hybrid", "neural-minimax", "self"])
    parser.add_argument("--games", type=int, default=100,
                        help="Games per opponent (default: 100)")
    parser.add_argument("--depth", type=int, default=4,
                        help="Minimax search depth (default: 4)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Checkpoint path to evaluate instead of the default latest model")
    parser.add_argument("--hybrid-depth", type=int, default=4,
                        help="Hybrid minimax depth when evaluating *-hybrid agents")
    parser.add_argument("--save-csv", type=str, default=None,
                        help="Append structured evaluation rows to this CSV path")
    parser.add_argument("--checkpoint-label", type=str, default=None,
                        help="Optional label stored alongside CSV rows")
    parser.add_argument("--eval-tier", type=str, default="manual",
                        help="Label for the evaluation tier (e.g. quick, full, baseline)")
    args = parser.parse_args()

    if args.opponent == "all":
        opponents = ["random", "heuristic", "minimax"]
    else:
        opponents = [args.opponent]

    print(f"\n{'='*60}")
    print(f"Evaluating {args.agent.upper()} ({args.games} games per opponent)")
    print(f"{'='*60}")

    rows = run_evaluation_suite(
        args.agent,
        opponents,
        games=args.games,
        depth=args.depth,
        model_path=args.model_path,
        hybrid_depth=args.hybrid_depth,
        save_csv=args.save_csv,
        checkpoint_label=args.checkpoint_label,
        eval_tier=args.eval_tier,
    )
    for row in rows:
        print_results(row)

    print()


if __name__ == "__main__":
    main()
