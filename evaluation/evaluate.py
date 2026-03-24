"""Evaluate a trained agent against opponents.

Usage:
    python -m evaluation.evaluate --agent dqn --opponent all
    python -m evaluation.evaluate --agent dqn --opponent random --games 200
    python -m evaluation.evaluate --agent dqn --opponent heuristic
    python -m evaluation.evaluate --agent dqn --opponent minimax --depth 4
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.connect4 import Connect4Env
from opponents import RandomOpponent, HeuristicOpponent, MinimaxOpponent


def load_agent(agent_type):
    """Load a trained agent from models/<type>/latest.pt."""
    if agent_type == "dqn":
        from agents.dqn.agent import DQNAgent
        agent = DQNAgent()
        path = "models/dqn/latest.pt"
        if not os.path.exists(path):
            print(f"Error: No saved model at {path}. Train first.")
            sys.exit(1)
        agent.load(path)
        agent.epsilon_start = 0.0  # greedy for evaluation
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
    else:
        print(f"Unknown agent type: {agent_type}")
        sys.exit(1)


def make_opponent(name, depth=6):
    if name == "random":
        return RandomOpponent()
    elif name == "heuristic":
        return HeuristicOpponent()
    elif name == "minimax":
        return MinimaxOpponent(depth=depth)
    else:
        raise ValueError(f"Unknown opponent: {name}")


def evaluate(agent, opponent, games=100):
    """Play games with alternating first-player. Returns (wins, draws, losses)."""
    env = Connect4Env()
    wins, draws, losses = 0, 0, 0

    for i in range(games):
        env.reset()
        # Alternate who goes first
        agent_player = 1 if i % 2 == 0 else 2

        while not env.done:
            if env.current_player == agent_player:
                action = agent.select_action(env, greedy=True)
            else:
                action = opponent.select_action(env)
            env.step(action)

        if env.winner == agent_player:
            wins += 1
        elif env.winner == 0:
            draws += 1
        else:
            losses += 1

    return wins, draws, losses


def print_results(agent_name, opp_name, wins, draws, losses, games):
    total = wins + draws + losses
    print(f"  vs {opp_name:>10s}:  W {wins:>4d} ({100*wins/total:5.1f}%)  "
          f"D {draws:>4d} ({100*draws/total:5.1f}%)  "
          f"L {losses:>4d} ({100*losses/total:5.1f}%)  "
          f"[{total} games, alternating first-player]")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained agent")
    parser.add_argument("--agent", required=True, choices=["dqn", "ppo"])
    parser.add_argument("--opponent", default="all",
                        choices=["all", "random", "heuristic", "minimax"])
    parser.add_argument("--games", type=int, default=100,
                        help="Games per opponent (default: 100)")
    parser.add_argument("--depth", type=int, default=4,
                        help="Minimax search depth (default: 4)")
    args = parser.parse_args()

    agent = load_agent(args.agent)

    if args.opponent == "all":
        opponents = ["random", "heuristic", "minimax"]
    else:
        opponents = [args.opponent]

    print(f"\n{'='*60}")
    print(f"Evaluating {args.agent.upper()} ({args.games} games per opponent)")
    print(f"{'='*60}")

    for opp_name in opponents:
        opp = make_opponent(opp_name, depth=args.depth)
        w, d, l = evaluate(agent, opp, args.games)
        print_results(args.agent.upper(), opp_name, w, d, l, args.games)

    print()


if __name__ == "__main__":
    main()
