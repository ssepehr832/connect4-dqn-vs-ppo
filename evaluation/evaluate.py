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


def make_opponent(name, depth=6):
    if name == "random":
        return RandomOpponent()
    elif name == "heuristic":
        return HeuristicOpponent()
    elif name == "minimax":
        return MinimaxOpponent(depth=depth)
    else:
        raise ValueError(f"Unknown opponent: {name}")


def evaluate(agent, opponent, games=100, n_envs=None):
    """Play games with alternating first-player using parallel envs.

    Returns (wins, draws, losses).
    """
    if n_envs is None:
        n_envs = min(games, 128)

    from envs.vec_connect4 import VecConnect4Env
    import numpy as np

    vec_env = VecConnect4Env(n_envs, opponent)
    wins, draws, losses = 0, 0, 0
    ep_count = 0

    states = vec_env.reset_all()

    while ep_count < games:
        # Agent selects actions (batched if possible)
        legal_batch = vec_env.get_legal_actions_batch()
        actions = np.empty(n_envs, dtype=np.int64)
        for i in range(n_envs):
            # Build a proxy env for the agent's select_action
            proxy = _EvalProxy(
                vec_env.boards[i], vec_env.agent_player[i], states[i], legal_batch[i]
            )
            actions[i] = agent.select_action(proxy, greedy=True)

        next_states, rewards, dones, _ = vec_env.step(actions)

        for i in range(n_envs):
            if dones[i]:
                ep_count += 1
                if rewards[i] > 0:
                    wins += 1
                elif rewards[i] < 0:
                    losses += 1
                else:
                    draws += 1
                if ep_count >= games:
                    break

        states = vec_env.get_states()

    return wins, draws, losses


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


def print_results(agent_name, opp_name, wins, draws, losses, games):
    total = wins + draws + losses
    print(f"  vs {opp_name:>10s}:  W {wins:>4d} ({100*wins/total:5.1f}%)  "
          f"D {draws:>4d} ({100*draws/total:5.1f}%)  "
          f"L {losses:>4d} ({100*losses/total:5.1f}%)  "
          f"[{total} games, alternating first-player]")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained agent")
    parser.add_argument("--agent", required=True,
                        choices=["dqn", "ppo", "dqn-hybrid", "ppo-hybrid"])
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
