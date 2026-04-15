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

# Depth for NeuralMinimaxAgent — set by main() from --nm-depth
_NM_DEPTH = 3


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
    elif agent_type == "dqn-neural-minimax":
        from agents.dqn.agent import DQNAgent
        from agents.neural_minimax import NeuralMinimaxAgent
        rl_agent = DQNAgent()
        path = "models/dqn/latest.pt"
        if not os.path.exists(path):
            print(f"Error: No saved model at {path}. Train first.")
            sys.exit(1)
        rl_agent.load(path)
        rl_agent.epsilon_start = 0.0
        rl_agent.epsilon_end = 0.0
        return NeuralMinimaxAgent(rl_agent, depth=_NM_DEPTH)
    else:
        print(f"Unknown agent type: {agent_type}")
        sys.exit(1)


class EvalSelfPlayOpponent:
    """Self-play opponent for evaluation: plays near-greedy with small epsilon."""

    def __init__(self, agent, epsilon=0.05):
        # Load a fresh copy instead of deepcopy to avoid ctypes pickle issues
        from agents.hybrid import HybridAgent
        if isinstance(agent, HybridAgent):
            # For hybrid, just copy the RL agent and wrap fresh
            import copy
            rl_copy = copy.deepcopy(agent.rl_agent)
            self._agent = HybridAgent(rl_copy, minimax_depth=agent.minimax.depth)
        else:
            import copy
            self._agent = copy.deepcopy(agent)
        self._epsilon = epsilon

    def select_action(self, env):
        import random as _rand
        if _rand.random() < self._epsilon:
            return _rand.choice(env.get_legal_actions())
        return self._agent.select_action(env, greedy=True)


def make_opponent(name, depth=6, agent=None):
    if name == "random":
        return RandomOpponent()
    elif name == "heuristic":
        return HeuristicOpponent()
    elif name == "minimax":
        return MinimaxOpponent(depth=depth)
    elif name == "self":
        if agent is None:
            raise ValueError("Self-play requires an agent to copy")
        return EvalSelfPlayOpponent(agent, epsilon=0.05)
    elif name == "hybrid":
        # Load the hybrid agent as an opponent (used to compare against neural-minimax)
        return _AgentAsOpponent(load_agent("dqn-hybrid"))
    elif name == "neural-minimax":
        return _AgentAsOpponent(load_agent("dqn-neural-minimax"))
    else:
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

    from envs.vec_connect4 import VecConnect4Env
    import numpy as np

    vec_env = VecConnect4Env(n_envs, opponent)
    wins, draws, losses = 0, 0, 0
    p1_w = p1_d = p1_l = 0
    p2_w = p2_d = p2_l = 0
    ep_count = 0

    states = vec_env.reset_all()

    while ep_count < games:
        # Capture which side the agent plays BEFORE the step (VecConnect4Env
        # swaps sides on auto-reset, so we need the pre-step value)
        agent_sides = vec_env.agent_player.copy()

        legal_batch = vec_env.get_legal_actions_batch()
        actions = np.empty(n_envs, dtype=np.int64)
        for i in range(n_envs):
            proxy = _EvalProxy(
                vec_env.boards[i], vec_env.agent_player[i], states[i], legal_batch[i]
            )
            actions[i] = agent.select_action(proxy, greedy=True)

        next_states, rewards, dones, _ = vec_env.step(actions)

        for i in range(n_envs):
            if dones[i]:
                ep_count += 1
                side = int(agent_sides[i])
                if rewards[i] > 0:
                    wins += 1
                    if side == 1: p1_w += 1
                    else: p2_w += 1
                elif rewards[i] < 0:
                    losses += 1
                    if side == 1: p1_l += 1
                    else: p2_l += 1
                else:
                    draws += 1
                    if side == 1: p1_d += 1
                    else: p2_d += 1
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


def print_results(agent_name, opp_name, wins, draws, losses, games,
                  p1_stats=None, p2_stats=None):
    total = wins + draws + losses
    print(f"  vs {opp_name:>10s}:  W {wins:>4d} ({100*wins/total:5.1f}%)  "
          f"D {draws:>4d} ({100*draws/total:5.1f}%)  "
          f"L {losses:>4d} ({100*losses/total:5.1f}%)  "
          f"[{total} games, alternating first-player]")
    if p1_stats is not None and p2_stats is not None:
        p1w, p1d, p1l = p1_stats
        p2w, p2d, p2l = p2_stats
        p1_total = p1w + p1d + p1l
        p2_total = p2w + p2d + p2l
        if p1_total:
            print(f"                as P1: W {p1w:>3d} D {p1d:>3d} L {p1l:>3d}  "
                  f"({100*p1w/p1_total:.0f}% win)")
        if p2_total:
            print(f"                as P2: W {p2w:>3d} D {p2d:>3d} L {p2l:>3d}  "
                  f"({100*p2w/p2_total:.0f}% win)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained agent")
    parser.add_argument("--agent", required=True,
                        choices=["dqn", "ppo", "dqn-hybrid", "ppo-hybrid", "dqn-neural-minimax"])
    parser.add_argument("--opponent", default="all",
                        choices=["all", "random", "heuristic", "minimax", "hybrid", "neural-minimax"])
    parser.add_argument("--games", type=int, default=100,
                        help="Games per opponent (default: 100)")
    parser.add_argument("--depth", type=int, default=4,
                        help="Minimax search depth (default: 4)")
    parser.add_argument("--nm-depth", type=int, default=3,
                        help="Neural-minimax search depth (default: 3)")
    args = parser.parse_args()

    global _NM_DEPTH
    _NM_DEPTH = args.nm_depth

    agent = load_agent(args.agent)

    if args.opponent == "all":
        opponents = ["random", "heuristic", "minimax"]
    else:
        opponents = [args.opponent]

    print(f"\n{'='*60}")
    print(f"Evaluating {args.agent.upper()} ({args.games} games per opponent)")
    print(f"{'='*60}")

    for opp_name in opponents:
        opp = make_opponent(opp_name, depth=args.depth, agent=agent)
        w, d, l, p1, p2 = evaluate(agent, opp, args.games)
        print_results(args.agent.upper(), opp_name, w, d, l, args.games, p1, p2)

    print()


if __name__ == "__main__":
    main()
