"""Hybrid agent: uses minimax to filter dangerous moves, RL model for strategy."""

import random
from opponents.minimax_opponent import MinimaxOpponent


class HybridAgent:
    """Wraps a DQN or PPO agent with minimax safety filtering.

    Logic:
        1. Run minimax at given depth to get per-column scores
        2. If agent has a forced win → play it (minimax move)
        3. Otherwise, filter out moves that lead to forced losses
        4. Let the RL agent choose among the remaining safe moves
        5. If ALL moves lose → play the least-bad one (minimax's best)
    """

    def __init__(self, rl_agent, minimax_depth=5):
        self.rl_agent = rl_agent
        self.minimax = MinimaxOpponent(depth=minimax_depth)

    def select_action(self, env, greedy=True):
        is_solved, scores = self.minimax.get_scores(env)
        legal = env.get_legal_actions()

        if is_solved:
            best_score = max(scores[c] for c in legal)

            # Agent has a forced win — play it
            if best_score >= 10000:
                best_actions = [c for c in legal if scores[c] == best_score]
                return random.choice(best_actions)

            # Some moves are forced losses — filter them out
            safe_moves = [c for c in legal if scores[c] > -10000]

            if not safe_moves:
                # All moves lose — pick the least-bad (highest score)
                best_actions = [c for c in legal if scores[c] == best_score]
                return random.choice(best_actions)

            if len(safe_moves) == 1:
                return safe_moves[0]

            # Multiple safe moves — let DQN choose among them
            return self.rl_agent.select_action(
                env, greedy=greedy, allowed_actions=safe_moves
            )

        # Nothing solved — full DQN control
        return self.rl_agent.select_action(env, greedy=greedy)

    def load(self, path):
        self.rl_agent.load(path)
