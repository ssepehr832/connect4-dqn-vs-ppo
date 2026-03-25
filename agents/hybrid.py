"""Hybrid agent: uses minimax for solved positions, RL model otherwise."""

from opponents.minimax_opponent import MinimaxOpponent


class HybridAgent:
    """Wraps a DQN or PPO agent with minimax endgame solving.

    Logic:
        1. Run minimax at given depth
        2. If the position is solved (forced win/loss within depth) → play minimax move
        3. Otherwise → use the RL agent's policy
    """

    def __init__(self, rl_agent, minimax_depth=5):
        self.rl_agent = rl_agent
        self.minimax = MinimaxOpponent(depth=minimax_depth)

    def select_action(self, env, greedy=True):
        is_solved, scores = self.minimax.get_scores(env)

        if is_solved:
            # Play the best minimax move
            legal = env.get_legal_actions()
            best_score = max(scores[c] for c in legal)
            best_actions = [c for c in legal if scores[c] == best_score]
            import random
            return random.choice(best_actions)

        # Not solved — defer to the RL agent
        return self.rl_agent.select_action(env, greedy=greedy)

    def load(self, path):
        self.rl_agent.load(path)
