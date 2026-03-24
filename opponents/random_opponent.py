import random


class RandomOpponent:
    """Selects a uniform random legal move."""

    def select_action(self, env):
        legal = env.get_legal_actions()
        return random.choice(legal)
