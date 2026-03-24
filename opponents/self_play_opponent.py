"""Self-play opponent: a frozen snapshot of the learning agent."""

import copy


class SelfPlayOpponent:
    """Wraps a frozen copy of an agent as an opponent.

    The snapshot plays greedily (no exploration).
    Call update_snapshot() periodically to refresh.
    """

    def __init__(self, agent):
        self._snapshot = copy.deepcopy(agent)

    def select_action(self, env):
        return self._snapshot.select_action(env, greedy=True)

    def update_snapshot(self, agent):
        """Replace the frozen snapshot with a fresh copy of the agent."""
        self._snapshot = copy.deepcopy(agent)
