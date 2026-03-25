"""Self-play opponent: a frozen snapshot of the learning agent."""

import copy
import numpy as np


class SelfPlayOpponent:
    """Wraps a frozen copy of an agent as an opponent.

    The snapshot plays greedily (no exploration).
    Call update_snapshot() periodically to refresh.
    """

    def __init__(self, agent):
        self._snapshot = copy.deepcopy(agent)

    def select_action(self, env):
        return self._snapshot.select_action(env, greedy=True)

    def select_actions_batch(self, states, legal_actions_batch):
        """Batch forward pass through the frozen snapshot.

        Args:
            states: (N, 6, 7, 2) float32 states from opponent's perspective
            legal_actions_batch: list of N legal action lists

        Returns:
            (N,) int array of actions
        """
        return self._snapshot.select_actions_batch(
            states, legal_actions_batch, greedy=True
        )

    def update_snapshot(self, agent):
        """Replace the frozen snapshot with a fresh copy of the agent."""
        self._snapshot = copy.deepcopy(agent)
