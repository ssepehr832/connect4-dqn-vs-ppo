"""Self-play opponent: a frozen snapshot of the learning agent."""

import copy
import numpy as np
import torch


class SelfPlayOpponent:
    """Wraps a frozen copy of an agent as an opponent.

    The snapshot plays greedily (no exploration).
    Call update_snapshot() periodically to refresh.
    """

    def __init__(self, agent):
        self.device = agent.device
        self._q_net = copy.deepcopy(agent.q_net)
        self._q_net.eval()

    def select_action(self, env):
        state = env.get_state()
        legal = env.get_legal_actions()

        # Input state is (6, 7, 2) [H, W, C]
        # QNetwork expects (N, 2, 6, 7) [N, C, H, W]
        state_t = (
            torch.from_numpy(state)
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            q_values = self._q_net(state_t).squeeze(0).cpu().numpy()

        # Greedy choice among legal actions
        best_q = -float("inf")
        best_action = legal[0]
        for a in legal:
            if q_values[a] > best_q:
                best_q = q_values[a]
                best_action = a
        return best_action

    def select_actions_batch(self, states, legal_actions_batch):
        """Batch forward pass through the frozen snapshot.

        Args:
            states: (N, 6, 7, 2) float32 states from opponent's perspective
            legal_actions_batch: list of N legal action lists

        Returns:
            (N,) int array of actions
        """
        n = len(states)
        # Permute (N, 6, 7, 2) -> (N, 2, 6, 7)
        states_t = torch.from_numpy(states).permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            q_values = self._q_net(states_t).cpu().numpy()

        actions = np.zeros(n, dtype=np.int64)
        for i in range(n):
            legal = legal_actions_batch[i]
            best_q = -float("inf")
            best_a = legal[0]
            for a in legal:
                if q_values[i, a] > best_q:
                    best_q = q_values[i, a]
                    best_a = a
            actions[i] = best_a
        return actions

    def update_snapshot(self, agent):
        """Replace the frozen snapshot weights with a fresh copy."""
        self._q_net.load_state_dict(agent.q_net.state_dict())
        self._q_net.eval()
