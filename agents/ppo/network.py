import torch
import torch.nn as nn


class ActorCriticNetwork(nn.Module):
    """CNN-based actor-critic for Connect 4.

    Input:  (batch, 2, 6, 7) — two binary feature planes
    Output: policy logits (batch, 7), value (batch, 1)

    Same backbone as DQN's QNetwork for fair comparison.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()

        # Policy head
        self.policy = nn.Sequential(
            nn.Linear(128 * 6 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 7),
        )

        # Value head
        self.value = nn.Sequential(
            nn.Linear(128 * 6 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 2, 6, 7) float tensor
        Returns:
            logits: (batch, 7) raw policy logits (pre-softmax)
            value:  (batch,) state value estimates
        """
        h = self.flatten(self.conv(x))
        logits = self.policy(h)
        value = self.value(h).squeeze(-1)
        return logits, value
