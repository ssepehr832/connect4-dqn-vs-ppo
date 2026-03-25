import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """CNN-based Q-network for Connect 4.

    Input:  (batch, 2, 6, 7)  — two binary feature planes
    Output: (batch, 7)        — Q-value per column
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 7, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 7),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 2, 6, 7) float tensor
        Returns:
            (batch, 7) Q-values for each column
        """
        h = self.conv(x).flatten(1)
        return self.fc(h)
