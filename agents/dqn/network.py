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
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 7),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 2, 6, 7) float tensor
        Returns:
            (batch, 7) Q-values for each column
        """
        h = self.conv(x)
        return self.fc(h)
