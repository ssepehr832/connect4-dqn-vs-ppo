import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """CNN-based Q-network for Connect 4 with dual conv paths.

    Input:  (batch, 2, 6, 7)  — two binary feature planes
    Output: (batch, 7)        — Q-value per column

    The frozen path (conv) contains pretrained weights from solved data.
    The trainable path (conv_train) learns additional patterns from RL.
    Both paths feed into the same FC head.
    """

    def __init__(self):
        super().__init__()
        # Pretrained path (frozen during RL)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
        )
        # Trainable path (learns from RL)
        self.conv_train = nn.Sequential(
            nn.Conv2d(2, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
        )
        # FC head takes concatenated features from both paths
        self.fc = nn.Sequential(
            nn.Linear(512 * 6 * 7, 1024), nn.ReLU(),
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
        h_frozen = self.conv(x)
        h_train = self.conv_train(x)
        h = torch.cat([h_frozen, h_train], dim=1).flatten(1)
        return self.fc(h)
