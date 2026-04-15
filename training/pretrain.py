"""Supervised pretraining on the Tromp Connect 4 dataset.

Trains the conv backbone to classify positions as win/loss/draw,
then transfers those features to the DQN network for RL fine-tuning.

Dataset: 67,557 board positions at 8-ply, labeled {win, loss, draw}
         from the perspective of the player to move (player x).

Usage:
    python -m training.pretrain --agent dqn --epochs 50
"""

import argparse
import io
import os
import subprocess
import sys
import time
import urllib.request
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.artifacts import append_csv_row, ensure_dir, now_iso, write_json


DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "connect-4.data"
)
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/26/connect%2B4.zip"


def ensure_dataset(path=DATA_PATH):
    """Ensure the Tromp/UCI dataset exists locally."""
    if os.path.exists(path):
        return path

    data_dir = os.path.dirname(path)
    ensure_dir(data_dir)

    print(f"Dataset missing at {path}; downloading from UCI...")
    with urllib.request.urlopen(UCI_ZIP_URL, timeout=60) as resp:
        payload = resp.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        compressed_name = "connect-4.data.Z"
        if compressed_name not in archive.namelist():
            raise RuntimeError(
                f"UCI archive did not contain {compressed_name}; found {archive.namelist()}"
            )
        compressed_path = os.path.join(data_dir, compressed_name)
        archive.extract(compressed_name, data_dir)

    # macOS/Linux gunzip can transparently decode historical .Z files.
    last_error = None
    for cmd in (["gunzip", "-c", compressed_path], ["gzip", "-dc", compressed_path]):
        try:
            with open(path, "wb") as handle:
                subprocess.run(cmd, check=True, stdout=handle, stderr=subprocess.PIPE)
            os.remove(compressed_path)
            print(f"Saved dataset to {path}")
            return path
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            last_error = exc

    raise RuntimeError(
        "Failed to decompress connect-4.data.Z after download. "
        "Please ensure gunzip or gzip is installed."
    ) from last_error


class Connect4Dataset(Dataset):
    """Load the Tromp dataset into (state, label) pairs.

    State: (2, 6, 7) float tensor — channel 0 = current player, channel 1 = opponent
    Label: 0 = loss, 1 = draw, 2 = win  (from current player's perspective)
    """

    def __init__(self, path=DATA_PATH):
        path = ensure_dataset(path)
        states = []
        labels = []
        label_map = {"loss": 0, "draw": 1, "win": 2}

        with open(path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != 43:
                    continue
                cells = parts[:42]
                outcome = parts[42]

                # Build board: attributes are column-major (a1-a6, b1-b6, ..., g1-g6)
                # a1 = bottom-left, a6 = top-left
                # Our board: row 0 = top, row 5 = bottom
                board = np.zeros((6, 7), dtype=np.float32)
                state = np.zeros((2, 6, 7), dtype=np.float32)

                for idx, cell in enumerate(cells):
                    col = idx // 6
                    row_from_bottom = idx % 6
                    row = 5 - row_from_bottom  # flip to our convention

                    if cell == "x":
                        state[0, row, col] = 1.0  # current player (x moves next)
                    elif cell == "o":
                        state[1, row, col] = 1.0  # opponent

                states.append(state)
                labels.append(label_map[outcome])

        self.states = np.array(states)
        self.labels = np.array(labels, dtype=np.int64)
        print(f"Loaded {len(self.labels)} positions: "
              f"win={np.sum(self.labels == 2)}, "
              f"draw={np.sum(self.labels == 1)}, "
              f"loss={np.sum(self.labels == 0)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.states[idx]), self.labels[idx]


class PretrainNet(nn.Module):
    """Conv backbone + classification head for position evaluation.

    Uses the same conv architecture as the DQN network so weights
    can be directly transferred.
    """

    def __init__(self, conv_backbone):
        super().__init__()
        self.conv = conv_backbone
        # Classification head: win / draw / loss
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 7, 512), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 3),
        )

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.classifier(h)


def get_conv_backbone(agent_type):
    """Create a fresh conv backbone matching the agent's architecture."""
    if agent_type == "dqn":
        from agents.dqn.network import QNetwork
        net = QNetwork()
        return net.conv
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def transfer_weights(agent_type, pretrained_conv, save_path=None, save_dir=None):
    """Transfer pretrained conv weights to the agent and save."""
    if save_path is None:
        if save_dir is None:
            save_dir = f"models/{agent_type}"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "latest.pt")

    if agent_type == "dqn":
        from agents.dqn.agent import DQNAgent
        agent = DQNAgent()
        # Load pretrained conv weights into both q_net and target_net
        agent.q_net.conv.load_state_dict(pretrained_conv.state_dict())
        agent.target_net.conv.load_state_dict(pretrained_conv.state_dict())
        agent.save(save_path)
        print(f"Transferred conv weights to DQN, saved to {save_path}")


def train(
    agent_type,
    epochs=50,
    batch_size=256,
    lr=1e-3,
    device=None,
    data_path=DATA_PATH,
    metrics_path=None,
    summary_path=None,
    save_path=None,
):
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"Device: {device}")

    # Load dataset
    dataset = Connect4Dataset(path=data_path)
    n_total = len(dataset)
    n_val = int(0.1 * n_total)
    n_train = n_total - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    # Build model
    conv = get_conv_backbone(agent_type)
    model = PretrainNet(conv).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Pretrain model: {n_params:,} parameters")

    # Class weights to handle imbalance (65% win, 25% loss, 10% draw)
    class_counts = np.bincount(dataset.labels, minlength=3)
    weights = 1.0 / class_counts.astype(np.float32)
    weights = weights / weights.sum() * 3  # normalize
    class_weights = torch.from_numpy(weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_conv_state = None

    print(f"\n{'Epoch':>5s} | {'Train Loss':>10s} | {'Train Acc':>9s} | {'Val Acc':>9s} | {'Time':>5s}")
    print("-" * 50)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        total_loss, correct, total = 0, 0, 0
        for states, labels in train_loader:
            states = states.to(device)
            labels = labels.to(device)

            logits = model(states)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for states, labels in val_loader:
                states = states.to(device)
                labels = labels.to(device)
                logits = model(states)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += len(labels)

        val_acc = val_correct / val_total
        elapsed = time.time() - t0

        print(f"{epoch:>5d} | {train_loss:>10.4f} | {train_acc:>8.1%} | {val_acc:>8.1%} | {elapsed:>4.1f}s")

        if metrics_path:
            append_csv_row(metrics_path, {
                "timestamp": now_iso(),
                "agent": agent_type,
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_acc": round(val_acc, 6),
                "lr": round(float(optimizer.param_groups[0]["lr"]), 10),
                "epoch_seconds": round(elapsed, 3),
                "device": str(device),
                "dataset_path": os.path.abspath(data_path),
            })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_conv_state = {k: v.cpu().clone() for k, v in conv.state_dict().items()}

        scheduler.step()

    print(f"\nBest validation accuracy: {best_val_acc:.1%}")

    # Transfer best conv weights to agent
    conv.load_state_dict(best_conv_state)
    default_save_dir = f"models/{agent_type}"
    os.makedirs(default_save_dir, exist_ok=True)
    if save_path is None:
        save_path = os.path.join(default_save_dir, "latest.pt")
    transfer_weights(agent_type, conv, save_path=save_path)

    if summary_path:
        write_json(summary_path, {
            "timestamp": now_iso(),
            "agent": agent_type,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "device": str(device),
            "dataset_path": os.path.abspath(data_path),
            "save_path": os.path.abspath(save_path),
            "best_val_acc": best_val_acc,
            "n_total": n_total,
            "n_train": n_train,
            "n_val": n_val,
        })

    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Supervised pretraining on Connect 4 dataset")
    parser.add_argument("--agent", default="dqn", choices=["dqn"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-path", type=str, default=DATA_PATH)
    parser.add_argument("--metrics-path", type=str, default=None)
    parser.add_argument("--summary-path", type=str, default=None)
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()

    train(
        args.agent,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        data_path=args.data_path,
        metrics_path=args.metrics_path,
        summary_path=args.summary_path,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
