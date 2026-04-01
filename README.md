# connect4-dqn-vs-ppo

Empirical comparison of **DQN** and **PPO** for learning Connect 4 through self-play against progressively stronger opponents.

## Overview

Connect 4 is a two-player, zero-sum game with sparse terminal rewards, making it a challenging testbed for deep reinforcement learning. This project compares two foundational RL algorithms — a value-based method (DQN) and a policy-gradient method (PPO) — across multiple dimensions: **win rate**, **sample efficiency**, and **training stability**.

To prevent agents from overfitting to a single opponent strategy, we train against increasingly difficult opponents:

| Opponent | Description |
|---|---|
| **Random** | Uniform random legal moves — baseline for initial learning |
| **Heuristic** | Rule-based: wins if possible, blocks opponent wins, prefers center (non-deterministic among equally good moves) |
| **Minimax** | Alpha-beta search (C-accelerated with pthreads, depth 5) — plays optimally in solved positions, randomly among top moves otherwise |
| **Self-play** | Frozen snapshot of the agent itself, updated every 3000 episodes during training |
| **Self-mixed** | Alternates between self-play and minimax in chunks of 1000 episodes — prevents self-play cycling |

## Methods

### DQN (Deep Q-Network)
Off-policy, value-based learning with experience replay and a target network. Uses 3-step returns for faster reward propagation. The agent learns Q-values for each column and selects the action with the highest estimated return.

### PPO (Proximal Policy Optimization)
On-policy, actor-critic method with clipped surrogate objectives and GAE. Directly optimizes a stochastic policy while maintaining training stability through conservative updates.

### Hybrid (RL + Minimax)
Combines learned strategy with exact endgame solving. Minimax evaluates the position:
- **Forced win detected** — play the winning move
- **Some moves lose, others safe** — filter out losing moves, let RL choose among safe ones
- **Position unsolved** — defer entirely to the RL agent's policy

Available as `dqn-hybrid` and `ppo-hybrid`.

### Supervised Pretraining
Conv layers are pretrained on the John Tromp solved Connect 4 dataset (~67k positions at 8 pieces, labeled win/loss/draw). This gives the network strong spatial features before RL training begins. During RL fine-tuning, conv layers are frozen (`--freeze-conv`) so only the FC head adapts — preserving the ground-truth features learned from solved data.

### Network Architecture
Both agents share the same CNN backbone:
- 5 convolutional layers (256 filters each, 3x3, padding=1) on a 6x7x2 input (one channel per player)
- Fully connected head: 1024 → 512 → 256 → output
- DQN outputs 7 Q-values; PPO outputs a policy distribution + value estimate

### Reward Shaping
- **Win:** +1.0 with a speed bonus of up to +0.3 for faster wins
- **Draw:** +0.2
- **Loss:** -1.1 (asymmetric to discourage passive play)

## Performance Optimizations

- **Vectorized environments** — N games run in parallel (configurable via `--n-envs`, default 16 for DQN, 64 for PPO)
- **C minimax engine** — alpha-beta pruning implemented in C, auto-compiled on first use
- **Parallel minimax with pthreads** — all N opponent minimax calls batched into one parallel C call
- **Batched self-play** — opponent forward passes batched on GPU instead of one-by-one
- **Arbiter (early termination)** — during self-play, minimax checks positions mid-game and ends lost games early (`--arbiter`)
- **Game uniqueness tracking** — tracks unique games via move-sequence hashing to monitor training diversity
- **MPS/CUDA support** — auto-detects Apple Silicon GPU or NVIDIA GPU

## Evaluation

- **Win / Draw / Loss rates** against each opponent type
- **6-panel visualization** — board heatmaps (wins/losses/all games), win timing distribution, opening preferences, Q-value curves, column preference by result
- **Stability analysis** — mean +/- std over multiple random seeds
- Agents evaluated with **alternating first-player assignment** for fairness
- Evaluation runs in parallel (128 envs) for fast results

## Project Structure

```
├── envs/
│   ├── connect4.py              # Core game (6x7, Gym-like API)
│   └── vec_connect4.py          # Vectorized N-game parallel environment
├── agents/
│   ├── dqn/                     # DQN (Q-network, 3-step replay buffer, agent)
│   ├── ppo/                     # PPO (actor-critic network, rollout buffer, agent)
│   └── hybrid.py                # Hybrid agent (minimax + RL)
├── opponents/
│   ├── random_opponent.py
│   ├── heuristic_opponent.py
│   ├── minimax_opponent.py      # Python wrapper (ctypes)
│   ├── minimax_engine.c         # C engine (alpha-beta + pthreads)
│   └── self_play_opponent.py
├── training/
│   ├── train_dqn.py             # DQN training loop
│   ├── train_ppo.py             # PPO training loop
│   └── pretrain.py              # Supervised pretraining from solved dataset
├── evaluation/
│   ├── evaluate.py              # Parallel evaluation against opponents
│   └── visualize.py             # 6-panel game analysis visualization
├── data/
│   └── connect-4.data           # John Tromp solved positions dataset
├── models/
│   ├── dqn/latest.pt
│   └── ppo/latest.pt
├── play.py                      # Interactive play (human or agent vs anything)
├── build.sh                     # Manual C compilation script
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.9+
- PyTorch
- NumPy, Matplotlib
- C compiler (gcc) — for the minimax engine

### Installation
```bash
git clone https://github.com/ssepehr832/connect4-dqn-vs-ppo.git
cd connect4-dqn-vs-ppo
pip install -r requirements.txt
```

The minimax C engine compiles automatically on first use. To compile manually:
```bash
bash build.sh
```

### Supervised Pretraining
```bash
# Pretrain conv layers on the solved dataset
python -m training.pretrain
```

### Training

Training auto-resumes from `models/<agent>/latest.pt` if it exists. Train incrementally against stronger opponents:

```bash
# DQN — curriculum training (with frozen pretrained conv)
python -m training.train_dqn --opponent random --episodes 5000 --freeze-conv
python -m training.train_dqn --opponent heuristic --episodes 10000 --freeze-conv --lr 5e-5
python -m training.train_dqn --opponent minimax --episodes 20000 --freeze-conv --lr 1e-5
python -m training.train_dqn --opponent self-mixed --episodes 50000 --freeze-conv --lr 1e-5

# PPO — curriculum training
python -m training.train_ppo --opponent random --episodes 5000
python -m training.train_ppo --opponent heuristic --episodes 10000
python -m training.train_ppo --opponent minimax --episodes 20000
python -m training.train_ppo --opponent self --episodes 50000

# Useful flags
--fresh                  # Start from scratch (ignore latest.pt)
--n-envs 128             # More parallel games (scale with CPU cores)
--freeze-conv            # Freeze pretrained conv layers, train FC head only
--arbiter                # Early termination via minimax during self-play
--eps-start 0.1          # Start epsilon (DQN)
--eps-decay 5000         # Epsilon decay steps (DQN)
--lr 1e-4                # Learning rate
--entropy-coef 0.05      # Exploration bonus (PPO)
```

### Evaluation

```bash
python -m evaluation.evaluate --agent dqn --opponent all
python -m evaluation.evaluate --agent dqn-hybrid --opponent all
python -m evaluation.evaluate --agent ppo --opponent minimax --games 200 --depth 6
```

### Visualization

```bash
python -m evaluation.visualize --agent dqn --opponent minimax --games 200
```

### Play

Play against any agent or opponent interactively, or watch agents play each other:

```bash
python play.py                            # You vs Heuristic (default)
python play.py -p2 dqn                    # You vs DQN
python play.py -p2 dqn-hybrid             # You vs DQN-Hybrid
python play.py -p2 minimax                # You vs Minimax
python play.py -p1 dqn -p2 ppo           # Watch DQN vs PPO
python play.py -p1 dqn-hybrid -p2 ppo-hybrid -n 10 --swap  # 10 games, alternate sides
```

## References

1. Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning.* Nature, 518(7540), 529-533.
2. Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
3. Mnih, V., et al. (2016). *Asynchronous methods for deep reinforcement learning.* ICML.
4. van Hasselt, H., Guez, A., & Silver, D. (2016). *Deep reinforcement learning with double Q-learning.* AAAI.
5. Shah, S., & Gupta, S. (2022). *Reinforcement Learning for ConnectX.* arXiv:2210.08263.
6. Knuth, D. E., & Moore, R. W. (1975). *An Analysis of Alpha-Beta Pruning.* Artificial Intelligence, 6(4), 293-326.
7. Tromp, J. (2008). *John's Connect Four Playground.* https://tromp.github.io/c4/c4.html

## License

MIT
