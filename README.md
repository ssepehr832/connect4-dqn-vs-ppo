# connect4-dqn-vs-ppo

Empirical comparison of **DQN** and **PPO** for learning Connect 4 through self-play against progressively stronger opponents.

## Overview

Connect 4 is a two-player, zero-sum game with sparse terminal rewards, making it a challenging testbed for deep reinforcement learning. This project compares two foundational RL algorithms — a value-based method (DQN) and a policy-gradient method (PPO) — across multiple dimensions: **win rate**, **sample efficiency**, and **training stability**.

To prevent agents from overfitting to a single opponent strategy, we train against increasingly difficult opponents:

| Opponent | Description |
|---|---|
| **Random** | Uniform random legal moves — baseline for initial learning |
| **Heuristic** | Rule-based: wins if possible, blocks opponent wins, prefers center columns |
| **Minimax** | Alpha-beta search (C-accelerated, depth 4) — tests generalization to stronger play |

## Methods

### DQN (Deep Q-Network)
Off-policy, value-based learning with experience replay and a target network. The agent learns Q-values for each column and selects the action with the highest estimated return.

### PPO (Proximal Policy Optimization)
On-policy, actor-critic method with clipped surrogate objectives. Directly optimizes a stochastic policy while maintaining training stability through conservative updates.

## Evaluation

- **Win / Draw / Loss rates** against each opponent type
- **Learning curves** — performance as a function of training steps
- **Stability analysis** — mean ± std over multiple random seeds (3–5 runs)
- Agents evaluated with **alternating first-player assignment** for fairness

## Project Structure

```
├── envs/                # Connect 4 environment (6x7, Gym-like API)
├── agents/
│   ├── dqn/             # DQN agent (Q-network, replay buffer, agent)
│   └── ppo/             # PPO agent (actor-critic network, agent)
├── opponents/           # Random, heuristic, and minimax (C + ctypes)
├── training/            # Training scripts (auto-resume from latest.pt)
├── evaluation/          # Evaluation scripts and metrics
├── models/
│   ├── dqn/             # DQN checkpoints (latest.pt)
│   └── ppo/             # PPO checkpoints (latest.pt)
├── results/             # Logs, plots
├── notebooks/           # Analysis and visualization notebooks
├── play.py              # Interactive play against any agent/opponent
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

### Training

Training auto-resumes from `models/<agent>/latest.pt` if it exists. Train incrementally against stronger opponents:

```bash
# DQN
python -m training.train_dqn --opponent random --episodes 5000
python -m training.train_dqn --opponent heuristic --episodes 10000
python -m training.train_dqn --opponent minimax --episodes 20000

# PPO
python -m training.train_ppo --opponent random --episodes 5000
python -m training.train_ppo --opponent heuristic --episodes 10000
python -m training.train_ppo --opponent minimax --episodes 20000

# Start fresh (ignore existing latest.pt)
python -m training.train_dqn --fresh --opponent random --episodes 5000
```

### Play

Play against any agent or opponent interactively, or watch agents play each other:

```bash
python play.py                            # You vs Heuristic (default)
python play.py -p2 dqn                    # You vs DQN
python play.py -p2 minimax                # You vs Minimax
python play.py -p1 dqn -p2 ppo           # Watch DQN vs PPO
python play.py -p1 dqn -p2 heuristic -n 10 --swap   # 10 games, alternate sides
```

## References

1. Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning.* Nature, 518(7540), 529–533.
2. Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
3. Mnih, V., et al. (2016). *Asynchronous methods for deep reinforcement learning.* ICML.
4. van Hasselt, H., Guez, A., & Silver, D. (2016). *Deep reinforcement learning with double Q-learning.* AAAI.
5. Shah, S., & Gupta, S. (2022). *Reinforcement Learning for ConnectX.* arXiv:2210.08263.
6. Knuth, D. E., & Moore, R. W. (1975). *An Analysis of Alpha-Beta Pruning.* Artificial Intelligence, 6(4), 293–326.

## License

MIT
