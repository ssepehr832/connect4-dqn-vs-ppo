"""PPO training script.

Auto-resumes from models/ppo/latest.pt if it exists.

Usage:
    python -m training.train_ppo --opponent random --episodes 5000
    python -m training.train_ppo --opponent heuristic --episodes 10000
    python -m training.train_ppo --opponent self --episodes 20000
    python -m training.train_ppo --fresh --opponent random   # start fresh
"""

import argparse
import os
import shutil
import sys
import time
import random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.vec_connect4 import VecConnect4Env
from agents.ppo.agent import PPOAgent
from opponents import RandomOpponent, HeuristicOpponent, MinimaxOpponent
from opponents.self_play_opponent import SelfPlayOpponent

N_ENVS = 64


def fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def progress_line(ep, total, t_start, extra=""):
    elapsed = time.time() - t_start
    eps_per_sec = ep / elapsed if elapsed > 0 else 0
    remaining = (total - ep) / eps_per_sec if eps_per_sec > 0 else 0

    pct = ep / total if total > 0 else 0
    bar_len = 20
    filled = int(bar_len * pct)
    bar = "█" * filled + "░" * (bar_len - filled)

    line = (
        f"  {bar} {ep}/{total} ({pct:.0%}) | "
        f"{eps_per_sec:.1f} ep/s | "
        f"{fmt_time(elapsed)}<{fmt_time(remaining)}"
    )
    if extra:
        line += f" | {extra}"

    # Truncate to terminal width so \r overwrites cleanly
    cols = shutil.get_terminal_size().columns
    if len(line) > cols - 1:
        line = line[:cols - 1]
    return f"\r{line}" + " " * max(0, cols - 1 - len(line))


def train_against(agent, opponent, opponent_name, episodes, save_every=500,
                   save_dir="models/ppo", self_play_update=3000, n_envs=64):
    """Train PPO agent against opponent using n_envs parallel games."""
    vec_env = VecConnect4Env(n_envs, opponent)
    os.makedirs(save_dir, exist_ok=True)
    is_self_play = isinstance(opponent, SelfPlayOpponent)

    lr = agent.optimizer.param_groups[0]["lr"]
    print(f"\n{'='*60}")
    print(f"Training PPO vs {opponent_name} for {episodes} episodes ({n_envs} parallel games)")
    print(f"  lr={lr:.1e} | rollout={agent.rollout_steps} steps | epochs={agent.ppo_epochs} | clip={agent.clip_eps}")
    if is_self_play:
        print(f"  snapshot updates every {self_play_update} episodes")
    print(f"{'='*60}")

    reward_history = []
    t_start = time.time()
    last_print = 0.0
    last_checkpoint = 0

    ep_count = 0
    states = vec_env.reset_all()

    while ep_count < episodes:
        # --- Collect rollout ---
        agent.rollout.clear()
        for step in range(agent.rollout_steps):
            legal_batch = vec_env.get_legal_actions_batch()
            actions, log_probs, values, legal_masks = agent.select_actions_batch(
                states, legal_batch
            )

            next_states, rewards, dones, next_legals = vec_env.step(actions)

            # Store each env's transition
            for i in range(n_envs):
                agent.rollout.push(
                    states[i], actions[i], log_probs[i],
                    rewards[i], dones[i], values[i], legal_masks[i],
                )

            # Count finished episodes
            n_done = int(dones.sum())
            for i in range(n_envs):
                if dones[i]:
                    reward_history.append(rewards[i])
            ep_count += n_done

            # Self-play snapshot update
            if is_self_play and n_done > 0 and ep_count % self_play_update < n_done:
                opponent.update_snapshot(agent)
                vec_env.opponent = opponent

            states = vec_env.get_states()

            # Throttled progress
            now = time.time()
            if now - last_print >= 0.5:
                recent = reward_history[-500:] if reward_history else []
                win_rate = sum(1 for r in recent if r > 0) / len(recent) if recent else 0
                sys.stdout.write(progress_line(
                    min(ep_count, episodes), episodes, t_start,
                    extra=f"win={win_rate:.0%}"
                ))
                sys.stdout.flush()
                last_print = now

            if ep_count >= episodes:
                break

        # --- PPO update ---
        next_states_for_bootstrap = vec_env.get_states()
        agent.update(next_states_for_bootstrap)

        # Periodic checkpoint
        if ep_count - last_checkpoint >= save_every:
            agent.save(os.path.join(save_dir, "latest.pt"))
            last_checkpoint = ep_count

    # Final progress + save
    recent = reward_history[-500:] if reward_history else []
    win_rate = sum(1 for r in recent if r > 0) / len(recent) if recent else 0
    sys.stdout.write(progress_line(
        min(ep_count, episodes), episodes, t_start,
        extra=f"win={win_rate:.0%}"
    ))
    sys.stdout.flush()
    print()

    agent.save(os.path.join(save_dir, "latest.pt"))
    return agent


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for Connect 4")
    parser.add_argument(
        "--opponent", type=str, default="random",
        choices=["random", "heuristic", "minimax", "self"],
    )
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--clip-eps", type=float, default=None, help="PPO clip epsilon")
    parser.add_argument("--entropy-coef", type=float, default=None, help="Entropy bonus coefficient")
    parser.add_argument("--rollout-steps", type=int, default=None, help="Steps per rollout")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="models/ppo")
    parser.add_argument("--n-envs", type=int, default=64, help="Number of parallel games (default: 64)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Build agent with any overridden hyperparams
    agent_kwargs = {}
    if args.rollout_steps is not None:
        agent_kwargs["rollout_steps"] = args.rollout_steps
    agent = PPOAgent(**agent_kwargs)

    # Auto-resume
    latest_path = os.path.join(args.save_dir, "latest.pt")
    if not args.fresh and os.path.isfile(latest_path):
        print(f"Resuming from {latest_path}")
        agent.load(latest_path)
    elif not args.fresh:
        print("No existing model found — starting fresh")

    # Override hyperparams
    if args.lr is not None:
        for pg in agent.optimizer.param_groups:
            pg["lr"] = args.lr
    if args.clip_eps is not None:
        agent.clip_eps = args.clip_eps
    if args.entropy_coef is not None:
        agent.entropy_coef = args.entropy_coef

    if args.opponent == "self":
        opponent = SelfPlayOpponent(agent)
    else:
        opponents = {
            "random": RandomOpponent(),
            "heuristic": HeuristicOpponent(),
            "minimax": MinimaxOpponent(depth=4),
        }
        opponent = opponents[args.opponent]

    train_against(
        agent, opponent, args.opponent,
        episodes=args.episodes, save_dir=args.save_dir,
        n_envs=args.n_envs,
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
