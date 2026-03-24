"""DQN training script.

Auto-resumes from models/dqn/latest.pt if it exists.

Usage:
    python -m training.train_dqn --opponent random --episodes 5000
    python -m training.train_dqn --opponent heuristic --episodes 10000
    python -m training.train_dqn --opponent minimax --episodes 20000
    python -m training.train_dqn --fresh --opponent random   # ignore latest, start fresh
"""

import argparse
import os
import sys
import time
import random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.connect4 import Connect4Env
from envs.vec_connect4 import VecConnect4Env
from agents.dqn.agent import DQNAgent
from opponents import RandomOpponent, HeuristicOpponent, MinimaxOpponent
from opponents.self_play_opponent import SelfPlayOpponent

N_ENVS = 16  # number of parallel games


def play_one_game(env, agent, opponent, agent_player):
    """Play a full game. Return (reward, loss_list).

    agent_player: 1 or 2 — which side the DQN agent plays.
    """
    env.reset()
    losses = []

    # If opponent goes first, let them move
    if agent_player == 2:
        opp_action = opponent.select_action(env)
        env.step(opp_action)

    while not env.done:
        # --- Agent's turn ---
        state = env.get_state()  # from agent's perspective (current player)
        action = agent.select_action(env)
        _, reward, done, info = env.step(action)

        if done:
            # Agent just moved and game ended
            # reward is from perspective of the player who moved (the agent)
            next_state = state  # terminal — doesn't matter
            agent.store_transition(state, action, reward, next_state, True, [])
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            return reward, losses

        # --- Opponent's turn ---
        opp_action = opponent.select_action(env)
        _, opp_reward, done, info = env.step(opp_action)

        if done:
            # Opponent just won or it's a draw
            # opp_reward is from the opponent's perspective:
            #   +1 means opponent won -> agent lost -> agent gets -1
            #    0 means draw -> agent gets 0
            agent_reward = -opp_reward
            next_state = env.get_state()  # terminal state
            agent.store_transition(state, action, agent_reward, next_state, True, [])
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            return agent_reward, losses

        # --- Neither won — store intermediate transition ---
        next_state = env.get_state()  # now it's agent's turn again
        next_legal = env.get_legal_actions()
        agent.store_transition(state, action, 0.0, next_state, False, next_legal)
        loss = agent.update()
        if loss is not None:
            losses.append(loss)

    return 0.0, losses


def evaluate(env, agent, opponent, n_games=100):
    """Evaluate agent vs opponent (no exploration). Returns win, draw, loss counts."""
    results = {"win": 0, "draw": 0, "loss": 0}
    for i in range(n_games):
        agent_player = 1 if i % 2 == 0 else 2  # alternate
        env.reset()

        if agent_player == 2:
            opp_action = opponent.select_action(env)
            env.step(opp_action)

        while not env.done:
            action = agent.select_action(env, greedy=True)
            _, reward, done, _ = env.step(action)
            if done:
                if reward == 1.0:
                    results["win"] += 1
                elif reward == 0.0:
                    results["draw"] += 1
                break

            opp_action = opponent.select_action(env)
            _, opp_reward, done, _ = env.step(opp_action)
            if done:
                if opp_reward == 1.0:
                    results["loss"] += 1
                else:
                    results["draw"] += 1
                break

    return results


def fmt_time(seconds):
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def progress_line(ep, total, t_start, extra=""):
    """Return a progress string: bar, speed, elapsed, ETA, plus optional extra info."""
    elapsed = time.time() - t_start
    eps_per_sec = ep / elapsed if elapsed > 0 else 0
    remaining = (total - ep) / eps_per_sec if eps_per_sec > 0 else 0

    pct = ep / total if total > 0 else 0
    bar_len = 20
    filled = int(bar_len * pct)
    bar = "█" * filled + "░" * (bar_len - filled)

    line = (
        f"\r  {bar} {ep}/{total} ({pct:.0%}) | "
        f"{eps_per_sec:.1f} ep/s | "
        f"{fmt_time(elapsed)}<{fmt_time(remaining)}"
    )
    if extra:
        line += f" | {extra}"
    return line


def train_against(agent, opponent, opponent_name, episodes, save_every=500,
                   save_dir="models/dqn", self_play_update=3000):
    """Train DQN agent against opponent using N_ENVS parallel games."""
    vec_env = VecConnect4Env(N_ENVS, opponent)
    os.makedirs(save_dir, exist_ok=True)
    is_self_play = isinstance(opponent, SelfPlayOpponent)

    lr = agent.optimizer.param_groups[0]["lr"]
    print(f"\n{'='*60}")
    print(f"Training DQN vs {opponent_name} for {episodes} episodes ({N_ENVS} parallel games)")
    print(f"  lr={lr:.1e} | ε={agent.epsilon:.3f}→{agent.epsilon_end} over {agent.epsilon_decay_steps} steps")
    if is_self_play:
        print(f"  snapshot updates every {self_play_update} episodes")
    print(f"{'='*60}")

    reward_history = []
    t_start = time.time()
    last_print = 0.0
    last_checkpoint = 0

    ep_count = 0
    states = vec_env.reset_all()  # (N, 6, 7, 2)

    while ep_count < episodes:
        legal_batch = vec_env.get_legal_actions_batch()
        actions = agent.select_actions_batch(states, legal_batch)

        next_states, rewards, dones, next_legals = vec_env.step(actions)

        # Store transitions for all envs (routed through n-step buffer)
        for i in range(N_ENVS):
            agent.store_transition(
                states[i], actions[i], rewards[i],
                next_states[i], dones[i], next_legals[i],
                env_id=i,
            )

        # Train on a batch from replay buffer
        agent.update()

        # Count finished episodes
        n_done = int(dones.sum())
        for i in range(N_ENVS):
            if dones[i]:
                reward_history.append(rewards[i])
        ep_count += n_done

        # Self-play snapshot update
        if is_self_play and n_done > 0 and ep_count % self_play_update < n_done:
            opponent.update_snapshot(agent)
            vec_env.opponent = opponent

        # After reset, get fresh states from vec_env
        states = vec_env.get_states()

        # Throttled progress bar
        now = time.time()
        if now - last_print >= 0.5 or ep_count >= episodes:
            # Rolling win rate from last 500 episodes
            recent = reward_history[-500:] if reward_history else []
            win_rate = sum(1 for r in recent if r > 0) / len(recent) if recent else 0

            sys.stdout.write(progress_line(
                min(ep_count, episodes), episodes, t_start,
                extra=f"ε={agent.epsilon:.3f} | win={win_rate:.0%}"
            ))
            sys.stdout.flush()
            last_print = now

        # Periodic checkpoint (no eval)
        if ep_count - last_checkpoint >= save_every:
            agent.save(os.path.join(save_dir, "latest.pt"))
            last_checkpoint = ep_count

        if ep_count >= episodes:
            break

    print()
    agent.save(os.path.join(save_dir, "latest.pt"))

    return agent


def main():
    parser = argparse.ArgumentParser(description="Train DQN agent for Connect 4")
    parser.add_argument(
        "--opponent", type=str, default="random",
        choices=["random", "heuristic", "minimax", "self"],
        help="Opponent to train against (default: random)",
    )
    parser.add_argument("--episodes", type=int, default=10_000, help="Number of episodes")
    parser.add_argument("--eps-start", type=float, default=None, help="Starting epsilon (default: 1.0, or current if resuming)")
    parser.add_argument("--eps-end", type=float, default=0.05, help="Minimum epsilon (default: 0.05)")
    parser.add_argument("--eps-decay", type=int, default=80_000, help="Epsilon decay steps (default: 80000)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 1e-4, or current if resuming)")
    parser.add_argument("--fresh", action="store_true", help="Start from scratch (ignore latest.pt)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-dir", type=str, default="models/dqn", help="Directory to save models")
    args = parser.parse_args()

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    agent = DQNAgent(
        epsilon_end=args.eps_end,
        epsilon_decay_steps=args.eps_decay,
    )

    # Auto-resume from latest.pt unless --fresh
    latest_path = os.path.join(args.save_dir, "latest.pt")
    if not args.fresh and os.path.isfile(latest_path):
        print(f"Resuming from {latest_path}")
        agent.load(latest_path)
    elif not args.fresh:
        print("No existing model found — starting fresh")

    # Override epsilon if --eps-start is given
    if args.eps_start is not None:
        agent.epsilon_start = args.eps_start
        agent.steps_done = 0  # reset decay schedule to start from eps_start

    # Override learning rate if --lr is given
    if args.lr is not None:
        for param_group in agent.optimizer.param_groups:
            param_group["lr"] = args.lr

    if args.opponent == "self":
        opponent = SelfPlayOpponent(agent)
    else:
        opponents = {
            "random": RandomOpponent(),
            "heuristic": HeuristicOpponent(),
            "minimax": MinimaxOpponent(depth=5),
        }
        opponent = opponents[args.opponent]

    train_against(
        agent, opponent, args.opponent,
        episodes=args.episodes, save_dir=args.save_dir,
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
