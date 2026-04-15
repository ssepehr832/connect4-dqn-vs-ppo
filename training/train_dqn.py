"""DQN training utilities and CLI."""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import time
from collections import deque

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.dqn.agent import DQNAgent
from envs.vec_connect4 import VecConnect4Env
from opponents import HeuristicOpponent, MinimaxOpponent, RandomOpponent
from opponents.self_play_opponent import SelfPlayOpponent
from training.artifacts import append_csv_row, ensure_dir, now_iso

N_ENVS = 16
DRAW_REWARD = 0.2


def seed_everything(seed):
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fmt_time(seconds):
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{sec:02d}s"


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
        f"  {bar} {ep}/{total} ({pct:.0%}) | "
        f"{eps_per_sec:.1f} ep/s | "
        f"{fmt_time(elapsed)}<{fmt_time(remaining)}"
    )
    if extra:
        line += f" | {extra}"

    cols = shutil.get_terminal_size().columns
    if len(line) > cols - 1:
        line = line[:cols - 1]
    return f"\r{line}" + " " * max(0, cols - 1 - len(line))


def _rolling_stats(values):
    """Return mean/std for a deque or list."""
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float32)
    return float(arr.mean()), float(arr.std())


def _outcome_rates(outcomes):
    """Return win/draw/loss rates from a sequence of outcome labels."""
    total = len(outcomes)
    if total == 0:
        return 0.0, 0.0, 0.0
    wins = sum(1 for item in outcomes if item == "win")
    draws = sum(1 for item in outcomes if item == "draw")
    losses = total - wins - draws
    return wins / total, draws / total, losses / total


def _find_snapshot_offset(save_dir):
    """Return the highest episode number across existing snapshot_*.pt files."""
    if not os.path.isdir(save_dir):
        return 0
    max_ep = 0
    for name in os.listdir(save_dir):
        if name.startswith("snapshot_") and name.endswith(".pt"):
            stem = name[len("snapshot_"):-len(".pt")]
            try:
                ep = int(stem)
            except ValueError:
                continue
            if ep > max_ep:
                max_ep = ep
    return max_ep


def _find_csv_episode_offset(path):
    """Return the max value in the 'episode' column of an existing CSV, or 0."""
    if not os.path.exists(path):
        return 0
    import csv as _csv

    max_ep = 0
    try:
        with open(path, "r") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                try:
                    ep = int(float(row.get("episode", 0)))
                    if ep > max_ep:
                        max_ep = ep
                except (ValueError, TypeError):
                    continue
    except Exception:
        return 0
    return max_ep


def find_episode_offset(save_dir):
    """Return the highest episode number seen in this save_dir across
    snapshot filenames, metrics.csv, and eval.csv.
    """
    offsets = [_find_snapshot_offset(save_dir)]
    for name in ("metrics.csv", "eval.csv"):
        offsets.append(_find_csv_episode_offset(os.path.join(save_dir, name)))
    return max(offsets)


class MetricsLogger:
    """Classmate's MetricsLogger to maintain metrics.csv compatibility."""

    HEADER = (
        "episode,phase,epsilon,unique_pct,"
        "win_rate,win_rate_self,win_rate_minimax\n"
    )

    def __init__(self, save_dir, phase, fresh=True):
        self.path = os.path.join(save_dir, "metrics.csv")
        self.phase = phase
        self._ensure_header(fresh)

    def _ensure_header(self, fresh):
        if fresh or not os.path.exists(self.path):
            with open(self.path, "w") as f:
                f.write(self.HEADER)
            return
        with open(self.path, "r") as f:
            first_line = f.readline()
        if first_line != self.HEADER:
            with open(self.path, "w") as f:
                f.write(self.HEADER)

    def log(
        self,
        episode,
        epsilon,
        unique_pct,
        win_rate=None,
        win_rate_self=None,
        win_rate_minimax=None,
    ):
        wr = f"{win_rate:.4f}" if win_rate is not None else ""
        wrs = f"{win_rate_self:.4f}" if win_rate_self is not None else ""
        wrm = f"{win_rate_minimax:.4f}" if win_rate_minimax is not None else ""
        with open(self.path, "a") as f:
            f.write(
                f"{episode},{self.phase},{epsilon:.4f},{unique_pct:.4f},"
                f"{wr},{wrs},{wrm}\n"
            )


class EvalLogger:
    """Classmate's EvalLogger to maintain eval.csv compatibility."""

    HEADER = "episode,phase,vs_random,vs_heuristic,vs_minimax\n"

    def __init__(self, save_dir, phase, fresh=True):
        self.path = os.path.join(save_dir, "eval.csv")
        self.phase = phase
        self._ensure_header(fresh)

    def _ensure_header(self, fresh):
        if fresh or not os.path.exists(self.path):
            with open(self.path, "w") as f:
                f.write(self.HEADER)
            return
        with open(self.path, "r") as f:
            first_line = f.readline()
        if first_line != self.HEADER:
            with open(self.path, "w") as f:
                f.write(self.HEADER)

    def log(self, episode, results):
        with open(self.path, "a") as f:
            f.write(
                f"{episode},{self.phase},{results['random']:.4f},"
                f"{results['heuristic']:.4f},{results['minimax']:.4f}\n"
            )


def run_periodic_eval(agent, games=50, depth=4):
    """Evaluate against fixed opponents for the in-loop eval."""
    from evaluation.evaluate import evaluate, make_opponent

    results = {}
    for name in ["random", "heuristic", "minimax"]:
        opp = make_opponent(name, depth=depth)
        w, d, l, _, _ = evaluate(agent, opp, games)
        total = w + d + l
        results[name] = w / total if total > 0 else 0.0
    return results


def make_checkpoint_metadata(
    agent,
    *,
    stage_name,
    opponent_name,
    stage_episode,
    cumulative_episode,
    n_envs,
    seed,
    save_dir,
    freeze_conv,
    official_hybrid_depth,
    arbiter,
    arbiter_min_pieces,
    run_metadata=None,
):
    """Build metadata stored alongside each checkpoint."""
    lr = agent.optimizer.param_groups[0]["lr"]
    metadata = {
        "timestamp": now_iso(),
        "stage_name": stage_name,
        "opponent_name": opponent_name,
        "stage_episode": stage_episode,
        "cumulative_episode": cumulative_episode,
        "seed": seed,
        "device": str(agent.device),
        "n_envs": n_envs,
        "learning_rate": lr,
        "epsilon_start": agent.epsilon_start,
        "epsilon_end": agent.epsilon_end,
        "epsilon_decay_steps": agent.epsilon_decay_steps,
        "steps_done": agent.steps_done,
        "freeze_conv": freeze_conv,
        "official_hybrid_depth": official_hybrid_depth,
        "arbiter_enabled": arbiter is not None,
        "arbiter_depth": getattr(arbiter, "depth", None),
        "arbiter_min_pieces": arbiter_min_pieces if arbiter is not None else None,
        "save_dir": os.path.abspath(save_dir),
        "command": " ".join(sys.argv),
    }
    if run_metadata:
        metadata.update(run_metadata)
    return metadata


def save_training_checkpoint(
    agent,
    *,
    checkpoints_dir,
    checkpoint_label,
    alias_latest_path,
    metadata,
):
    """Save a versioned checkpoint and update the latest alias."""
    ensure_dir(checkpoints_dir)
    checkpoint_path = os.path.join(checkpoints_dir, f"{checkpoint_label}.pt")
    agent.save(checkpoint_path, metadata=metadata)
    if alias_latest_path:
        shutil.copy2(checkpoint_path, alias_latest_path)
    return checkpoint_path


def maybe_log_metrics(
    *,
    metrics_path,
    stage_name,
    opponent_name,
    train_mode,
    stage_episode,
    cumulative_episode,
    total_target_episodes,
    agent,
    tracker,
    unique_games,
    total_games,
    seed,
    official_hybrid_depth,
    n_envs,
    freeze_conv,
    t_start,
):
    """Append our rich metrics to training_metrics.csv."""
    if not metrics_path:
        return None

    reward_mean, reward_std = _rolling_stats(tracker["recent_rewards"])
    loss_mean, loss_std = _rolling_stats(tracker["recent_losses"])
    length_mean, length_std = _rolling_stats(tracker["recent_lengths"])
    win_rate, draw_rate, loss_rate = _outcome_rates(tracker["recent_outcomes"])

    recent_terms = tracker["recent_termination_sources"]
    solved_rate = (
        sum(1 for source in recent_terms if source == "solved") / len(recent_terms)
        if recent_terms else 0.0
    )
    natural_rate = (
        sum(1 for source in recent_terms if source == "natural") / len(recent_terms)
        if recent_terms else 0.0
    )
    elapsed = time.time() - t_start
    eps_per_sec = stage_episode / elapsed if elapsed > 0 else 0.0
    lr = agent.optimizer.param_groups[0]["lr"]

    row = {
        "timestamp": now_iso(),
        "stage_name": stage_name,
        "opponent_name": opponent_name,
        "train_mode": train_mode,
        "stage_episode": stage_episode,
        "cumulative_episode": cumulative_episode,
        "stage_progress": round(stage_episode / max(total_target_episodes, 1), 6),
        "n_envs": n_envs,
        "seed": seed,
        "device": str(agent.device),
        "learning_rate": round(float(lr), 10),
        "epsilon": round(float(agent.epsilon), 6),
        "steps_done": int(agent.steps_done),
        "replay_size": len(agent.replay_buffer),
        "reward_mean": round(reward_mean, 6),
        "reward_std": round(reward_std, 6),
        "td_loss_mean": round(loss_mean, 6),
        "td_loss_std": round(loss_std, 6),
        "win_rate": round(win_rate, 6),
        "draw_rate": round(draw_rate, 6),
        "loss_rate": round(loss_rate, 6),
        "game_length_mean": round(length_mean, 6),
        "game_length_std": round(length_std, 6),
        "episodes_per_second": round(eps_per_sec, 6),
        "unique_game_ratio": round(len(unique_games) / total_games, 6) if total_games else 1.0,
        "solved_termination_rate": round(solved_rate, 6),
        "natural_termination_rate": round(natural_rate, 6),
        "freeze_conv": bool(freeze_conv),
        "official_hybrid_depth": official_hybrid_depth,
        "window_episodes": len(tracker["recent_outcomes"]),
    }
    append_csv_row(metrics_path, row)
    return row


def _consume_finished_records(records, tracker, unique_games):
    """Update rolling trackers from finished-game records."""
    for record in records:
        tracker["recent_outcomes"].append(record["outcome"])
        tracker["recent_rewards"].append(record["reward"])
        tracker["recent_lengths"].append(record["game_length"])
        tracker["recent_termination_sources"].append(record["termination_source"])
        unique_games.add(record["game_hash"])


def _print_header(agent, opponent_name, episodes, n_envs, self_play_update, chunk_size, arbiter, self_play_frac=0.5):
    """Print the stage header."""
    lr = agent.optimizer.param_groups[0]["lr"]
    print(f"\n{'='*60}")
    print(f"Training DQN vs {opponent_name} for {episodes} episodes ({n_envs} parallel games)")
    print(f"  lr={lr:.1e} | ε={agent.epsilon:.3f}→{agent.epsilon_end} over {agent.epsilon_decay_steps} steps")
    if self_play_update is not None:
        print(f"  snapshot updates every {self_play_update} episodes")
    if opponent_name == "self-mixed":
        # Ensure we show at least 1 episode per chunk in the header if chunk_size is valid
        c_size = chunk_size or 1000
        s_chunk = max(1, int(round(c_size * self_play_frac)))
        m_chunk = max(1, c_size - s_chunk)
        print(f"  cycle: {s_chunk} self-play + {m_chunk} minimax")
    elif chunk_size is not None:
        print(f"  chunks of {chunk_size} episodes")
    if arbiter is not None:
        print(f"  arbiter: minimax depth {arbiter.depth}, active after configured piece threshold")
    print(f"{'='*60}")


def run_training_stage(
    agent,
    *,
    opponent_name,
    episodes,
    save_dir="models/dqn",
    checkpoint_every=500,
    metrics_every=250,
    rolling_window=500,
    self_play_update=3000,
    n_envs=16,
    arbiter=None,
    arbiter_min_pieces=12,
    seed=42,
    stage_name=None,
    stage_episode_start=0,
    cumulative_start=0,
    official_hybrid_depth=4,
    freeze_conv=False,
    run_metadata=None,
    on_checkpoint=None,
    self_play_frac=0.5,
    snapshot_every=10000,
    eval_every=0,
    eval_games=50,
    eval_depth=4,
    fresh_logs=True,
):
    """Unified training loop supporting single opponent and self-mixed curriculum."""
    stage_name = stage_name or opponent_name
    ensure_dir(save_dir)
    checkpoints_dir = ensure_dir(os.path.join(save_dir, "checkpoints"))
    alias_latest_path = os.path.join(save_dir, "latest.pt")
    metrics_path = run_metadata.get("metrics_path") if run_metadata else None

    # Trackers for our rich metrics
    tracker = {
        "recent_rewards": deque(maxlen=rolling_window),
        "recent_losses": deque(maxlen=rolling_window),
        "recent_outcomes": deque(maxlen=rolling_window),
        "recent_lengths": deque(maxlen=rolling_window),
        "recent_termination_sources": deque(maxlen=rolling_window),
    }
    # Trackers for classmate's metrics.csv
    reward_history_self = []
    reward_history_minimax = []
    
    unique_games = set()
    total_games = 0
    last_print = 0.0
    last_checkpoint = stage_episode_start
    last_metrics = stage_episode_start
    last_snapshot = stage_episode_start
    last_metrics_log = stage_episode_start
    last_eval = stage_episode_start
    t_start = time.time()

    # Offset for classmate's cumulative timeline
    episode_offset = run_metadata.get("episode_offset", 0) if run_metadata else 0
    phase_label = run_metadata.get("phase_label") if run_metadata else stage_name
    
    metrics_logger = MetricsLogger(save_dir, phase=phase_label, fresh=fresh_logs)
    eval_logger = EvalLogger(save_dir, phase=phase_label, fresh=fresh_logs) if eval_every > 0 else None

    if opponent_name == "self-mixed":
        self_opp = SelfPlayOpponent(agent)
        minimax_opp = MinimaxOpponent(depth=5)
        vec_self = VecConnect4Env(n_envs, self_opp, arbiter=arbiter, arbiter_min_pieces=arbiter_min_pieces)
        vec_minimax = VecConnect4Env(n_envs, minimax_opp)
        chunk_size = int(run_metadata.get("chunk_size", 1000)) if run_metadata else 1000
        self_chunk = max(1, int(round(chunk_size * self_play_frac)))
        minimax_chunk = max(1, chunk_size - self_chunk)
        
        _print_header(agent, opponent_name, episodes, n_envs, self_play_update, chunk_size, arbiter, self_play_frac)
        vec_env = vec_self
        use_self = True
        chunk_ep = 0
        current_chunk_size = self_chunk
        states = vec_env.reset_all()
        opponent_obj = None
    else:
        if opponent_name == "self":
            opponent_obj = SelfPlayOpponent(agent)
        elif opponent_name == "random":
            opponent_obj = RandomOpponent()
        elif opponent_name == "heuristic":
            opponent_obj = HeuristicOpponent()
        elif opponent_name == "minimax":
            opponent_obj = MinimaxOpponent(depth=5)
        else:
            raise ValueError(f"Unknown training opponent: {opponent_name}")
        
        chunk_size = None
        _print_header(agent, opponent_name, episodes, n_envs, self_play_update if opponent_name == "self" else None, None, arbiter)
        vec_env = VecConnect4Env(n_envs, opponent_obj, arbiter=arbiter, arbiter_min_pieces=arbiter_min_pieces)
        use_self = False
        chunk_ep = 0
        states = vec_env.reset_all()

    ep_count = stage_episode_start
    last_metric_row = None

    while ep_count < episodes:
        legal_batch = vec_env.get_legal_actions_batch()
        actions = agent.select_actions_batch(states, legal_batch)
        next_states, rewards, dones, next_legals = vec_env.step(actions)

        for i in range(n_envs):
            agent.store_transition(states[i], actions[i], rewards[i], next_states[i], dones[i], next_legals[i], env_id=i)

        n_updates = max(1, n_envs // agent.batch_size)
        for _ in range(n_updates):
            loss = agent.update()
            if loss is not None:
                tracker["recent_losses"].append(loss)
        agent.step_schedule()

        # Capture the train_mode that generated the transitions in this step
        if opponent_name == "self-mixed":
            train_mode = "self" if use_self else "minimax"
        else:
            train_mode = opponent_name

        finished_records = vec_env.drain_game_records()
        n_done = len(finished_records)
        if n_done:
            # Clamp to remaining budget to prevent overshoot
            remaining = episodes - ep_count
            if n_done > remaining:
                finished_records = finished_records[:remaining]
                n_done = remaining

            _consume_finished_records(finished_records, tracker, unique_games)
            # Update reward history for classmate's win-rate metrics
            cur_hist = reward_history_self if use_self else reward_history_minimax
            for record in finished_records:
                cur_hist.append(record["reward"])

            total_games += n_done
            ep_count += n_done
            chunk_ep += n_done

        # Periodic updates (self-play snapshot, chunk switching)
        if opponent_name == "self-mixed":
            if use_self and n_done > 0 and ep_count % self_play_update < n_done:
                self_opp.update_snapshot(agent)
                vec_self.opponent = self_opp

            if chunk_ep >= current_chunk_size:
                chunk_ep = 0
                use_self = not use_self
                current_chunk_size = self_chunk if use_self else minimax_chunk
                vec_env = vec_self if use_self else vec_minimax
                agent.flush_n_step_buffers()
                states = vec_env.reset_all()
            else:
                states = vec_env.get_states()
        else:
            if (
                opponent_name == "self"
                and n_done > 0
                and ep_count % self_play_update < n_done
            ):
                opponent_obj.update_snapshot(agent)
                vec_env.opponent = opponent_obj
            states = vec_env.get_states()

        # UI Progress
        now = time.time()
        if now - last_print >= 0.5 or ep_count >= episodes:
            win_rate, draw_rate, _ = _outcome_rates(tracker["recent_outcomes"])
            uniq_pct = len(unique_games) / total_games if total_games > 0 else 1.0
            extra = f"ε={agent.epsilon:.3f} | W={win_rate:.0%} | D={draw_rate:.0%} | uniq={uniq_pct:.0%}"
            if opponent_name == "self-mixed":
                extra += f" | {train_mode}"
            sys.stdout.write(progress_line(min(ep_count, episodes), episodes, t_start, extra=extra))
            sys.stdout.flush()
            last_print = now

        cumulative_episode = cumulative_start + ep_count
        
        # Log our rich metrics
        if ep_count - last_metrics >= metrics_every and ep_count > 0:
            last_metric_row = maybe_log_metrics(
                metrics_path=metrics_path, stage_name=stage_name, opponent_name=opponent_name,
                train_mode=train_mode, stage_episode=min(ep_count, episodes),
                cumulative_episode=cumulative_episode, total_target_episodes=episodes,
                agent=agent, tracker=tracker, unique_games=unique_games, total_games=total_games,
                seed=seed, official_hybrid_depth=official_hybrid_depth, n_envs=n_envs,
                freeze_conv=freeze_conv, t_start=t_start
            )
            last_metrics = ep_count

        # Checkpoints
        if ep_count - last_checkpoint >= checkpoint_every and 0 < ep_count < episodes:
            checkpoint_label = f"{stage_name}_ep{ep_count:07d}_total{cumulative_episode:07d}"
            metadata = make_checkpoint_metadata(
                agent, stage_name=stage_name, opponent_name=opponent_name,
                stage_episode=min(ep_count, episodes), cumulative_episode=cumulative_episode,
                n_envs=n_envs, seed=seed, save_dir=save_dir, freeze_conv=freeze_conv,
                official_hybrid_depth=official_hybrid_depth, arbiter=arbiter,
                arbiter_min_pieces=arbiter_min_pieces, run_metadata=run_metadata
            )
            checkpoint_path = save_training_checkpoint(agent, checkpoints_dir=checkpoints_dir, checkpoint_label=checkpoint_label, alias_latest_path=alias_latest_path, metadata=metadata)
            if on_checkpoint:
                on_checkpoint({"checkpoint_path": checkpoint_path, "checkpoint_label": checkpoint_label, "stage_name": stage_name, "opponent_name": opponent_name, "stage_episode": min(ep_count, episodes), "cumulative_episode": cumulative_episode, "train_mode": train_mode, "metadata": metadata})
            last_checkpoint = ep_count

        # Periodic snapshots (Classmate's feature)
        if ep_count - last_snapshot >= snapshot_every:
            cum_offset = episode_offset + ep_count
            snap_path = os.path.join(save_dir, f"snapshot_{cum_offset}.pt")
            agent.save(snap_path)
            last_snapshot = ep_count

        # Periodic metrics.csv log (Classmate's feature)
        if ep_count - last_metrics_log >= metrics_every:
            def _win_rate(hist):
                recent = hist[-500:] if hist else []
                return sum(1 for r in recent if r > 0) / len(recent) if recent else None
            
            uniq_pct = len(unique_games) / total_games if total_games > 0 else 1.0
            metrics_logger.log(
                episode=episode_offset + ep_count, epsilon=agent.epsilon, unique_pct=uniq_pct,
                win_rate=_win_rate(reward_history_self if opponent_name != "self-mixed" else reward_history_self + reward_history_minimax),
                win_rate_self=_win_rate(reward_history_self) if opponent_name == "self-mixed" else None,
                win_rate_minimax=_win_rate(reward_history_minimax) if opponent_name == "self-mixed" else None
            )
            last_metrics_log = ep_count

        # Periodic eval.csv log (Classmate's feature)
        if eval_logger is not None and ep_count - last_eval >= eval_every:
            sys.stdout.write("\n  running periodic eval...")
            sys.stdout.flush()
            results = run_periodic_eval(agent, games=eval_games, depth=eval_depth)
            eval_logger.log(episode_offset + ep_count, results)
            sys.stdout.write(f"\r  eval @ {episode_offset + ep_count}: rnd={results['random']:.0%} heur={results['heuristic']:.0%} mm={results['minimax']:.0%}\n")
            sys.stdout.flush()
            last_eval = ep_count

        if ep_count >= episodes:
            break

    # Final Save
    print()
    final_cumulative = cumulative_start + ep_count
    final_label = f"{stage_name}_final_total{final_cumulative:07d}"
    final_metadata = make_checkpoint_metadata(
        agent, stage_name=stage_name, opponent_name=opponent_name,
        stage_episode=ep_count, cumulative_episode=final_cumulative,
        n_envs=n_envs, seed=seed, save_dir=save_dir, freeze_conv=freeze_conv,
        official_hybrid_depth=official_hybrid_depth, arbiter=arbiter,
        arbiter_min_pieces=arbiter_min_pieces, run_metadata=run_metadata
    )
    final_path = save_training_checkpoint(agent, checkpoints_dir=checkpoints_dir, checkpoint_label=final_label, alias_latest_path=alias_latest_path, metadata=final_metadata)

    if on_checkpoint:
        on_checkpoint({"checkpoint_path": final_path, "checkpoint_label": final_label, "stage_name": stage_name, "opponent_name": opponent_name, "stage_episode": ep_count, "cumulative_episode": final_cumulative, "train_mode": train_mode, "metadata": final_metadata, "final": True})

    if last_metrics < ep_count:
        last_metric_row = maybe_log_metrics(
            metrics_path=metrics_path, stage_name=stage_name, opponent_name=opponent_name,
            train_mode=train_mode, stage_episode=ep_count,
            cumulative_episode=final_cumulative, total_target_episodes=episodes,
            agent=agent, tracker=tracker, unique_games=unique_games, total_games=total_games,
            seed=seed, official_hybrid_depth=official_hybrid_depth, n_envs=n_envs,
            freeze_conv=freeze_conv, t_start=t_start
        )

    return {
        "agent": agent, "stage_name": stage_name, "opponent_name": opponent_name,
        "episodes_completed": ep_count, "cumulative_episode": final_cumulative,
        "latest_checkpoint_path": final_path, "latest_checkpoint_label": final_label,
        "metrics_row": last_metric_row
    }


def build_agent(*, eps_end=0.05, eps_decay=80_000, n_envs=16):
    """Construct a DQN agent with a replay buffer sized for the chosen parallelism."""
    buffer_cap = max(100_000, n_envs * 100)
    return DQNAgent(epsilon_end=eps_end, epsilon_decay_steps=eps_decay, buffer_capacity=buffer_cap, n_envs=n_envs)


def main():
    parser = argparse.ArgumentParser(description="Train DQN agent for Connect 4")
    parser.add_argument("--opponent", type=str, default="random", choices=["random", "heuristic", "minimax", "self", "self-mixed"])
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--eps-start", type=float, default=None)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=int, default=80000)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="models/dqn")
    parser.add_argument("--load-path", type=str, default=None)
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    parser.add_argument("--arbiter", action="store_true")
    parser.add_argument("--arbiter-depth", type=int, default=4)
    parser.add_argument("--arbiter-min-pieces", type=int, default=12)
    parser.add_argument("--freeze-conv", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--metrics-every", type=int, default=250)
    parser.add_argument("--metrics-path", type=str, default=None)
    parser.add_argument("--stage-name", type=str, default=None)
    parser.add_argument("--cumulative-start", type=int, default=0)
    parser.add_argument("--official-hybrid-depth", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--self-play-frac", type=float, default=0.5)
    parser.add_argument("--eval-every", type=int, default=10000)
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--eval-depth", type=int, default=4)
    parser.add_argument("--phase-label", type=str, default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    agent = build_agent(eps_end=args.eps_end, eps_decay=args.eps_decay, n_envs=args.n_envs)

    if args.freeze_conv:
        agent.set_freeze_conv(True)

    latest_path = os.path.join(args.save_dir, "latest.pt")
    load_path = args.load_path or (latest_path if not args.fresh and os.path.isfile(latest_path) else None)
    if load_path:
        print(f"Loading checkpoint from {load_path}")
        agent.load(load_path)

    if args.eps_start is not None:
        agent.epsilon_start = args.eps_start
        agent.steps_done = 0
    if args.lr is not None:
        for pg in agent.optimizer.param_groups: pg["lr"] = args.lr

    episode_offset = 0 if args.fresh else find_episode_offset(args.save_dir)
    arbiter = MinimaxOpponent(depth=args.arbiter_depth) if args.arbiter else None
    
    run_training_stage(
        agent, opponent_name=args.opponent, episodes=args.episodes, save_dir=args.save_dir,
        checkpoint_every=args.checkpoint_every, metrics_every=args.metrics_every,
        n_envs=args.n_envs, arbiter=arbiter, arbiter_min_pieces=args.arbiter_min_pieces,
        seed=args.seed, stage_name=args.stage_name, cumulative_start=args.cumulative_start,
        official_hybrid_depth=args.official_hybrid_depth, freeze_conv=args.freeze_conv,
        run_metadata={"metrics_path": args.metrics_path, "chunk_size": args.chunk_size, "episode_offset": episode_offset, "phase_label": args.phase_label},
        self_play_frac=args.self_play_frac, eval_every=args.eval_every, eval_games=args.eval_games, eval_depth=args.eval_depth, fresh_logs=args.fresh
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
