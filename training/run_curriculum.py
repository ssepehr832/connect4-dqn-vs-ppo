"""Run a full DQN curriculum with artifact tracking and plots."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluate import load_agent, run_evaluation_suite
from opponents.minimax_opponent import MinimaxOpponent
from training.artifacts import ensure_dir, now_iso, read_json, sanitize_name, write_json
from training.pretrain import DATA_PATH, train as pretrain_train
from training.train_dqn import build_agent, run_training_stage, seed_everything


def read_csv_rows(path):
    """Read CSV rows into a list of dictionaries."""
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def bool_text(value):
    """Render booleans consistently in CSV-friendly extra metadata."""
    return "true" if value else "false"


def save_manifest(path, manifest):
    """Persist the run manifest."""
    manifest["updated_at"] = now_iso()
    write_json(path, manifest)


def stage_configs_from_args(args):
    """Build ordered stage configs from CLI args, following classmate's 6-stage plan."""
    if args.audit:
        return [
            {"name": "random", "opponent": "random", "episodes": 1000, "lr": 1e-4, "eps_start": 1.0, "eps_end": 0.5, "eps_decay": 50, "arbiter": False},
            {"name": "random+arbiter", "opponent": "random", "episodes": 1000, "lr": 1e-4, "eps_start": 0.5, "eps_end": 0.3, "eps_decay": 50, "arbiter": True},
            {"name": "heuristic", "opponent": "heuristic", "episodes": 1000, "lr": 1e-4, "eps_start": 0.3, "eps_end": 0.1, "eps_decay": 50, "arbiter": False},
            {"name": "minimax", "opponent": "minimax", "episodes": 1000, "lr": 1e-4, "eps_start": 0.15, "eps_end": 0.05, "eps_decay": 50, "arbiter": True},
            {"name": "self-mixed-0.6", "opponent": "self-mixed", "episodes": 1000, "lr": 3e-5, "eps_start": 0.1, "eps_end": 0.05, "eps_decay": 50, "self_play_frac": 0.6, "arbiter": True},
            {"name": "self-mixed-0.8", "opponent": "self-mixed", "episodes": 1000, "lr": 3e-5, "eps_start": 0.05, "eps_end": 0.05, "eps_decay": 50, "self_play_frac": 0.8, "arbiter": True},
        ]
    return [
        {
            "name": "random",
            "opponent": "random",
            "episodes": 50000,
            "lr": 1e-4,
            "eps_start": 1.0,
            "eps_end": 0.5,
            "eps_decay": 200,
            "arbiter": False,
        },
        {
            "name": "random+arbiter",
            "opponent": "random",
            "episodes": 50000,
            "lr": 1e-4,
            "eps_start": 0.5,
            "eps_end": 0.3,
            "eps_decay": 200,
            "arbiter": True,
        },
        {
            "name": "heuristic",
            "opponent": "heuristic",
            "episodes": 50000,
            "lr": 1e-4,
            "eps_start": 0.3,
            "eps_end": 0.1,
            "eps_decay": 200,
            "arbiter": False,
        },
        {
            "name": "minimax",
            "opponent": "minimax",
            "episodes": 1000000,
            "lr": 1e-4,
            "eps_start": 0.15,
            "eps_end": 0.05,
            "eps_decay": 300,
            "arbiter": True,
        },
        {
            "name": "self-mixed-0.6",
            "opponent": "self-mixed",
            "episodes": 750000,
            "lr": 3e-5,
            "eps_start": 0.1,
            "eps_end": 0.05,
            "eps_decay": 300,
            "self_play_frac": 0.6,
            "arbiter": True,
        },
        {
            "name": "self-mixed-0.8",
            "opponent": "self-mixed",
            "episodes": 1200000,
            "lr": 3e-5,
            "eps_start": 0.05,
            "eps_end": 0.05,
            "eps_decay": 200,
            "self_play_frac": 0.8,
            "arbiter": True,
        },
    ]


def build_base_eval_extra(run_name, checkpoint_path, checkpoint_label, stage_name, stage_episode, cumulative_episode, source, seed):
    """Create a consistent metadata payload for evaluation rows."""
    return {
        "run_name": run_name,
        "checkpoint_path": os.path.abspath(checkpoint_path) if checkpoint_path else "",
        "checkpoint_source": source,
        "stage_name": stage_name or "",
        "stage_episode": stage_episode if stage_episode is not None else "",
        "cumulative_episode": cumulative_episode if cumulative_episode is not None else "",
        "seed": seed,
        "stage_end": bool_text(source in {"stage_end", "final"}),
        "final_eval": bool_text(source == "final"),
        "checkpoint_alias": checkpoint_label or "",
    }


def suite_done(manifest, suite_id):
    """Check whether an evaluation suite already ran."""
    return suite_id in manifest["completed_evaluation_suites"]


def mark_suite_done(manifest, suite_id):
    """Record a completed evaluation suite."""
    manifest["completed_evaluation_suites"].append(suite_id)


def maybe_promote_best(run_dir, manifest, rows, checkpoint_path):
    """Promote the best checkpoint based on hybrid win rate vs minimax depth 6."""
    target_rows = [
        row for row in rows
        if row["agent"] == "dqn-hybrid"
        and row["opponent"] == "minimax"
        and str(row["minimax_depth"]) == "6"
    ]
    if not target_rows:
        return

    row = target_rows[0]
    metric = float(row["win_rate"])
    best_metric = manifest.get("best_metric")
    if best_metric is None or metric > best_metric:
        best_path = os.path.join(run_dir, "checkpoints", "best.pt")
        shutil.copy2(checkpoint_path, best_path)
        manifest["best_metric"] = metric
        manifest["best_checkpoint_path"] = best_path
        manifest["best_checkpoint_label"] = row.get("checkpoint_label", "")
        manifest["best_checkpoint_source"] = row.get("checkpoint_source", "")
        manifest["best_metric_updated_at"] = now_iso()


def run_quick_evals(args, run_dir, manifest, checkpoint_info, evaluation_csv):
    """Run quick checkpoint evaluations and update best-checkpoint tracking."""
    label = checkpoint_info["checkpoint_label"]
    base_extra = build_base_eval_extra(
        manifest["run_name"],
        checkpoint_info["checkpoint_path"],
        label,
        checkpoint_info["stage_name"],
        checkpoint_info["stage_episode"],
        checkpoint_info["cumulative_episode"],
        "checkpoint",
        args.seed,
    )
    base_extra["train_mode"] = checkpoint_info.get("train_mode", "")

    quick_rows = []
    suite_specs = [
        ("quick|dqn-hybrid|random-heuristic|" + label, "dqn-hybrid", ["random", "heuristic"], 4),
        ("quick|dqn-hybrid|minimax6|" + label, "dqn-hybrid", ["minimax"], 6),
    ]
    for suite_id, agent_name, opponents, depth in suite_specs:
        if suite_done(manifest, suite_id):
            continue
        rows = run_evaluation_suite(
            agent_name,
            opponents,
            games=args.quick_games,
            depth=depth,
            model_path=checkpoint_info["checkpoint_path"],
            hybrid_depth=args.official_hybrid_depth,
            save_csv=evaluation_csv,
            checkpoint_label=label,
            eval_tier="quick",
            extra=base_extra,
        )
        quick_rows.extend(rows)
        mark_suite_done(manifest, suite_id)

    maybe_promote_best(run_dir, manifest, quick_rows, checkpoint_info["checkpoint_path"])
    return quick_rows


def run_full_stage_evals(args, manifest, checkpoint_info, evaluation_csv, source_label):
    """Run stage-end or final evaluations for raw and hybrid agents."""
    label = checkpoint_info["checkpoint_label"]
    base_extra = build_base_eval_extra(
        manifest["run_name"],
        checkpoint_info["checkpoint_path"],
        label,
        checkpoint_info["stage_name"],
        checkpoint_info["stage_episode"],
        checkpoint_info["cumulative_episode"],
        source_label,
        args.seed,
    )
    rows = []
    suite_specs = [
        (f"{source_label}|dqn|nonminimax|{label}", "dqn", ["random", "heuristic", "self"], 4),
        (f"{source_label}|dqn|minimax4|{label}", "dqn", ["minimax"], 4),
        (f"{source_label}|dqn|minimax6|{label}", "dqn", ["minimax"], 6),
        (f"{source_label}|dqn-hybrid|nonminimax|{label}", "dqn-hybrid", ["random", "heuristic", "self"], 4),
        (f"{source_label}|dqn-hybrid|minimax4|{label}", "dqn-hybrid", ["minimax"], 4),
        (f"{source_label}|dqn-hybrid|minimax6|{label}", "dqn-hybrid", ["minimax"], 6),
    ]
    for suite_id, agent_name, opponents, depth in suite_specs:
        if suite_done(manifest, suite_id):
            continue
        suite_rows = run_evaluation_suite(
            agent_name,
            opponents,
            games=args.full_games,
            depth=depth,
            model_path=checkpoint_info["checkpoint_path"],
            hybrid_depth=args.official_hybrid_depth,
            save_csv=evaluation_csv,
            checkpoint_label=label,
            eval_tier="full",
            extra=base_extra,
        )
        rows.extend(suite_rows)
        mark_suite_done(manifest, suite_id)
    return rows


def run_depth_sweep(args, manifest, checkpoint_path, evaluation_csv):
    """Evaluate the best/final hybrid checkpoint against multiple minimax depths."""
    rows = []
    checkpoint_label = os.path.splitext(os.path.basename(checkpoint_path))[0]
    for depth in range(4, args.depth_sweep_max + 1):
        suite_id = f"depth-sweep|dqn-hybrid|minimax{depth}|{checkpoint_label}"
        if suite_done(manifest, suite_id):
            continue
        suite_rows = run_evaluation_suite(
            "dqn-hybrid",
            ["minimax"],
            games=args.depth_sweep_games,
            depth=depth,
            model_path=checkpoint_path,
            hybrid_depth=args.official_hybrid_depth,
            save_csv=evaluation_csv,
            checkpoint_label=checkpoint_label,
            eval_tier="depth-sweep",
            extra=build_base_eval_extra(
                manifest["run_name"],
                checkpoint_path,
                checkpoint_label,
                "final",
                "",
                "",
                "depth-sweep",
                args.seed,
            ),
        )
        rows.extend(suite_rows)
        mark_suite_done(manifest, suite_id)
    return rows


def run_baseline_eval(args, manifest, evaluation_csv):
    """Evaluate the existing teammate checkpoint without modifying it."""
    baseline_path = os.path.join("models", "dqn", "latest.pt")
    if not os.path.exists(baseline_path):
        manifest["baseline_evaluated"] = False
        return
    if manifest.get("baseline_evaluated"):
        return

    extra = build_base_eval_extra(
        manifest["run_name"],
        baseline_path,
        "existing_latest",
        "baseline",
        "",
        "",
        "baseline",
        args.seed,
    )
    run_evaluation_suite(
        "dqn",
        ["random", "heuristic", "self"],
        games=args.full_games,
        depth=4,
        model_path=baseline_path,
        hybrid_depth=args.official_hybrid_depth,
        save_csv=evaluation_csv,
        checkpoint_label="existing_latest",
        eval_tier="baseline",
        extra=extra,
    )
    run_evaluation_suite(
        "dqn",
        ["minimax"],
        games=args.full_games,
        depth=6,
        model_path=baseline_path,
        hybrid_depth=args.official_hybrid_depth,
        save_csv=evaluation_csv,
        checkpoint_label="existing_latest",
        eval_tier="baseline",
        extra=extra,
    )
    run_evaluation_suite(
        "dqn-hybrid",
        ["random", "heuristic", "self"],
        games=args.full_games,
        depth=4,
        model_path=baseline_path,
        hybrid_depth=args.official_hybrid_depth,
        save_csv=evaluation_csv,
        checkpoint_label="existing_latest",
        eval_tier="baseline",
        extra=extra,
    )
    run_evaluation_suite(
        "dqn-hybrid",
        ["minimax"],
        games=args.full_games,
        depth=6,
        model_path=baseline_path,
        hybrid_depth=args.official_hybrid_depth,
        save_csv=evaluation_csv,
        checkpoint_label="existing_latest",
        eval_tier="baseline",
        extra=extra,
    )
    manifest["baseline_evaluated"] = True


def add_stage_boundaries(ax, manifest):
    """Annotate plot stage boundaries."""
    for stage_name, stage_state in manifest["stages"].items():
        if not stage_state.get("completed"):
            continue
        boundary = stage_state.get("cumulative_episode", 0)
        if not boundary:
            continue
        ax.axvline(boundary, color="#cccccc", linestyle="--", linewidth=1)
        ax.text(boundary, ax.get_ylim()[1], stage_name, rotation=90, va="bottom", ha="right", fontsize=8)


def plot_pretrain_metrics(run_dir):
    """Generate a figure for pretraining loss/accuracy."""
    import matplotlib.pyplot as plt
    metrics_path = os.path.join(run_dir, "pretrain_metrics.csv")
    if not os.path.exists(metrics_path):
        return None
    rows = read_csv_rows(metrics_path)
    if not rows:
        return None

    epochs = [int(row["epoch"]) for row in rows]
    loss = [float(row["train_loss"]) for row in rows]
    train_acc = [float(row["train_acc"]) for row in rows]
    val_acc = [float(row["val_acc"]) for row in rows]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color="tab:red")
    ax1.plot(epochs, loss, color="tab:red", label="Train Loss")
    ax1.tick_params(axis='y', labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color="tab:blue")
    ax2.plot(epochs, train_acc, color="tab:blue", linestyle="--", alpha=0.6, label="Train Acc")
    ax2.plot(epochs, val_acc, color="tab:blue", label="Val Acc")
    ax2.tick_params(axis='y', labelcolor="tab:blue")

    plt.title("Supervised Pretraining Metrics")
    fig.tight_layout()
    out_path = os.path.join(run_dir, "plots", "pretrain_curves.png")
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_training_metrics(run_dir, manifest):
    """Generate the main training-telemetry figure."""
    import matplotlib.pyplot as plt

    metrics_path = os.path.join(run_dir, "training_metrics.csv")
    rows = read_csv_rows(metrics_path)
    if not rows:
        return None

    episodes = [int(float(row["cumulative_episode"])) for row in rows]
    win_rate = [float(row["win_rate"]) for row in rows]
    draw_rate = [float(row["draw_rate"]) for row in rows]
    loss_rate = [float(row["loss_rate"]) for row in rows]
    reward_mean = [float(row["reward_mean"]) for row in rows]
    td_loss_mean = [float(row["td_loss_mean"]) for row in rows]
    epsilon = [float(row["epsilon"]) for row in rows]
    unique_ratio = [float(row["unique_game_ratio"]) for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.ravel()

    axes[0].plot(episodes, win_rate, label="Win rate", color="#2ecc71")
    axes[0].plot(episodes, draw_rate, label="Draw rate", color="#95a5a6")
    axes[0].plot(episodes, loss_rate, label="Loss rate", color="#e74c3c")
    axes[0].set_title("Rolling Outcomes")
    axes[0].set_ylabel("Rate")
    axes[0].legend()

    axes[1].plot(episodes, reward_mean, label="Reward mean", color="#3498db")
    axes[1].plot(episodes, td_loss_mean, label="TD loss mean", color="#9b59b6")
    axes[1].set_title("Reward and TD Loss")
    axes[1].legend()

    axes[2].plot(episodes, epsilon, color="#f39c12")
    axes[2].set_title("Epsilon Schedule")
    axes[2].set_xlabel("Cumulative episodes")
    axes[2].set_ylabel("Epsilon")

    axes[3].plot(episodes, unique_ratio, color="#1abc9c")
    axes[3].set_title("Unique-Game Ratio")
    axes[3].set_xlabel("Cumulative episodes")
    axes[3].set_ylabel("Ratio")

    for ax in axes:
        ax.grid(alpha=0.25)
        add_stage_boundaries(ax, manifest)

    fig.tight_layout()
    out_path = os.path.join(run_dir, "plots", "training_curves.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_quick_minimax_curve(run_dir):
    """Plot quick-eval hybrid performance vs minimax depth 6 over time."""
    import matplotlib.pyplot as plt

    rows = [
        row for row in read_csv_rows(os.path.join(run_dir, "evaluation_summary.csv"))
        if row.get("eval_tier") == "quick"
        and row.get("agent") == "dqn-hybrid"
        and row.get("opponent") == "minimax"
        and str(row.get("minimax_depth")) == "6"
    ]
    if not rows:
        return None

    rows.sort(key=lambda row: int(float(row.get("cumulative_episode") or 0)))
    x = [int(float(row.get("cumulative_episode") or 0)) for row in rows]
    y = [float(row["win_rate"]) for row in rows]
    labels = [row.get("checkpoint_label", "") for row in rows]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, y, marker="o", color="#2ecc71")
    for xi, yi, label in zip(x, y, labels, strict=False):
        ax.text(xi, yi, label, fontsize=7, rotation=30, ha="left", va="bottom")
    ax.set_title("Quick Eval: Hybrid@4 vs Minimax@6")
    ax.set_xlabel("Cumulative episodes")
    ax.set_ylabel("Win rate")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path = os.path.join(run_dir, "plots", "quick_eval_hybrid_vs_minimax6.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def stage_opponent_label(row):
    """Format opponent labels for grouped stage plots."""
    if row["opponent"] != "minimax":
        return row["opponent"]
    return f"minimax d{row['minimax_depth']}"


def plot_stage_end_comparison(run_dir, manifest):
    """Plot stage-end hybrid performance across opponents."""
    import matplotlib.pyplot as plt

    rows = [
        row for row in read_csv_rows(os.path.join(run_dir, "evaluation_summary.csv"))
        if row.get("eval_tier") == "full"
        and row.get("agent") == "dqn-hybrid"
        and row.get("stage_end") == "true"
    ]
    if not rows:
        return None

    rows.sort(key=lambda row: (row.get("stage_name", ""), row.get("opponent", ""), row.get("minimax_depth", "")))
    stages = [cfg for cfg in manifest["stage_order"] if manifest["stages"].get(cfg, {}).get("completed")]
    opponent_labels = []
    for row in rows:
        label = stage_opponent_label(row)
        if label not in opponent_labels:
            opponent_labels.append(label)

    width = min(0.8 / max(len(stages), 1), 0.22)
    positions = list(range(len(opponent_labels)))
    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, stage_name in enumerate(stages):
        stage_rows = {stage_opponent_label(row): float(row["win_rate"]) for row in rows if row.get("stage_name") == stage_name}
        values = [stage_rows.get(label, 0.0) for label in opponent_labels]
        offsets = [pos + (idx - (len(stages) - 1) / 2) * width for pos in positions]
        ax.bar(offsets, values, width=width, label=stage_name)

    ax.set_xticks(positions)
    ax.set_xticklabels(opponent_labels)
    ax.set_ylabel("Win rate")
    ax.set_title("Stage-End Hybrid Performance")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path = os.path.join(run_dir, "plots", "stage_end_hybrid_comparison.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_depth_sweep(run_dir):
    """Plot final hybrid performance against deeper minimax opponents."""
    import matplotlib.pyplot as plt

    rows = [
        row for row in read_csv_rows(os.path.join(run_dir, "evaluation_summary.csv"))
        if row.get("eval_tier") == "depth-sweep"
        and row.get("agent") == "dqn-hybrid"
        and row.get("opponent") == "minimax"
    ]
    if not rows:
        return None

    rows.sort(key=lambda row: int(row["minimax_depth"]))
    depths = [int(row["minimax_depth"]) for row in rows]
    win_rates = [float(row["win_rate"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(depths, win_rates, marker="o", color="#34495e")
    ax.set_title("Final Hybrid@4 Depth Sweep")
    ax.set_xlabel("Opponent minimax depth")
    ax.set_ylabel("Win rate")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path = os.path.join(run_dir, "plots", "final_depth_sweep.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_qualitative_plot(args, run_dir, checkpoint_path):
    """Save one qualitative gameplay figure for the report appendix/slides."""
    import matplotlib.pyplot as plt

    from evaluation.visualize import visualize_opponent

    if not checkpoint_path:
        return None
    agent = load_agent("dqn-hybrid", model_path=checkpoint_path, hybrid_depth=args.official_hybrid_depth)
    fig = visualize_opponent(agent, "dqn-hybrid", "minimax", min(args.quick_games, 60), 6)
    out_path = os.path.join(run_dir, "plots", "qualitative_hybrid_vs_minimax6.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_run_summary(run_dir, manifest):
    """Write a concise Markdown summary of the run outputs."""
    rows = read_csv_rows(os.path.join(run_dir, "evaluation_summary.csv"))
    final_rows = [
        row for row in rows
        if row.get("eval_tier") == "full"
        and row.get("final_eval") == "true"
        and row.get("agent") == "dqn-hybrid"
    ]
    summary_path = os.path.join(run_dir, "run_summary.md")
    lines = [
        f"# DQN Curriculum Run: {manifest['run_name']}",
        "",
        f"- Created: {manifest['created_at']}",
        f"- Updated: {manifest['updated_at']}",
        f"- Best checkpoint: `{manifest.get('best_checkpoint_path', '')}`",
        f"- Best hybrid@4 vs minimax@6 win rate: {manifest.get('best_metric', 'n/a')}",
        "",
        "## Completed Stages",
    ]
    for stage_name in manifest["stage_order"]:
        state = manifest["stages"][stage_name]
        lines.append(
            f"- `{stage_name}`: completed={state.get('completed', False)} "
            f"episodes={state.get('stage_episode', 0)}/{state.get('episodes_target', 0)} "
            f"checkpoint=`{state.get('latest_checkpoint_path', '')}`"
        )

    if final_rows:
        lines.extend(["", "## Final Hybrid Evaluation", "", "| Opponent | Win | Draw | Loss |", "|---|---:|---:|---:|"])
        for row in sorted(final_rows, key=lambda item: (item["opponent"], item.get("minimax_depth", ""))):
            opp = stage_opponent_label(row)
            lines.append(
                f"| {opp} | {float(row['win_rate']):.3f} | {float(row['draw_rate']):.3f} | {float(row['loss_rate']):.3f} |"
            )

    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return summary_path


def build_parser():
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(description="Run the full DQN curriculum with artifacts")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parser.add_argument("--run-name", type=str, default=f"dqn-curriculum-{timestamp}")
    parser.add_argument("--resume", action="store_true", help="Resume an existing run directory")
    parser.add_argument("--audit", action="store_true", help="Run a tiny curriculum for pipeline verification")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=1024)
    parser.add_argument("--official-hybrid-depth", type=int, default=4)
    parser.add_argument("--quick-games", type=int, default=48)
    parser.add_argument("--full-games", type=int, default=200)
    parser.add_argument("--depth-sweep-games", type=int, default=120)
    parser.add_argument("--depth-sweep-max", type=int, default=7)
    parser.add_argument("--checkpoint-every", type=int, default=50000)
    parser.add_argument("--metrics-every", type=int, default=1000)
    parser.add_argument("--self-play-update", type=int, default=3000)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--arbiter-depth", type=int, default=4)
    parser.add_argument("--arbiter-min-pieces", type=int, default=12)
    parser.add_argument("--no-arbiter", action="store_true", help="Disable the minimax arbiter globally")
    parser.add_argument("--unfreeze-conv", action="store_true", help="Do not freeze conv layers during RL")
    parser.add_argument("--publish-latest", action="store_true",
                        help="Copy the best checkpoint to models/dqn/latest.pt at the end")
    parser.add_argument("--data-path", type=str, default=DATA_PATH)
    parser.add_argument("--artifacts-root", type=str, default="artifacts/runs/dqn")
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--pretrain-batch-size", type=int, default=256)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--snapshot-every", type=int, default=1000000000,
                        help="Frequency of duplicate snapshots (disabled by default in curriculum)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.audit:
        args.quick_games = 16
        args.full_games = 32
        args.depth_sweep_games = 16
        args.checkpoint_every = 500
        args.metrics_every = 100

    run_name = sanitize_name(args.run_name)
    run_dir = os.path.join(args.artifacts_root, run_name)
    if os.path.exists(run_dir) and not args.resume:
        parser.error(f"Run directory already exists: {run_dir}. Use --resume or a new --run-name.")

    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "checkpoints"))
    ensure_dir(os.path.join(run_dir, "plots"))

    config_path = os.path.join(run_dir, "config.json")
    manifest_path = os.path.join(run_dir, "manifest.json")
    pretrain_metrics_csv = os.path.join(run_dir, "pretrain_metrics.csv")
    training_metrics_csv = os.path.join(run_dir, "training_metrics.csv")
    evaluation_csv = os.path.join(run_dir, "evaluation_summary.csv")
    pretrain_summary_json = os.path.join(run_dir, "pretrain_summary.json")
    pretrain_checkpoint_path = os.path.join(run_dir, "checkpoints", "pretrain_init.pt")

    stage_order = [cfg["name"] for cfg in stage_configs_from_args(args)]
    config_payload = {
        "run_name": run_name,
        "seed": args.seed,
        "n_envs": args.n_envs,
        "official_hybrid_depth": args.official_hybrid_depth,
        "checkpoint_every": args.checkpoint_every,
        "metrics_every": args.metrics_every,
        "quick_games": args.quick_games,
        "full_games": args.full_games,
        "depth_sweep_games": args.depth_sweep_games,
        "depth_sweep_max": args.depth_sweep_max,
        "freeze_conv": not args.unfreeze_conv,
        "arbiter_depth": args.arbiter_depth,
        "arbiter_min_pieces": args.arbiter_min_pieces,
        "self_play_update": args.self_play_update,
        "chunk_size": args.chunk_size,
        "pretrain_epochs": args.pretrain_epochs,
        "pretrain_batch_size": args.pretrain_batch_size,
        "pretrain_lr": args.pretrain_lr,
        "stages": stage_configs_from_args(args),
        "data_path": os.path.abspath(args.data_path),
    }
    write_json(config_path, config_payload)

    manifest = read_json(manifest_path, default=None)
    if manifest is None:
        manifest = {
            "run_name": run_name,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "status": "in_progress",
            "stage_order": stage_order,
            "baseline_evaluated": False,
            "pretrain_completed": False,
            "pretrain_checkpoint_path": "",
            "best_metric": None,
            "best_checkpoint_path": "",
            "best_checkpoint_label": "",
            "best_checkpoint_source": "",
            "completed_evaluation_suites": [],
            "stages": {
                stage_name: {
                    "completed": False,
                    "episodes_target": next(cfg["episodes"] for cfg in config_payload["stages"] if cfg["name"] == stage_name),
                    "stage_episode": 0,
                    "cumulative_episode": 0,
                    "latest_checkpoint_path": "",
                    "latest_checkpoint_label": "",
                }
                for stage_name in stage_order
            },
        }
        save_manifest(manifest_path, manifest)

    try:
        seed_everything(args.seed)

        run_baseline_eval(args, manifest, evaluation_csv)
        save_manifest(manifest_path, manifest)

        if not manifest.get("pretrain_completed"):
            print("\n=== Pretraining conv backbone ===")
            pretrain_epochs = 2 if args.audit else args.pretrain_epochs
            pretrain_train(
                "dqn",
                epochs=pretrain_epochs,
                batch_size=args.pretrain_batch_size,
                lr=args.pretrain_lr,
                data_path=args.data_path,
                metrics_path=pretrain_metrics_csv,
                summary_path=pretrain_summary_json,
                save_path=pretrain_checkpoint_path,
            )
            manifest["pretrain_completed"] = True
            manifest["pretrain_checkpoint_path"] = pretrain_checkpoint_path
            save_manifest(manifest_path, manifest)

        if manifest.get("pretrain_completed"):
            print("\n=== Evaluating pretrained model ===")
            pretrain_info = {
                "checkpoint_path": pretrain_checkpoint_path,
                "checkpoint_label": "pretrain_init",
                "stage_name": "pretrain",
                "stage_episode": args.pretrain_epochs,
                "cumulative_episode": 0,
            }
            run_full_stage_evals(args, manifest, pretrain_info, evaluation_csv, "pretrain")
            save_manifest(manifest_path, manifest)

        current_checkpoint_path = manifest.get("pretrain_checkpoint_path", pretrain_checkpoint_path)
        cumulative_completed = 0
        freeze_conv = not args.unfreeze_conv

        for stage_cfg in stage_configs_from_args(args):
            stage_name = stage_cfg["name"]
            stage_state = manifest["stages"][stage_name]

            if stage_state.get("completed"):
                current_checkpoint_path = stage_state["latest_checkpoint_path"] or current_checkpoint_path
                cumulative_completed = int(stage_state.get("cumulative_episode", cumulative_completed))
                continue

            load_path = stage_state.get("latest_checkpoint_path") or current_checkpoint_path
            stage_episode_start = int(stage_state.get("stage_episode", 0))
            cumulative_start = (
                int(stage_state.get("cumulative_episode", 0)) - stage_episode_start
                if stage_episode_start > 0 else cumulative_completed
            )

            agent = build_agent(n_envs=args.n_envs)
            if freeze_conv:
                agent.set_freeze_conv(True)
            if load_path and os.path.exists(load_path):
                print(f"\n=== Loading stage checkpoint for {stage_name}: {load_path} ===")
                agent.load(load_path)
            
            # Apply stage-specific hyperparams
            for group in agent.optimizer.param_groups:
                group["lr"] = stage_cfg["lr"]
            agent.epsilon_start = stage_cfg["eps_start"]
            agent.epsilon_end = stage_cfg["eps_end"]
            agent.epsilon_decay_steps = stage_cfg["eps_decay"]
            agent.steps_done = 0 # restart decay each stage

            # Setup stage-specific arbiter
            stage_arbiter = None
            if stage_cfg.get("arbiter") and not args.no_arbiter:
                stage_arbiter = MinimaxOpponent(depth=args.arbiter_depth)

            def on_checkpoint(checkpoint_info, *, stage_name=stage_name):
                stage_state["latest_checkpoint_path"] = checkpoint_info["checkpoint_path"]
                stage_state["latest_checkpoint_label"] = checkpoint_info["checkpoint_label"]
                stage_state["stage_episode"] = checkpoint_info["stage_episode"]
                stage_state["cumulative_episode"] = checkpoint_info["cumulative_episode"]
                save_manifest(manifest_path, manifest)

                print(f"\nCheckpoint saved: {checkpoint_info['checkpoint_label']}")
                run_quick_evals(args, run_dir, manifest, checkpoint_info, evaluation_csv)
                if checkpoint_info.get("final"):
                    stage_state["completed"] = True
                    run_full_stage_evals(args, manifest, checkpoint_info, evaluation_csv, "final" if stage_name == stage_order[-1] else "stage_end")
                save_manifest(manifest_path, manifest)

            result = run_training_stage(
                agent,
                opponent_name=stage_cfg["opponent"],
                episodes=stage_cfg["episodes"],
                save_dir=run_dir,
                checkpoint_every=args.checkpoint_every,
                metrics_every=args.metrics_every,
                self_play_update=args.self_play_update,
                n_envs=args.n_envs,
                arbiter=stage_arbiter,
                arbiter_min_pieces=args.arbiter_min_pieces,
                seed=args.seed,
                stage_name=stage_name,
                stage_episode_start=stage_episode_start,
                cumulative_start=cumulative_start,
                official_hybrid_depth=args.official_hybrid_depth,
                freeze_conv=freeze_conv,
                run_metadata={
                    "run_name": run_name,
                    "metrics_path": training_metrics_csv,
                    "chunk_size": stage_cfg.get("chunk_size", args.chunk_size),
                    "stage_target_episodes": stage_cfg["episodes"],
                    "episode_offset": cumulative_start,
                    "phase_label": stage_name,
                },
                on_checkpoint=on_checkpoint,
                self_play_frac=stage_cfg.get("self_play_frac", 0.5),
                snapshot_every=args.snapshot_every,
                fresh_logs=True if (not args.resume and stage_name == stage_order[0] and stage_episode_start == 0) else False,
            )

            current_checkpoint_path = result["latest_checkpoint_path"]
            cumulative_completed = result["cumulative_episode"]
            stage_state["completed"] = True
            stage_state["stage_episode"] = result["episodes_completed"]
            stage_state["cumulative_episode"] = result["cumulative_episode"]
            stage_state["latest_checkpoint_path"] = result["latest_checkpoint_path"]
            stage_state["latest_checkpoint_label"] = result["latest_checkpoint_label"]
            save_manifest(manifest_path, manifest)

        final_checkpoint = manifest.get("best_checkpoint_path") or current_checkpoint_path
        if final_checkpoint:
            run_depth_sweep(args, manifest, final_checkpoint, evaluation_csv)
            save_manifest(manifest_path, manifest)

            try:
                plot_pretrain_metrics(run_dir)
                plot_training_metrics(run_dir, manifest)
                plot_quick_minimax_curve(run_dir)
                plot_stage_end_comparison(run_dir, manifest)
                plot_depth_sweep(run_dir)
                save_qualitative_plot(args, run_dir, final_checkpoint)
                manifest["plots_status"] = "completed"
            except ImportError as exc:
                manifest["plots_status"] = f"skipped: {exc}"
            write_run_summary(run_dir, manifest)

        if args.publish_latest and final_checkpoint:
            ensure_dir(os.path.join("models", "dqn"))
            shutil.copy2(final_checkpoint, os.path.join("models", "dqn", "latest.pt"))
            manifest["published_latest_path"] = os.path.abspath(os.path.join("models", "dqn", "latest.pt"))

        manifest["status"] = "completed"
        save_manifest(manifest_path, manifest)
        print(f"\nRun complete. Artifacts saved under {run_dir}")

    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        save_manifest(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
