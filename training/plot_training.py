"""Plot training curves from metrics.csv and eval.csv.

Produces PNGs in <models-dir>/plots/:
  - eval_win_rate.png        : win rate vs random/heuristic/minimax over episodes
  - training_win_rate.png    : rolling training win rate + epsilon on twin axis
                               (self-mixed phases plot two lines: self-play and minimax chunks)
  - exploration.png          : epsilon decay + unique-games %

All plots show phase bands in the background labeled with the phase name
(random / heuristic / minimax / self / self-mixed) so multi-stage curriculum
training is readable in one graph.

Usage:
    python -m training.plot_training
    python -m training.plot_training --models-dir models/dqn_rerun
"""

import argparse
import csv
import os
import sys


# Background colors for each training phase. Anything not in this map falls
# back to a neutral grey. Feel free to extend with your own --phase-label
# conventions (e.g. 'random+arbiter', 'minimax-d6').
PHASE_COLORS = {
    "random":           "#cce5ff",  # light blue
    "random+arbiter":   "#9ec5e9",  # slightly darker blue
    "heuristic":        "#d4edda",  # light green
    "minimax":          "#fff3cd",  # light yellow
    "self":             "#f8d7da",  # light red
    "self-mixed":       "#e2d9f3",  # light purple
}
_DEFAULT_BAND_COLOR = "#ececec"


def _read_csv(path):
    """Return (rows, columns). rows = list of dicts, columns = list of column names."""
    if not os.path.exists(path):
        return None, None
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return None, None
    cols = list(rows[0].keys())
    return rows, cols


def _col(rows, name, as_float=True):
    """Extract a column as a list. Empty strings become None."""
    out = []
    for row in rows:
        v = row.get(name, "")
        if v == "" or v is None:
            out.append(None)
        else:
            if as_float:
                try:
                    out.append(float(v))
                except ValueError:
                    out.append(None)
            else:
                out.append(v)
    return out


def _filter_non_none(xs, ys):
    out_x, out_y = [], []
    for x, y in zip(xs, ys):
        if y is not None and x is not None:
            out_x.append(x)
            out_y.append(y)
    return out_x, out_y


def _phase_spans(episodes, phases):
    """Return a list of (phase_name, start_ep, end_ep) tuples from parallel lists."""
    spans = []
    if not episodes or not phases:
        return spans
    cur = phases[0]
    start = episodes[0]
    prev = episodes[0]
    for ep, ph in zip(episodes[1:], phases[1:]):
        if ph != cur:
            spans.append((cur, start, prev))
            cur = ph
            start = ep
        prev = ep
    spans.append((cur, start, prev))
    return spans


def _draw_phase_bands(ax, spans, y_label_frac=0.96):
    """Draw colored background bands with phase labels at the top."""
    if not spans:
        return
    y_lo, y_hi = ax.get_ylim()
    y_text = y_lo + (y_hi - y_lo) * y_label_frac
    for phase, start, end in spans:
        if end <= start:
            end = start + 1
        color = PHASE_COLORS.get(phase, _DEFAULT_BAND_COLOR)
        ax.axvspan(start, end, color=color, alpha=0.35, zorder=0)
        mid = (start + end) / 2
        ax.text(
            mid, y_text, phase,
            ha="center", va="top",
            fontsize=9, color="#333333",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none", alpha=0.7),
            zorder=5,
        )


def plot_eval(rows, out_path):
    import matplotlib.pyplot as plt

    episodes = _col(rows, "episode")
    phases = _col(rows, "phase", as_float=False)
    vs_random = _col(rows, "vs_random")
    vs_heuristic = _col(rows, "vs_heuristic")
    vs_minimax = _col(rows, "vs_minimax")

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Lines (drop rows where these are missing)
    xs, ys = _filter_non_none(episodes, vs_random)
    ax.plot(xs, ys, label="vs random", marker="o", linewidth=2, color="#2ca02c", zorder=3)
    xs, ys = _filter_non_none(episodes, vs_heuristic)
    ax.plot(xs, ys, label="vs heuristic", marker="s", linewidth=2, color="#1f77b4", zorder=3)
    xs, ys = _filter_non_none(episodes, vs_minimax)
    ax.plot(xs, ys, label="vs minimax", marker="^", linewidth=2, color="#d62728", zorder=3)

    ax.set_xlabel("Training episodes (cumulative)")
    ax.set_ylabel("Win rate")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Evaluation win rate vs fixed opponents")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    _draw_phase_bands(ax, _phase_spans(episodes, phases))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_training_win_rate(rows, out_path):
    import matplotlib.pyplot as plt

    episodes = _col(rows, "episode")
    phases = _col(rows, "phase", as_float=False)
    epsilon = _col(rows, "epsilon")
    win_rate = _col(rows, "win_rate")
    win_rate_self = _col(rows, "win_rate_self")
    win_rate_minimax = _col(rows, "win_rate_minimax")

    fig, ax1 = plt.subplots(figsize=(9.5, 5.5))

    any_plotted = False

    xs, ys = _filter_non_none(episodes, win_rate)
    if xs:
        ax1.plot(xs, ys, label="win rate (training)", color="#1f77b4",
                 linewidth=1.8, zorder=3)
        any_plotted = True

    xs, ys = _filter_non_none(episodes, win_rate_self)
    if xs:
        ax1.plot(xs, ys, label="self-play chunks", color="#9467bd",
                 linewidth=1.8, alpha=0.9, zorder=3)
        any_plotted = True

    xs, ys = _filter_non_none(episodes, win_rate_minimax)
    if xs:
        ax1.plot(xs, ys, label="minimax chunks", color="#d62728",
                 linewidth=1.8, alpha=0.9, zorder=3)
        any_plotted = True

    if not any_plotted:
        print("  (no win-rate columns in metrics.csv — skipping)")
        plt.close(fig)
        return

    ax1.set_xlabel("Training episodes (cumulative)")
    ax1.set_ylabel("Rolling win rate (500-ep window)")
    ax1.set_ylim(-0.02, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Training win rate over time")

    # Epsilon on twin axis
    ax2 = ax1.twinx()
    xs, ys = _filter_non_none(episodes, epsilon)
    ax2.plot(xs, ys, label="ε", color="#666666", linestyle="--",
             linewidth=1.4, zorder=2)
    ax2.set_ylabel("ε (exploration)", color="#555555")
    ax2.set_ylim(-0.02, 1.05)
    ax2.tick_params(axis="y", colors="#555555")

    # Phase bands (on ax1 because that's what the text-y coordinates track)
    _draw_phase_bands(ax1, _phase_spans(episodes, phases))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_exploration(rows, out_path):
    import matplotlib.pyplot as plt

    episodes = _col(rows, "episode")
    phases = _col(rows, "phase", as_float=False)
    epsilon = _col(rows, "epsilon")
    unique_pct = _col(rows, "unique_pct")

    fig, ax1 = plt.subplots(figsize=(9.5, 5.5))

    xs, ys = _filter_non_none(episodes, epsilon)
    ax1.plot(xs, ys, color="#1f77b4", label="ε (exploration rate)",
             linewidth=2, zorder=3)
    ax1.set_xlabel("Training episodes (cumulative)")
    ax1.set_ylabel("ε", color="#1f77b4")
    ax1.set_ylim(-0.02, 1.05)
    ax1.tick_params(axis="y", colors="#1f77b4")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    xs, ys = _filter_non_none(episodes, unique_pct)
    ax2.plot(xs, ys, color="#2ca02c", label="unique games %",
             linewidth=2, zorder=3)
    ax2.set_ylabel("Unique games %", color="#2ca02c")
    ax2.set_ylim(-0.02, 1.05)
    ax2.tick_params(axis="y", colors="#2ca02c")

    ax1.set_title("Exploration: ε decay and game diversity")

    _draw_phase_bands(ax1, _phase_spans(episodes, phases))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot training curves")
    parser.add_argument("--models-dir", default="models/dqn",
                        help="Directory with metrics.csv and eval.csv (default: models/dqn)")
    parser.add_argument("--out-dir", default=None,
                        help="Where to save PNGs (default: <models-dir>/plots)")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.models_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(args.models_dir, "metrics.csv")
    eval_path = os.path.join(args.models_dir, "eval.csv")

    metrics_rows, _ = _read_csv(metrics_path)
    eval_rows, _ = _read_csv(eval_path)

    if metrics_rows is None and eval_rows is None:
        print(f"No metrics.csv or eval.csv found in {args.models_dir}")
        sys.exit(1)

    print(f"Plotting from {args.models_dir}")

    if metrics_rows is not None:
        plot_training_win_rate(metrics_rows, os.path.join(out_dir, "training_win_rate.png"))
        plot_exploration(metrics_rows, os.path.join(out_dir, "exploration.png"))
    else:
        print("  (no metrics.csv — skipping training/exploration plots)")

    if eval_rows is not None:
        plot_eval(eval_rows, os.path.join(out_dir, "eval_win_rate.png"))
    else:
        print("  (no eval.csv — skipping evaluation plot)")

    print("Done.")


if __name__ == "__main__":
    main()
