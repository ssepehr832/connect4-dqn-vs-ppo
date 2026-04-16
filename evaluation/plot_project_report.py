"""Plot figures from project_report_evals.csv — one PNG per figure (no subplot grids).

Usage:
    python -m evaluation.plot_project_report
    python -m evaluation.plot_project_report --csv path/to/project_report_evals.csv \\
        --out-dir path/to/plots
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any

sys.path.insert(0, os.path.abspath("."))

from evaluation.report_evaluator import CHECKPOINTS


def _default_csv_path() -> str:
    from evaluation import report_evaluator

    return report_evaluator.CSV_OUT


def _checkpoint_order() -> list[str]:
    return [c.rsplit(".", 1)[0] for c in CHECKPOINTS]


def load_rows(csv_path: str) -> list[dict[str, Any]]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str) -> float:
    v = row.get(key, "")
    if v == "" or v is None:
        return float("nan")
    return float(v)


def _depth(row: dict) -> int | None:
    if row.get("opponent") != "minimax":
        return None
    d = row.get("minimax_depth", "")
    if d == "" or d is None:
        return None
    return int(d)


def index_rows(rows: list[dict]) -> dict[tuple, dict]:
    """Key: (ckpt_label, opponent, depth_or_None)."""
    out: dict[tuple, dict] = {}
    for r in rows:
        label = r.get("_ckpt_label") or r.get("checkpoint_label") or ""
        opp = r.get("opponent", "")
        d = _depth(r)
        out[(label, opp, d)] = r
    return out


def ordered_labels(rows: list[dict], preferred: list[str]) -> list[str]:
    seen = set()
    labels: list[str] = []
    for p in preferred:
        if any((r.get("_ckpt_label") or r.get("checkpoint_label")) == p for r in rows):
            labels.append(p)
            seen.add(p)
    for r in rows:
        lab = r.get("_ckpt_label") or r.get("checkpoint_label") or ""
        if lab and lab not in seen:
            labels.append(lab)
            seen.add(lab)
    return labels


def _save_fig(path: str) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_minimax_depth_sweep(rows: list[dict], labels: list[str], out_path: str) -> None:
    import matplotlib.pyplot as plt

    idx = index_rows(rows)
    depths_avail = sorted(
        {d for (_, o, d) in idx if o == "minimax" and d is not None}
    )
    if not depths_avail:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    cmap = plt.get_cmap("tab10")
    for i, lab in enumerate(labels):
        ys = []
        xs = []
        for d in depths_avail:
            r = idx.get((lab, "minimax", d))
            if r is None:
                continue
            xs.append(d)
            ys.append(_f(r, "win_rate"))
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2,
            label=lab,
            color=cmap(i % 10),
        )

    ax.set_xlabel("Minimax search depth")
    ax.set_ylabel("Win rate")
    ax.set_title("Win rate vs minimax depth by checkpoint")
    ax.set_xticks(depths_avail)
    ax.set_ylim(0, 1.05)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    _save_fig(out_path)


def plot_opponents_grouped_by_checkpoint(
    rows: list[dict],
    labels: list[str],
    out_path: str,
    *,
    minimax_depth: int = 6,
) -> None:
    """One grouped bar chart: each checkpoint has bars for random, heuristic, self, minimax@depth."""
    import matplotlib.pyplot as plt
    import numpy as np

    idx = index_rows(rows)
    series_spec: list[tuple[tuple[str, int | None], str]] = [
        (("random", None), "vs random"),
        (("heuristic", None), "vs heuristic"),
        (("self", None), "vs self"),
        (("minimax", minimax_depth), f"vs minimax d{minimax_depth}"),
    ]

    fig, ax = plt.subplots(figsize=(13, 5.8))
    x = np.arange(len(labels))
    n = len(series_spec)
    width = min(0.8 / n, 0.2)
    cmap = plt.get_cmap("tab10")

    for j, ((opp, depth), leg) in enumerate(series_spec):
        offsets = (j - (n - 1) / 2) * width
        heights = []
        for lab in labels:
            key = (lab, opp, depth) if opp == "minimax" else (lab, opp, None)
            r = idx.get(key)
            heights.append(_f(r, "win_rate") if r else float("nan"))
        ax.bar(
            x + offsets,
            heights,
            width=width * 0.92,
            label=leg,
            color=cmap(j % 10),
            edgecolor="#333333",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Win rate")
    ax.set_title("Win rate by opponent (grouped per checkpoint)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=4, framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.3)
    _save_fig(out_path)


def plot_best_vs_final(
    rows: list[dict],
    best_label: str,
    final_label: str,
    out_path: str,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    idx = index_rows(rows)
    # Matchups to compare (opponent, depth or None)
    specs: list[tuple[str, str, int | None]] = [
        ("random", "vs random", None),
        ("heuristic", "vs heuristic", None),
        ("self", "vs self", None),
        ("minimax", "minimax d4", 4),
        ("minimax", "minimax d6", 6),
        ("minimax", "minimax d8", 8),
    ]

    labels_txt = []
    best_vals = []
    final_vals = []
    for opp, title, d in specs:
        if d is None:
            rb = idx.get((best_label, opp, None))
            rf = idx.get((final_label, opp, None))
        else:
            rb = idx.get((best_label, "minimax", d))
            rf = idx.get((final_label, "minimax", d))
        if rb is None and rf is None:
            continue
        labels_txt.append(title)
        best_vals.append(_f(rb, "win_rate") if rb else float("nan"))
        final_vals.append(_f(rf, "win_rate") if rf else float("nan"))

    if not labels_txt:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(labels_txt))
    w = 0.35
    ax.bar(x - w / 2, best_vals, width=w, label=best_label, color="#9467bd")
    ax.bar(x + w / 2, final_vals, width=w, label=final_label, color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_txt, rotation=20, ha="right")
    ax.set_ylabel("Win rate")
    ax.set_title("Best checkpoint vs final checkpoint")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _save_fig(out_path)


def plot_heatmap(rows: list[dict], labels: list[str], out_path: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    idx = index_rows(rows)
    col_specs: list[tuple[str, int | None, str]] = []
    for opp, title in [("random", "random"), ("heuristic", "heuristic"), ("self", "self")]:
        col_specs.append((opp, None, title))
    for d in sorted({d for (_, o, d) in idx if o == "minimax" and d is not None}):
        col_specs.append(("minimax", d, f"mm d{d}"))

    mat = np.full((len(labels), len(col_specs)), np.nan)
    for i, lab in enumerate(labels):
        for j, (opp, d, _) in enumerate(col_specs):
            if d is None:
                r = idx.get((lab, opp, None))
            else:
                r = idx.get((lab, "minimax", d))
            if r:
                mat[i, j] = _f(r, "win_rate")

    fig, ax = plt.subplots(figsize=(max(8, len(col_specs) * 0.9), max(5, len(labels) * 0.45)))
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xticks(range(len(col_specs)))
    ax.set_xticklabels([c[2] for c in col_specs], rotation=45, ha="right", fontsize=9)
    ax.set_title("Win rate heatmap (checkpoint × opponent)")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Win rate")
    _save_fig(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot project report eval CSV (one PNG per figure).")
    parser.add_argument("--csv", default=_default_csv_path(), help="Path to project_report_evals.csv")
    parser.add_argument(
        "--out-dir",
        default="",
        help="Directory for PNGs (default: <csv_dir>/project_report_plots)",
    )
    parser.add_argument(
        "--best-label",
        default="best",
        help="Checkpoint label for best.pt row",
    )
    parser.add_argument(
        "--final-label",
        default="self-mixed-0.8_final_total3100000",
        help="Checkpoint label treated as training final",
    )
    parser.add_argument(
        "--grouped-minimax-depth",
        type=int,
        default=6,
        help="Minimax depth for the grouped opponent bar chart (default: 6)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.csv)), "project_report_plots")
    os.makedirs(out_dir, exist_ok=True)

    rows = load_rows(args.csv)
    preferred = _checkpoint_order()
    labels = ordered_labels(rows, preferred)

    plot_minimax_depth_sweep(rows, labels, os.path.join(out_dir, "minimax_depth_sweep.png"))
    plot_opponents_grouped_by_checkpoint(
        rows,
        labels,
        os.path.join(out_dir, "opponents_by_checkpoint.png"),
        minimax_depth=args.grouped_minimax_depth,
    )
    plot_best_vs_final(
        rows,
        args.best_label,
        args.final_label,
        os.path.join(out_dir, "best_vs_final.png"),
    )
    plot_heatmap(rows, labels, os.path.join(out_dir, "win_rate_heatmap.png"))

    print(f"Wrote figures to {out_dir}/")


if __name__ == "__main__":
    main()
