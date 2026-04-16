"""Run the same eval suite as report_evaluator.py on one checkpoint; separate outputs.

Uses FAST_OPPONENTS + DEEP_OPPONENTS from report_evaluator (random/heuristic/self/minimax
depths 4–8 with the same game counts).

Usage:
    python -m evaluation.run_single_project_eval --model /path/to/weights.pt
    python -m evaluation.run_single_project_eval --model /path/to/a.pt \\
        --out-dir artifacts/evals/my_run --label my_label
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath("."))

from evaluation.report_evaluator import (
    DEEP_OPPONENTS,
    FAST_OPPONENTS,
    run_one_job,
)


def build_jobs(ckpt_path: str, label: str, *, include_deep: bool) -> list:
    jobs = []
    for opp, depth, games in FAST_OPPONENTS:
        jobs.append((ckpt_path, label, opp, depth, games))
    if include_deep:
        for opp, depth, games in DEEP_OPPONENTS:
            jobs.append((ckpt_path, label, opp, depth, games))
    return jobs


def write_csv(all_rows: list, csv_path: str) -> None:
    if not all_rows:
        return
    fieldnames = list(all_rows[0].keys())
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)


def write_markdown(all_rows: list, md_path: str, section_labels: list[str]) -> None:
    by_ckpt = defaultdict(list)
    for row in all_rows:
        by_ckpt[row["_ckpt_label"]].append(row)

    os.makedirs(os.path.dirname(os.path.abspath(md_path)) or ".", exist_ok=True)
    with open(md_path, "w") as f:
        f.write("# Project evaluation (single checkpoint)\n\n")
        for label in section_labels:
            if label not in by_ckpt:
                continue
            rows = by_ckpt[label]
            f.write(f"## `{label}`\n\n")
            f.write(
                "| Opponent | Depth | Games | Win % | Draw % | Loss % | W-D-L | Time |\n"
            )
            f.write(
                "|----------|-------|-------|-------|--------|--------|-------|------|\n"
            )
            for r in rows:
                d = r.get("minimax_depth", "-")
                d = "-" if d == "" else d
                wp = float(r["win_rate"]) * 100
                dp = float(r["draw_rate"]) * 100
                lp = float(r["loss_rate"]) * 100
                f.write(
                    f"| {r['opponent']} | {d} | {r['games']} "
                    f"| {wp:.1f}% | {dp:.1f}% | {lp:.1f}% "
                    f"| {r['wins']}-{r['draws']}-{r['losses']} "
                    f"| {r['_elapsed']:.0f}s |\n"
                )
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project-report eval for one checkpoint."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to .pt checkpoint (dqn-hybrid compatible)",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Row label in CSV (default: stem of --model)",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Directory for project_report_evals.csv/.md (default: artifacts/evals/<label>)",
    )
    parser.add_argument(
        "--no-deep",
        action="store_true",
        help="Skip minimax depth 7/8 jobs",
    )
    args = parser.parse_args()

    ckpt_path = os.path.abspath(os.path.expanduser(args.model))
    if not os.path.isfile(ckpt_path):
        print(f"Not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    label = args.label or os.path.splitext(os.path.basename(ckpt_path))[0]
    out_dir = args.out_dir or os.path.join("artifacts", "evals", label)
    csv_path = os.path.join(out_dir, "project_report_evals.csv")

    jobs = build_jobs(ckpt_path, label, include_deep=not args.no_deep)
    total = len(jobs)
    print(f"\n{'='*60}")
    print(f"  Single-checkpoint project eval: {total} jobs ({label})")
    print(f"  Model: {ckpt_path}")
    print(f"  Out:   {csv_path}")
    print(f"{'='*60}\n", flush=True)

    fast_jobs = [j for j in jobs if j[3] < 7]
    deep_jobs = [j for j in jobs if j[3] >= 7]
    all_rows: list = []

    if fast_jobs:
        print(
            f"Phase 1: {len(fast_jobs)} fast jobs (up to {mp.cpu_count()} parallel)\n",
            flush=True,
        )
        with mp.Pool(processes=min(len(fast_jobs), mp.cpu_count())) as pool:
            all_rows.extend(pool.map(run_one_job, fast_jobs))
        print("\n  Phase 1 complete.\n", flush=True)

    if deep_jobs:
        deep_workers = min(len(deep_jobs), max(2, mp.cpu_count() // 3))
        print(
            f"Phase 2: {len(deep_jobs)} deep jobs ({deep_workers} parallel)\n",
            flush=True,
        )
        with mp.Pool(processes=deep_workers) as pool:
            all_rows.extend(pool.map(run_one_job, deep_jobs))
        print("\n  Phase 2 complete.\n", flush=True)

    write_csv(all_rows, csv_path)
    write_markdown(all_rows, csv_path.replace(".csv", ".md"), [label])
    print(f"\n{'='*60}")
    print(f"  Done — {len(all_rows)} rows")
    print(f"  CSV: {csv_path}")
    print(f"  MD:  {csv_path.replace('.csv', '.md')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
