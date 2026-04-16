"""Parallel project-report evaluation runner.

Fans out checkpoint × opponent jobs across CPU cores using multiprocessing.
Fast opponents run in parallel; deep minimax opponents (depth 7/8) get
dedicated cores to avoid CPU contention.
"""

import csv
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.abspath("."))

# ── Job definitions ──────────────────────────────────────────────────────

CHECKPOINT_DIR = "artifacts/runs/dqn/CS4453-FINAL-OVERNIGHT/checkpoints"

CHECKPOINTS = [
    "pretrain_init.pt",
    "random_final_total0050000.pt",
    "random+arbiter_final_total0100000.pt",
    "heuristic_final_total0150000.pt",
    "minimax_final_total1150000.pt",
    "self-mixed-0.6_final_total1900000.pt",
    "self-mixed-0.8_final_total3100000.pt",
    "best.pt",
]

# (opponent, depth, games)
FAST_OPPONENTS = [
    ("random", 4, 1500),
    ("heuristic", 4, 1500),
    ("self", 4, 1500),
    ("minimax", 4, 1500),
    ("minimax", 5, 1000),
    ("minimax", 6, 1000),
]

# Applied to all checkpoints for full capability evolution data
DEEP_OPPONENTS = [
    ("minimax", 7, 150),
    ("minimax", 8, 60),
]

DEEP_CHECKPOINT_NAMES = set(CHECKPOINTS)  # Apply to everything

CSV_OUT = "artifacts/runs/dqn/CS4453-FINAL-OVERNIGHT/project_report_evals.csv"


def run_one_job(args_tuple):
    """Worker function: evaluate one (checkpoint, opponent, depth, games) combo."""
    ckpt_path, ckpt_label, opp_name, depth, games = args_tuple

    # Each worker must import inside the function (fork safety)
    import os
    import sys

    sys.path.insert(0, os.path.abspath("."))
    from evaluation.evaluate import run_evaluation_suite

    tier = "project_eval_deep" if depth >= 7 else "project_eval_fast"

    t0 = time.time()
    rows = run_evaluation_suite(
        "dqn-hybrid",
        [opp_name],
        games=games,
        depth=depth,
        model_path=ckpt_path,
        hybrid_depth=4,
        checkpoint_label=ckpt_label,
        eval_tier=tier,
    )
    elapsed = time.time() - t0

    row = rows[0]
    row["_elapsed"] = round(elapsed, 1)
    row["_ckpt_label"] = ckpt_label

    opp_str = f"{opp_name}" if opp_name != "minimax" else f"minimax-d{depth}"
    print(
        f"  ✓ {ckpt_label:40s} vs {opp_str:14s}  "
        f"W:{row['wins']:>3d} D:{row['draws']:>3d} L:{row['losses']:>3d}  "
        f"({elapsed:.1f}s)",
        flush=True,
    )
    return row


def build_jobs():
    """Build the full list of (ckpt_path, label, opp, depth, games) tuples."""
    jobs = []
    for ckpt_name in CHECKPOINTS:
        path = os.path.join(CHECKPOINT_DIR, ckpt_name)
        if not os.path.exists(path):
            print(f"⚠ Skipping {ckpt_name} (not found)")
            continue
        label = ckpt_name.rsplit(".", 1)[0]

        for opp, depth, games in FAST_OPPONENTS:
            jobs.append((path, label, opp, depth, games))

        if ckpt_name in DEEP_CHECKPOINT_NAMES:
            for opp, depth, games in DEEP_OPPONENTS:
                jobs.append((path, label, opp, depth, games))

    return jobs


def write_csv(all_rows):
    """Write all result rows to CSV."""
    if not all_rows:
        return
    fieldnames = list(all_rows[0].keys())
    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)


def write_markdown(all_rows):
    """Write a nicely formatted Markdown report grouped by checkpoint."""
    md_path = CSV_OUT.replace(".csv", ".md")
    by_ckpt = defaultdict(list)
    for row in all_rows:
        by_ckpt[row["_ckpt_label"]].append(row)

    with open(md_path, "w") as f:
        f.write("# Final Project Robust Evaluation Report\n\n")
        for ckpt_name in CHECKPOINTS:
            label = ckpt_name.rsplit(".", 1)[0]
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


def main():
    # Clean previous output
    if os.path.exists(CSV_OUT):
        os.remove(CSV_OUT)

    jobs = build_jobs()
    total = len(jobs)
    print(f"\n{'='*60}")
    print(f"  Project Evaluation: {total} jobs across {mp.cpu_count()} cores")
    print(f"{'='*60}\n", flush=True)

    # Split into fast jobs (can parallelize heavily) and deep jobs (CPU-bound)
    fast_jobs = [j for j in jobs if j[3] < 7]  # depth < 7
    deep_jobs = [j for j in jobs if j[3] >= 7]  # depth >= 7

    all_rows = []

    # Phase 1: fast jobs — use all cores
    if fast_jobs:
        print(
            f"Phase 1: {len(fast_jobs)} fast jobs (up to {mp.cpu_count()} parallel)\n",
            flush=True,
        )
        with mp.Pool(processes=min(len(fast_jobs), mp.cpu_count())) as pool:
            results = pool.map(run_one_job, fast_jobs)
        all_rows.extend(results)
        print(f"\n  Phase 1 complete: {len(results)} evaluations done.\n", flush=True)

    # Phase 2: deep jobs — limit concurrency to avoid thrashing
    if deep_jobs:
        deep_workers = min(len(deep_jobs), max(2, mp.cpu_count() // 3))
        print(
            f"Phase 2: {len(deep_jobs)} deep jobs ({deep_workers} parallel)\n",
            flush=True,
        )
        with mp.Pool(processes=deep_workers) as pool:
            results = pool.map(run_one_job, deep_jobs)
        all_rows.extend(results)
        print(f"\n  Phase 2 complete: {len(results)} evaluations done.\n", flush=True)

    # Write outputs
    write_csv(all_rows)
    write_markdown(all_rows)
    print(f"\n{'='*60}")
    print(f"  ALL DONE — {len(all_rows)} evaluations saved")
    print(f"  CSV: {CSV_OUT}")
    print(f"  MD:  {CSV_OUT.replace('.csv', '.md')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
