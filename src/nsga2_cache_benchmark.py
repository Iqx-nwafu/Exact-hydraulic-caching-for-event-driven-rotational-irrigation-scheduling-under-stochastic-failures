from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

from network_topology import add_network_args, network_cli_args, network_paths


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Benchmark NSGA-II evaluation caches.")
    p.add_argument("--main-script", type=Path, default=Path(__file__).with_name("main_experiment_after_verification.py"))
    add_network_args(p, root / "data" / "network")
    p.add_argument("--output-base", type=Path, default=root / "outputs" / "cache_benchmark")
    p.add_argument("--workers", type=str, default="auto")
    p.add_argument("--normal-scenarios", type=int, default=20)
    p.add_argument("--single-fault-scenarios", type=int, default=20)
    p.add_argument("--multi-fault-scenarios", type=int, default=50)
    p.add_argument("--random-repeats", type=int, default=5)
    p.add_argument("--nsga-repeats", type=int, default=2)
    p.add_argument("--population", type=int, default=48)
    p.add_argument("--generations", type=int, default=30)
    p.add_argument("--base-seed", type=int, default=20260709)
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    return p.parse_args()


def run_case(args: argparse.Namespace, label: str, disable_cache: bool) -> Path:
    out_dir = args.output_base / label
    cmd = [
        str(args.python),
        str(args.main_script),
        "--output-dir",
        str(out_dir),
        "--workers",
        args.workers,
        "--normal-scenarios",
        str(args.normal_scenarios),
        "--single-fault-scenarios",
        str(args.single_fault_scenarios),
        "--multi-fault-scenarios",
        str(args.multi_fault_scenarios),
        "--random-repeats",
        str(args.random_repeats),
        "--nsga-repeats",
        str(args.nsga_repeats),
        "--dynamic-algorithms",
        "nsga2",
        "--population",
        str(args.population),
        "--generations",
        str(args.generations),
        "--base-seed",
        str(args.base_seed),
    ] + network_cli_args(*network_paths(args))
    if disable_cache:
        cmd.append("--disable-cache")
    subprocess.run(cmd, check=True)
    return out_dir


def read_run(path: Path) -> tuple[pd.DataFrame, dict]:
    metrics = pd.read_csv(path / "all_policy_metrics_after_verification.csv")
    metadata = json.loads((path / "experiment_metadata_after_verification.json").read_text(encoding="utf-8"))
    return metrics, metadata


def compare_runs(output_base: Path, no_dir: Path, cache_dir: Path) -> None:
    no, no_meta = read_run(no_dir)
    cache, cache_meta = read_run(cache_dir)
    keys = ["experiment", "scenario_id", "policy", "repeat", "seed"]
    merged = no.merge(cache, on=keys, suffixes=("_no_cache", "_with_cache"))
    metric_cols = [
        "achievement_pct",
        "unmet_emitter_ratio_pct",
        "average_deficit_m3_per_emitter",
        "max_deficit_m3",
        "pressure_uniformity_index_pct",
        "feasible_group_ratio_pct",
        "total_reoptimization_time_s",
        "initial_optimization_time_s",
    ]
    metric_rows: list[dict] = []
    for metric in metric_cols:
        left_name = f"{metric}_no_cache"
        right_name = f"{metric}_with_cache"
        if left_name not in merged.columns or right_name not in merged.columns:
            continue
        left = pd.to_numeric(merged[left_name], errors="coerce")
        right = pd.to_numeric(merged[right_name], errors="coerce")
        diff = right - left
        metric_rows.append(
            {
                "metric": metric,
                "n_pairs": int(diff.notna().sum()),
                "mean_no_cache": float(left.mean()),
                "mean_with_cache": float(right.mean()),
                "mean_difference_with_minus_no": float(diff.mean()),
                "max_abs_difference": float(diff.abs().max()),
                "nonzero_pairs_gt_1e-9": int((diff.abs() > 1e-9).sum()),
            }
        )
    metric_compare = pd.DataFrame(metric_rows)

    summary_rows: list[dict] = []
    cache_cols = [
        "total_reoptimization_time_s",
        "initial_optimization_time_s",
        "cache_group_hits",
        "cache_group_misses",
        "cache_group_hit_rate_pct",
        "cache_pipe_loss_hits",
        "cache_pipe_loss_misses",
        "cache_pipe_loss_hit_rate_pct",
        "cache_plan_hits",
        "cache_plan_misses",
        "cache_plan_hit_rate_pct",
        "initial_cache_group_hits",
        "initial_cache_group_misses",
        "initial_cache_group_hit_rate_pct",
        "initial_cache_pipe_loss_hits",
        "initial_cache_pipe_loss_misses",
        "initial_cache_pipe_loss_hit_rate_pct",
        "initial_cache_plan_hits",
        "initial_cache_plan_misses",
        "initial_cache_plan_hit_rate_pct",
    ]
    for label, df, meta in [("no_cache", no, no_meta), ("with_cache", cache, cache_meta)]:
        for (experiment, policy), group in df.groupby(["experiment", "policy"], dropna=False):
            row = {
                "run": label,
                "experiment": experiment,
                "policy": policy,
                "n": len(group),
                "wall_time_s": float(meta["total_wall_time_s"]),
            }
            for col in cache_cols:
                if col in group.columns:
                    row[f"{col}_mean"] = float(pd.to_numeric(group[col], errors="coerce").mean())
            summary_rows.append(row)
    time_summary = pd.DataFrame(summary_rows)

    wall_no = float(no_meta["total_wall_time_s"])
    wall_cache = float(cache_meta["total_wall_time_s"])
    wall_compare = pd.DataFrame(
        [
            {
                "benchmark": "nsga2_cache_acceleration_benchmark",
                "normal_scenarios": no_meta["normal_scenarios"],
                "single_fault_scenarios": no_meta["single_fault_scenarios"],
                "multi_fault_scenarios": no_meta["multi_fault_scenarios"],
                "workers": no_meta["workers"],
                "population": no_meta["population"],
                "generations": no_meta["generations"],
                "no_cache_wall_time_s": wall_no,
                "with_cache_wall_time_s": wall_cache,
                "wall_time_delta_s": wall_cache - wall_no,
                "speedup_factor_no_cache_div_with_cache": wall_no / wall_cache if wall_cache else math.nan,
                "time_reduction_pct": (1.0 - wall_cache / wall_no) * 100.0 if wall_no else math.nan,
            }
        ]
    )

    output_base.mkdir(parents=True, exist_ok=True)
    metric_compare.to_csv(output_base / "v2_cache_metric_equivalence_comparison.csv", index=False, encoding="utf-8-sig")
    time_summary.to_csv(output_base / "v2_cache_time_and_hit_rate_summary.csv", index=False, encoding="utf-8-sig")
    wall_compare.to_csv(output_base / "v2_cache_wall_time_comparison.csv", index=False, encoding="utf-8-sig")
    report = [
        "# NSGA-II Cache Acceleration Benchmark",
        "",
        f"- No-cache wall time: {wall_no:.3f} s",
        f"- With-cache wall time: {wall_cache:.3f} s",
        f"- Speedup factor: {wall_no / wall_cache:.4f}",
        f"- Time reduction: {(1.0 - wall_cache / wall_no) * 100.0:.2f}%",
        "",
        "## Metric Equivalence",
        "",
        metric_compare.to_markdown(index=False),
        "",
        "## Time And Cache Hit Rate Summary",
        "",
        time_summary.to_markdown(index=False),
    ]
    (output_base / "v2_cache_acceleration_benchmark_report.md").write_text("\n".join(report), encoding="utf-8")
    print(wall_compare.to_string(index=False))


def main() -> None:
    args = parse_args()
    no_dir = run_case(args, "v2_no_cache", disable_cache=True)
    cache_dir = run_case(args, "v2_with_cache", disable_cache=False)
    compare_runs(args.output_base, no_dir, cache_dir)


if __name__ == "__main__":
    main()
