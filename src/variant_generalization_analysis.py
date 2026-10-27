from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network_topology import add_network_args, network_paths


BASE_MODULE: Any | None = None
BASE_NETWORK: Any | None = None
BASE_PARAMS: Any | None = None


@dataclass(frozen=True)
class VariantSpec:
    category: str
    level: str
    elevation_sd_m: float
    fault_probability: float
    pressure_min_m: float
    pressure_max_m: float
    baseline_level: str


AGRICULTURAL_METRICS = [
    "achievement_pct",
    "unmet_emitter_ratio_pct",
    "average_deficit_m3_per_emitter",
    "max_deficit_m3",
    "pressure_uniformity_index_pct",
    "feasible_group_ratio_pct",
    "affected_mean_completion_pct",
    "affected_target_met_pct",
    "total_reoptimization_time_s",
    "mean_reoptimization_time_s",
]

BENEFICIAL_HIGHER = {
    "achievement_pct",
    "pressure_uniformity_index_pct",
    "feasible_group_ratio_pct",
    "affected_mean_completion_pct",
    "affected_target_met_pct",
}


def load_base_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("verified_main_experiment", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base experiment module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def variant_specs() -> list[VariantSpec]:
    specs: list[VariantSpec] = []
    for sd in [0.0, 1.0, 2.0]:
        specs.append(
            VariantSpec(
                category="elevation_disturbance",
                level=f"sd_{sd:.1f}m",
                elevation_sd_m=sd,
                fault_probability=0.50,
                pressure_min_m=20.0,
                pressure_max_m=30.0,
                baseline_level="sd_0.0m",
            )
        )
    for prob in [0.25, 0.50, 0.75]:
        specs.append(
            VariantSpec(
                category="fault_probability",
                level=f"p_{prob:.2f}",
                elevation_sd_m=0.0,
                fault_probability=prob,
                pressure_min_m=20.0,
                pressure_max_m=30.0,
                baseline_level="p_0.50",
            )
        )
    for p_min, p_max in [(23.0, 27.0), (20.0, 30.0), (17.0, 33.0)]:
        specs.append(
            VariantSpec(
                category="pressure_range",
                level=f"{p_min:.0f}_{p_max:.0f}m",
                elevation_sd_m=0.0,
                fault_probability=0.50,
                pressure_min_m=p_min,
                pressure_max_m=p_max,
                baseline_level="20_30m",
            )
        )
    return specs


SCENARIO_KEY_COLUMNS = ("variant_category", "variant_level", "experiment", "scenario_id", "policy", "repeat")


def canonical_row_order(df: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap resampling and float sums depend on row order; fix it independently of worker completion order."""
    keys = [c for c in SCENARIO_KEY_COLUMNS if c in df.columns]
    return df.sort_values(keys, kind="mergesort").reset_index(drop=True)


def worker_count(value: str) -> int:
    cpu = os.cpu_count() or 1
    if value.lower() == "auto":
        return max(1, math.floor(cpu * 0.80))
    return max(1, int(value))


def init_worker(
    main_script: str,
    nodes: str,
    pipes: str,
    roughness: dict[str, float],
    cache_enabled: bool,
    group_cache_max_items: int,
    plan_cache_max_items: int,
    pipe_loss_cache_max_items: int,
    emitters: str | None = None,
) -> None:
    global BASE_MODULE, BASE_NETWORK, BASE_PARAMS
    BASE_MODULE = load_base_module(Path(main_script))
    params = BASE_MODULE.Params()
    BASE_MODULE.init_worker(
        nodes,
        pipes,
        params.__dict__,
        roughness,
        cache_enabled,
        group_cache_max_items,
        plan_cache_max_items,
        pipe_loss_cache_max_items,
        emitters,
    )
    BASE_NETWORK = BASE_MODULE.WORKER_NETWORK
    BASE_PARAMS = BASE_MODULE.WORKER_PARAMS


def source_profile(seed: int, params: Any, p_min: float, p_max: float, hours: int = 121) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(p_min, p_max) for _ in range(hours)]


def network_with_elevation_disturbance(network: Any, sd_m: float, seed: int) -> Any:
    if sd_m <= 0.0:
        return network
    rng = np.random.default_rng(seed)
    node_z = dict(network.node_z)
    nodes = network.nodes.copy()
    for node in node_z:
        if node == network.source_node:
            continue
        node_z[node] = float(node_z[node] + rng.normal(0.0, sd_m))
    nodes["Z"] = nodes["node_id"].map(node_z).astype(float)
    return replace(network, nodes=nodes, node_z=node_z)


def fault_events(seed: int, network: Any, params: Any, fault_probability: float, recovery_probability: float = 0.75) -> list[Any]:
    assert BASE_MODULE is not None
    rng = random.Random(seed)
    nodes = list(network.irrigation_nodes)
    rng.shuffle(nodes)
    events: list[Any] = []
    for _ in range(3):
        if not nodes or rng.random() > fault_probability:
            continue
        node = nodes.pop()
        fault_t = rng.uniform(0.0, 110.0)
        events.append(BASE_MODULE.Event(fault_t, "fault", node))
        if rng.random() < recovery_probability:
            recovery_t = rng.uniform(fault_t, params.horizon_h)
            events.append(BASE_MODULE.Event(recovery_t, "recovery", node))
    return sorted(events, key=lambda e: (e.time_h, 0 if e.kind == "fault" else 1))


def add_variant_identity(row: dict[str, Any], task: dict[str, Any], experiment: str, policy: str, repeat: int, seed: int) -> dict[str, Any]:
    out = dict(row)
    out.update(
        {
            "variant_category": task["category"],
            "variant_level": task["level"],
            "baseline_level": task["baseline_level"],
            "elevation_sd_m": task["elevation_sd_m"],
            "fault_probability": task["fault_probability"],
            "pressure_min_m": task["pressure_min_m"],
            "pressure_max_m": task["pressure_max_m"],
            "scenario_id": task["scenario_id"],
            "experiment": experiment,
            "policy": policy,
            "repeat": repeat,
            "seed": seed,
        }
    )
    return out


def run_variant_task(task: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assert BASE_MODULE is not None and BASE_NETWORK is not None and BASE_PARAMS is not None
    base = BASE_MODULE
    params = BASE_PARAMS
    seed = int(task["seed"])
    population = int(task["population"])
    generations = int(task["generations"])
    random_repeats = int(task["random_repeats"])

    network = network_with_elevation_disturbance(BASE_NETWORK, float(task["elevation_sd_m"]), seed + 17)
    base.WORKER_NETWORK = network
    profile = source_profile(seed + 101, params, float(task["pressure_min_m"]), float(task["pressure_max_m"]))

    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    original = base.original_plan(network, params)
    original_order = [e for group in original for e in group.emitters]

    metrics, _, _ = base.evaluate_plan(network, params, original, profile)
    metrics["optimizer_algorithm"] = "none"
    rows.append(add_variant_identity(metrics, task, "normal", "original_near_far", 0, seed))

    for repeat in range(1, random_repeats + 1):
        plan = base.random_plan(network, params, random.Random(seed + repeat * 101))
        metrics, _, _ = base.evaluate_plan(network, params, plan, profile)
        metrics["optimizer_algorithm"] = "none"
        rows.append(add_variant_identity(metrics, task, "normal", "random_rotation", repeat, seed + repeat * 101))

    plan, opt_metrics, elapsed, trace = base.optimize_nsga2_plan(
        list(range(len(network.emitters))),
        params,
        profile,
        seed + 1009,
        population,
        generations,
        initial_orders=[original_order],
        algorithm="nsga2",
    )
    metrics, _, _ = base.evaluate_plan(network, params, plan, profile)
    base.copy_cache_metrics(metrics, opt_metrics)
    metrics["total_reoptimization_time_s"] = elapsed
    metrics["mean_reoptimization_time_s"] = elapsed
    metrics["reoptimization_count"] = 1
    metrics["optimizer_algorithm"] = "nsga2"
    rows.append(add_variant_identity(metrics, task, "normal", "nsga2_generalized", 1, seed + 1009))
    for row in trace:
        row.update(
            {
                "variant_category": task["category"],
                "variant_level": task["level"],
                "scenario_id": task["scenario_id"],
                "experiment": "normal",
                "policy": "nsga2_generalized",
                "repeat": 1,
            }
        )
        traces.append(row)

    events = fault_events(seed + 333, network, params, float(task["fault_probability"]))
    event_text = ";".join(f"{event.kind}:{event.node}@{event.time_h:.2f}" for event in events)
    for dynamic, policy in [(False, "nsga2_static_no_reoptimization"), (True, "nsga2_dynamic_reoptimization")]:
        dyn_metrics, _, dyn_trace = base.simulate_plan_with_events(
            network,
            params,
            plan,
            profile,
            events,
            dynamic,
            seed + (5000 if dynamic else 6000),
            population,
            max(10, generations // 2),
            optimizer_algorithm="nsga2",
        )
        dyn_metrics["initial_optimization_time_s"] = elapsed
        base.copy_cache_metrics(dyn_metrics, opt_metrics, prefix="initial_")
        dyn_metrics["event_count"] = len(events)
        dyn_metrics["fault_count"] = sum(1 for event in events if event.kind == "fault")
        dyn_metrics["recovery_count"] = sum(1 for event in events if event.kind == "recovery")
        dyn_metrics["events"] = event_text
        dyn_metrics["optimizer_algorithm"] = "nsga2"
        rows.append(add_variant_identity(dyn_metrics, task, "dynamic_fault", policy, 1, seed))
        for row in dyn_trace:
            row.update(
                {
                    "variant_category": task["category"],
                    "variant_level": task["level"],
                    "scenario_id": task["scenario_id"],
                    "experiment": "dynamic_fault",
                    "policy": policy,
                    "repeat": 1,
                }
            )
            traces.append(row)
    return rows, traces


def bootstrap_mean_ci(values: pd.Series, rng: np.random.Generator, n_boot: int = 1000) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return math.nan, math.nan
    if len(clean) == 1:
        return float(clean[0]), float(clean[0])
    idx = rng.integers(0, len(clean), size=(n_boot, len(clean)))
    means = clean[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def aggregate_metrics(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    group_cols = ["variant_category", "variant_level", "experiment", "policy"]
    for keys, group in df.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys))
        base["n"] = len(group)
        for metric in AGRICULTURAL_METRICS:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            low, high = bootstrap_mean_ci(values, rng)
            base[f"{metric}_mean"] = float(values.mean())
            base[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            base[f"{metric}_ci95_low"] = low
            base[f"{metric}_ci95_high"] = high
            base[f"{metric}_worst"] = float(values.min() if metric in BENEFICIAL_HIGHER else values.max())
        rows.append(base)
    return pd.DataFrame(rows)


def scenario_policy_means(df: pd.DataFrame) -> pd.DataFrame:
    value_cols = [c for c in AGRICULTURAL_METRICS if c in df.columns]
    return df.groupby(
        ["variant_category", "variant_level", "baseline_level", "scenario_id", "experiment", "policy"],
        dropna=False,
        as_index=False,
    )[value_cols].mean()


def safe_wilcoxon(left: pd.Series, right: pd.Series) -> tuple[float, float, int]:
    a = pd.to_numeric(left, errors="coerce")
    b = pd.to_numeric(right, errors="coerce")
    mask = a.notna() & b.notna()
    a = a[mask].to_numpy(dtype=float)
    b = b[mask].to_numpy(dtype=float)
    if len(a) == 0:
        return math.nan, math.nan, 0
    if np.all(np.abs(a - b) <= 1e-12):
        return 0.0, 1.0, int(len(a))
    stat, p_value = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return float(stat), float(p_value), int(len(a))


def paired_wilcoxon(df: pd.DataFrame) -> pd.DataFrame:
    means = scenario_policy_means(df)
    rows: list[dict[str, Any]] = []
    for (category, level), group in means.groupby(["variant_category", "variant_level"], dropna=False):
        normal = group[group["experiment"].eq("normal")]
        nsga = normal[normal["policy"].eq("nsga2_generalized")]
        for baseline in ["original_near_far", "random_rotation"]:
            other = normal[normal["policy"].eq(baseline)]
            merged = nsga.merge(other, on=["variant_category", "variant_level", "scenario_id", "experiment"], suffixes=("_nsga", "_base"))
            for metric in [m for m in AGRICULTURAL_METRICS if f"{m}_nsga" in merged.columns]:
                stat, p_value, n = safe_wilcoxon(merged[f"{metric}_nsga"], merged[f"{metric}_base"])
                rows.append(
                    {
                        "variant_category": category,
                        "variant_level": level,
                        "comparison": f"nsga2_generalized_vs_{baseline}",
                        "metric": metric,
                        "n_pairs": n,
                        "wilcoxon_statistic": stat,
                        "p_value": p_value,
                        "mean_difference_left_minus_right": float((merged[f"{metric}_nsga"] - merged[f"{metric}_base"]).mean()) if n else math.nan,
                    }
                )
        dyn = group[group["experiment"].eq("dynamic_fault")]
        left = dyn[dyn["policy"].eq("nsga2_dynamic_reoptimization")]
        right = dyn[dyn["policy"].eq("nsga2_static_no_reoptimization")]
        merged = left.merge(right, on=["variant_category", "variant_level", "scenario_id", "experiment"], suffixes=("_dyn", "_static"))
        for metric in [m for m in AGRICULTURAL_METRICS if f"{m}_dyn" in merged.columns]:
            stat, p_value, n = safe_wilcoxon(merged[f"{metric}_dyn"], merged[f"{metric}_static"])
            rows.append(
                {
                    "variant_category": category,
                    "variant_level": level,
                    "comparison": "dynamic_reoptimization_vs_static",
                    "metric": metric,
                    "n_pairs": n,
                    "wilcoxon_statistic": stat,
                    "p_value": p_value,
                    "mean_difference_left_minus_right": float((merged[f"{metric}_dyn"] - merged[f"{metric}_static"]).mean()) if n else math.nan,
                }
            )
    return pd.DataFrame(rows)


def dynamic_gain_table(df: pd.DataFrame) -> pd.DataFrame:
    means = scenario_policy_means(df)
    dyn = means[means["experiment"].eq("dynamic_fault")]
    left = dyn[dyn["policy"].eq("nsga2_dynamic_reoptimization")]
    right = dyn[dyn["policy"].eq("nsga2_static_no_reoptimization")]
    merged = left.merge(right, on=["variant_category", "variant_level", "baseline_level", "scenario_id", "experiment"], suffixes=("_dyn", "_static"))
    rows = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "variant_category": row["variant_category"],
                "variant_level": row["variant_level"],
                "baseline_level": row["baseline_level"],
                "scenario_id": int(row["scenario_id"]),
                "achievement_gain_pct_points": row["achievement_pct_dyn"] - row["achievement_pct_static"],
                "unmet_ratio_reduction_pct_points": row["unmet_emitter_ratio_pct_static"] - row["unmet_emitter_ratio_pct_dyn"],
                "average_deficit_reduction_m3": row["average_deficit_m3_per_emitter_static"] - row["average_deficit_m3_per_emitter_dyn"],
                "affected_completion_gain_pct_points": row["affected_mean_completion_pct_dyn"] - row["affected_mean_completion_pct_static"],
                "dynamic_reoptimization_time_s": row["total_reoptimization_time_s_dyn"],
            }
        )
    return pd.DataFrame(rows)


def degradation_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline_by_category = {
        "elevation_disturbance": "sd_0.0m",
        "fault_probability": "p_0.50",
        "pressure_range": "20_30m",
    }
    for category, group in summary.groupby("variant_category", dropna=False):
        baseline_level = baseline_by_category.get(category, "")
        group = group.assign(_baseline_level=baseline_level)
        for experiment, policy in [
            ("normal", "nsga2_generalized"),
            ("dynamic_fault", "nsga2_dynamic_reoptimization"),
        ]:
            sub = group[group["experiment"].eq(experiment) & group["policy"].eq(policy)]
            if sub.empty:
                continue
            baseline = sub[sub["variant_level"].eq(sub["_baseline_level"].iloc[0])]
            if baseline.empty:
                continue
            base_row = baseline.iloc[0]
            for _, row in sub.iterrows():
                rows.append(
                    {
                        "variant_category": category,
                        "variant_level": row["variant_level"],
                        "baseline_level": sub["_baseline_level"].iloc[0],
                        "experiment": experiment,
                        "policy": policy,
                        "achievement_delta_pct_points": row.get("achievement_pct_mean", math.nan) - base_row.get("achievement_pct_mean", math.nan),
                        "unmet_ratio_delta_pct_points": row.get("unmet_emitter_ratio_pct_mean", math.nan) - base_row.get("unmet_emitter_ratio_pct_mean", math.nan),
                        "avg_deficit_delta_m3": row.get("average_deficit_m3_per_emitter_mean", math.nan) - base_row.get("average_deficit_m3_per_emitter_mean", math.nan),
                        "pressure_uniformity_delta_pct_points": row.get("pressure_uniformity_index_pct_mean", math.nan) - base_row.get("pressure_uniformity_index_pct_mean", math.nan),
                        "feasible_group_ratio_delta_pct_points": row.get("feasible_group_ratio_pct_mean", math.nan) - base_row.get("feasible_group_ratio_pct_mean", math.nan),
                    }
                )
    return pd.DataFrame(rows)


def sensitivity_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (category, experiment, policy), group in summary.groupby(["variant_category", "experiment", "policy"], dropna=False):
        for metric in [
            "achievement_pct_mean",
            "unmet_emitter_ratio_pct_mean",
            "average_deficit_m3_per_emitter_mean",
            "pressure_uniformity_index_pct_mean",
            "feasible_group_ratio_pct_mean",
        ]:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "variant_category": category,
                    "experiment": experiment,
                    "policy": policy,
                    "metric": metric,
                    "level_range": float(values.max() - values.min()),
                    "level_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "min_level_mean": float(values.min()),
                    "max_level_mean": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def create_figures(output_dir: Path, df: pd.DataFrame, gain: pd.DataFrame) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for category, group in df.groupby("variant_category"):
        for metric in [
            "achievement_pct",
            "unmet_emitter_ratio_pct",
            "average_deficit_m3_per_emitter",
            "pressure_uniformity_index_pct",
            "feasible_group_ratio_pct",
        ]:
            if metric not in group.columns:
                continue
            plt.figure(figsize=(10, 5))
            labels = []
            data = []
            for level, level_group in group[group["policy"].eq("nsga2_generalized")].groupby("variant_level"):
                vals = pd.to_numeric(level_group[metric], errors="coerce").dropna().to_numpy(dtype=float)
                if len(vals):
                    labels.append(level)
                    data.append(vals)
            if data:
                plt.boxplot(data, tick_labels=labels, showmeans=True)
                plt.title(f"{category}: NSGA-II {metric}")
                plt.ylabel(metric)
                plt.tight_layout()
                plt.savefig(fig_dir / f"boxplot_{category}_nsga2_{metric}.png", dpi=220)
            plt.close()

    if not gain.empty:
        for category, group in gain.groupby("variant_category"):
            for metric in [
                "achievement_gain_pct_points",
                "unmet_ratio_reduction_pct_points",
                "average_deficit_reduction_m3",
                "affected_completion_gain_pct_points",
            ]:
                vals_by_level = []
                labels = []
                for level, level_group in group.groupby("variant_level"):
                    vals = pd.to_numeric(level_group[metric], errors="coerce").dropna().to_numpy(dtype=float)
                    if len(vals):
                        labels.append(level)
                        vals_by_level.append(vals)
                if not vals_by_level:
                    continue
                plt.figure(figsize=(10, 5))
                plt.boxplot(vals_by_level, tick_labels=labels, showmeans=True)
                plt.title(f"{category}: dynamic gain {metric}")
                plt.ylabel(metric)
                plt.tight_layout()
                plt.savefig(fig_dir / f"boxplot_{category}_dynamic_gain_{metric}.png", dpi=220)
                plt.close()


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    degradation: pd.DataFrame,
    sensitivity: pd.DataFrame,
    dynamic_gain: pd.DataFrame,
    wall_time_s: float,
) -> None:
    def table(df: pd.DataFrame, max_rows: int = 30) -> str:
        if df.empty:
            return "No records."
        return df.head(max_rows).to_markdown(index=False)

    key_summary_cols = [
        "variant_category",
        "variant_level",
        "experiment",
        "policy",
        "n",
        "achievement_pct_mean",
        "unmet_emitter_ratio_pct_mean",
        "average_deficit_m3_per_emitter_mean",
        "pressure_uniformity_index_pct_mean",
        "feasible_group_ratio_pct_mean",
        "affected_mean_completion_pct_mean",
        "total_reoptimization_time_s_mean",
    ]
    key_summary = summary[[c for c in key_summary_cols if c in summary.columns]]
    gain_desc = pd.DataFrame()
    if not dynamic_gain.empty:
        gain_desc = dynamic_gain.groupby(["variant_category", "variant_level"], as_index=False)[
            [
                "achievement_gain_pct_points",
                "unmet_ratio_reduction_pct_points",
                "average_deficit_reduction_m3",
                "affected_completion_gain_pct_points",
                "dynamic_reoptimization_time_s",
            ]
        ].mean()

    lines = [
        "# Variant Network Generalization Analysis",
        "",
        f"Total wall time: {wall_time_s / 60.0:.2f} min",
        "",
        "## Design",
        "",
        "- Variant classes: elevation disturbance, fault probability, and pressure range.",
        "- Each class has three levels.",
        "- Metrics are reported as agricultural management indicators.",
        "",
        "## Policy Summary",
        "",
        table(key_summary, 60),
        "",
        "## Generalization Degradation Relative To Baseline Level",
        "",
        table(degradation, 60),
        "",
        "## Sensitivity Across Levels",
        "",
        table(sensitivity, 60),
        "",
        "## Dynamic Reoptimization Gain",
        "",
        table(gain_desc, 60),
        "",
        "## Output Files",
        "",
        "- `variant_all_metrics.csv`",
        "- `variant_summary_with_ci.csv`",
        "- `variant_generalization_degradation.csv`",
        "- `variant_sensitivity_indices.csv`",
        "- `variant_dynamic_reoptimization_gain.csv`",
        "- `variant_paired_wilcoxon_tests.csv`",
        "- `variant_convergence_traces.csv`",
        "- `figures/*.png`",
    ]
    (output_dir / "variant_generalization_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    network_dir = root / "data" / "network"
    p = argparse.ArgumentParser(description="Variant network generalization analysis.")
    p.add_argument("--main-script", type=Path, default=Path(__file__).with_name("main_experiment_after_verification.py"))
    add_network_args(p, network_dir)
    p.add_argument("--output-dir", type=Path, default=root / "outputs" / "variant_network")
    p.add_argument("--workers", type=str, default="auto")
    p.add_argument("--base-seed", type=int, default=20260709)
    p.add_argument("--scenarios-per-level", type=int, default=30)
    p.add_argument("--random-repeats", type=int, default=10)
    p.add_argument("--population", type=int, default=48)
    p.add_argument("--generations", type=int, default=30)
    p.add_argument("--roughness-upvc-mm", type=float, default=0.018)
    p.add_argument("--roughness-pe-mm", type=float, default=0.020)
    p.add_argument("--disable-cache", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = variant_specs()
    workers = worker_count(args.workers)
    roughness = {"UPVC": args.roughness_upvc_mm, "PE": args.roughness_pe_mm}
    tasks: list[dict[str, Any]] = []
    for spec_idx, spec in enumerate(specs, start=1):
        for scenario_id in range(1, args.scenarios_per_level + 1):
            tasks.append(
                {
                    "category": spec.category,
                    "level": spec.level,
                    "baseline_level": spec.baseline_level,
                    "elevation_sd_m": spec.elevation_sd_m,
                    "fault_probability": spec.fault_probability,
                    "pressure_min_m": spec.pressure_min_m,
                    "pressure_max_m": spec.pressure_max_m,
                    "scenario_id": scenario_id,
                    "seed": args.base_seed + spec_idx * 100000 + scenario_id * 100,
                    "random_repeats": args.random_repeats,
                    "population": args.population,
                    "generations": args.generations,
                }
            )

    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "workers": workers,
        "cpu_count": os.cpu_count(),
        "scenarios_per_level": args.scenarios_per_level,
        "variant_levels": [spec.__dict__ for spec in specs],
        "random_repeats": args.random_repeats,
        "population": args.population,
        "generations": args.generations,
        "cache_enabled": not args.disable_cache,
    }
    (args.output_dir / "variant_experiment_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    nodes, pipes, emitters = network_paths(args)
    init_args = (
        str(args.main_script),
        nodes,
        pipes,
        roughness,
        not args.disable_cache,
        50000,
        20000,
        20000,
        emitters,
    )
    all_rows: list[dict[str, Any]] = []
    all_traces: list[dict[str, Any]] = []
    start = time.perf_counter()
    print(f"Workers: {workers} of {os.cpu_count() or 1} logical CPUs", flush=True)
    print(f"Variant tasks: {len(tasks)}", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=init_args) as executor:
        futures = [executor.submit(run_variant_task, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows, traces = future.result()
            all_rows.extend(rows)
            all_traces.extend(traces)
            if completed == 1 or completed % 20 == 0 or completed == len(futures):
                elapsed = time.perf_counter() - start
                print(f"Completed {completed}/{len(futures)} tasks; elapsed={elapsed/60:.1f} min", flush=True)

    metrics_df = pd.DataFrame(all_rows)
    traces_df = pd.DataFrame(all_traces)
    summary_input = canonical_row_order(metrics_df)
    summary_df = aggregate_metrics(summary_input, args.base_seed + 901)
    dynamic_gain = dynamic_gain_table(summary_input)
    degradation = degradation_table(summary_df)
    sensitivity = sensitivity_table(summary_df)
    wilcoxon_df = paired_wilcoxon(summary_input)
    create_figures(args.output_dir, summary_input, dynamic_gain)
    elapsed = time.perf_counter() - start

    metrics_df.to_csv(args.output_dir / "variant_all_metrics.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(args.output_dir / "variant_summary_with_ci.csv", index=False, encoding="utf-8-sig")
    degradation.to_csv(args.output_dir / "variant_generalization_degradation.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(args.output_dir / "variant_sensitivity_indices.csv", index=False, encoding="utf-8-sig")
    dynamic_gain.to_csv(args.output_dir / "variant_dynamic_reoptimization_gain.csv", index=False, encoding="utf-8-sig")
    wilcoxon_df.to_csv(args.output_dir / "variant_paired_wilcoxon_tests.csv", index=False, encoding="utf-8-sig")
    traces_df.to_csv(args.output_dir / "variant_convergence_traces.csv", index=False, encoding="utf-8-sig")
    metadata["total_wall_time_s"] = elapsed
    (args.output_dir / "variant_experiment_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.output_dir, summary_df, degradation, sensitivity, dynamic_gain, elapsed)

    print(f"Finished in {elapsed/60:.2f} min. Output: {args.output_dir}", flush=True)
    print(
        summary_df[
            [
                "variant_category",
                "variant_level",
                "experiment",
                "policy",
                "n",
                "achievement_pct_mean",
                "unmet_emitter_ratio_pct_mean",
                "average_deficit_m3_per_emitter_mean",
                "feasible_group_ratio_pct_mean",
            ]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
