from __future__ import annotations

import argparse
import json
import math
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


AGRICULTURAL_METRICS = {
    "achievement_pct": "higher",
    "unmet_emitter_ratio_pct": "lower",
    "average_deficit_m3_per_emitter": "lower",
    "max_deficit_m3": "lower",
    "pressure_uniformity_index_pct": "higher",
    "feasible_group_ratio_pct": "higher",
    "affected_target_met_pct": "higher",
    "affected_average_deficit_m3": "lower",
    "mean_reoptimization_time_s": "lower",
}
BASE_AGRICULTURAL_METRICS = tuple(AGRICULTURAL_METRICS)


def bootstrap_mean_ci(values: np.ndarray, seed: int, samples: int = 10_000) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    chunk = 1000
    means: list[np.ndarray] = []
    for start in range(0, samples, chunk):
        size = min(chunk, samples - start)
        indices = rng.integers(0, len(clean), size=(size, len(clean)))
        means.append(clean[indices].mean(axis=1))
    distribution = np.concatenate(means)
    return float(np.quantile(distribution, 0.025)), float(np.quantile(distribution, 0.975))


def descriptive_rows(
    frame: pd.DataFrame,
    group_columns: list[str],
    metrics: Iterable[str],
    analysis: str,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        identity = dict(zip(group_columns, key_values))
        for metric_index, metric in enumerate(metrics):
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            low, high = bootstrap_mean_ci(values, seed + metric_index * 101 + len(rows))
            rows.append(
                {
                    "analysis": analysis,
                    **identity,
                    "metric": metric,
                    "n_independent_scenarios": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "ci95_low": low,
                    "ci95_high": high,
                    "median": float(np.median(values)),
                    "best": float(np.max(values) if AGRICULTURAL_METRICS.get(metric) == "higher" else np.min(values)),
                    "worst": float(np.min(values) if AGRICULTURAL_METRICS.get(metric) == "higher" else np.max(values)),
                }
            )
    return rows


def rank_biserial(difference: np.ndarray) -> float:
    difference = np.asarray(difference, dtype=float)
    difference = difference[np.abs(difference) > 1.0e-15]
    if not len(difference):
        return 0.0
    ranks = rankdata(np.abs(difference))
    positive = float(np.sum(ranks[difference > 0]))
    negative = float(np.sum(ranks[difference < 0]))
    return (positive - negative) / max(positive + negative, 1.0e-15)


def paired_test(
    frame: pd.DataFrame,
    id_columns: list[str],
    group_column: str,
    first: str,
    second: str,
    metric: str,
    analysis: str,
) -> dict[str, Any] | None:
    subset = frame[frame[group_column].isin([first, second])]
    pivot = subset.pivot_table(index=id_columns, columns=group_column, values=metric, aggfunc="mean").dropna()
    if len(pivot) < 5 or first not in pivot or second not in pivot:
        return None
    first_values = pivot[first].to_numpy(dtype=float)
    second_values = pivot[second].to_numpy(dtype=float)
    raw_difference = first_values - second_values
    benefit_difference = raw_difference if AGRICULTURAL_METRICS.get(metric, "higher") == "higher" else -raw_difference
    try:
        result = wilcoxon(benefit_difference, zero_method="wilcox", alternative="two-sided")
        statistic, p_value = float(result.statistic), float(result.pvalue)
        if not math.isfinite(p_value):
            statistic, p_value = 0.0, 1.0
    except ValueError:
        statistic, p_value = 0.0, 1.0
    return {
        "analysis": analysis,
        "group_a": first,
        "group_b": second,
        "metric": metric,
        "n_paired_scenarios": len(pivot),
        "mean_a": float(np.mean(first_values)),
        "mean_b": float(np.mean(second_values)),
        "mean_raw_a_minus_b": float(np.mean(raw_difference)),
        "median_benefit_a_over_b": float(np.median(benefit_difference)),
        "wilcoxon_statistic": statistic,
        "p_value": p_value,
        "rank_biserial_benefit_a_over_b": rank_biserial(benefit_difference),
    }


def holm_adjust(rows: list[dict[str, Any]]) -> None:
    families: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        families.setdefault((str(row["analysis"]), str(row["metric"])), []).append(index)
    for indices in families.values():
        ordered = sorted(indices, key=lambda index: rows[index]["p_value"])
        running = 0.0
        total = len(ordered)
        for rank, index in enumerate(ordered):
            adjusted = min(1.0, (total - rank) * rows[index]["p_value"])
            running = max(running, adjusted)
            rows[index]["p_holm"] = running


def scenario_aggregate(frame: pd.DataFrame, identity: list[str], metrics: Iterable[str]) -> pd.DataFrame:
    available = [metric for metric in metrics if metric in frame.columns]
    return frame.groupby(identity, as_index=False, dropna=False)[available].mean(numeric_only=True)


def main_policy_analysis(data_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, pd.DataFrame]]:
    main = pd.read_csv(data_root / "main_experiments" / "revised_policy_metrics.csv")
    metrics = list(BASE_AGRICULTURAL_METRICS)
    descriptive: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    exports: dict[str, pd.DataFrame] = {}

    normal = main[main["experiment"].eq("normal")].copy()
    normal["comparison_policy"] = normal["policy"].replace(
        {"original_near_far": "Original near-to-far", "random_rotation": "Random", "tuned_nsga2_causal": "Tuned NSGA-II"}
    )
    normal_scenarios = scenario_aggregate(normal, ["scenario_id", "comparison_policy"], metrics)
    exports["normal_scenario_aggregates.csv"] = normal_scenarios
    descriptive.extend(descriptive_rows(normal_scenarios, ["comparison_policy"], metrics, "normal_policy", 1001))
    normal_pairs = [
        ("Tuned NSGA-II", "Original near-to-far"),
        ("Tuned NSGA-II", "Random"),
        ("Random", "Original near-to-far"),
    ]
    for first, second in normal_pairs:
        for metric in metrics:
            row = paired_test(normal_scenarios, ["scenario_id"], "comparison_policy", first, second, metric, "normal_policy")
            if row:
                tests.append(row)

    faults = main[main["experiment"].isin(["single_fault", "multi_fault"])].copy()
    faults["decision_mode"] = np.where(faults["policy"].str.contains("dynamic_reoptimization"), "Dynamic", "Static")
    fault_scenarios = scenario_aggregate(faults, ["experiment", "scenario_id", "algorithm", "decision_mode"], metrics)
    exports["fault_scenario_aggregates.csv"] = fault_scenarios
    descriptive.extend(descriptive_rows(fault_scenarios, ["experiment", "algorithm", "decision_mode"], metrics, "fault_policy", 2001))
    for (experiment, algorithm), block in fault_scenarios.groupby(["experiment", "algorithm"]):
        for metric in metrics:
            row = paired_test(block, ["scenario_id"], "decision_mode", "Dynamic", "Static", metric, f"{experiment}_{algorithm}_dynamic_value")
            if row:
                tests.append(row)

    dynamic = fault_scenarios[fault_scenarios["decision_mode"].eq("Dynamic")]
    algorithms = sorted(dynamic["algorithm"].dropna().unique())
    for experiment, block in dynamic.groupby("experiment"):
        for first, second in combinations(algorithms, 2):
            for metric in metrics:
                row = paired_test(block, ["scenario_id"], "algorithm", first, second, metric, f"{experiment}_algorithm_agriculture")
                if row:
                    tests.append(row)
    return descriptive, tests, exports


def algorithm_quality_analysis(data_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    quality = pd.read_csv(data_root / "algorithm_quality" / "pareto_quality_metrics.csv")
    metrics = ["hv_4d", "igd_plus_4d", "additive_epsilon_4d", "nondominated_count", "spacing_4d", "elapsed_s"]
    direction = {"hv_4d": "higher", "nondominated_count": "higher"}
    for metric in metrics:
        AGRICULTURAL_METRICS.setdefault(metric, direction.get(metric, "lower"))
    scenario = quality.groupby(["context_id", "phase", "algorithm"], as_index=False)[metrics].mean()
    descriptive = descriptive_rows(scenario, ["phase", "algorithm"], metrics, "algorithm_quality", 3001)
    tests: list[dict[str, Any]] = []
    global_rows: list[dict[str, Any]] = []
    for phase in ["all", *sorted(scenario["phase"].unique())]:
        block = scenario if phase == "all" else scenario[scenario["phase"].eq(phase)]
        algorithms = sorted(block["algorithm"].unique())
        for metric in metrics:
            pivot = block.pivot(index="context_id", columns="algorithm", values=metric).dropna()
            if len(pivot) >= 5 and len(algorithms) >= 3 and all(algorithm in pivot for algorithm in algorithms):
                statistic, p_value = friedmanchisquare(*[pivot[algorithm] for algorithm in algorithms])
                global_rows.append(
                    {
                        "analysis": "algorithm_quality",
                        "phase": phase,
                        "metric": metric,
                        "n_contexts": len(pivot),
                        "algorithms": ";".join(algorithms),
                        "friedman_statistic": float(statistic),
                        "p_value": float(p_value),
                    }
                )
            for first, second in combinations(algorithms, 2):
                row = paired_test(block, ["context_id"], "algorithm", first, second, metric, f"algorithm_quality_{phase}")
                if row:
                    tests.append(row)
    return descriptive, tests, global_rows, scenario


def optional_analyses(data_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, pd.DataFrame]]:
    descriptive: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    exports: dict[str, pd.DataFrame] = {}
    metrics = list(BASE_AGRICULTURAL_METRICS)

    information_path = data_root / "information_set" / "information_set_policy_metrics.csv"
    if information_path.exists():
        information = pd.read_csv(information_path)
        scenario = scenario_aggregate(information, ["experiment", "scenario_id", "policy"], metrics)
        exports["information_set_scenario_aggregates.csv"] = scenario
        descriptive.extend(descriptive_rows(scenario, ["experiment", "policy"], metrics, "information_set", 4001))
        for experiment, block in scenario.groupby("experiment"):
            policies = sorted(block["policy"].unique())
            for first, second in combinations(policies, 2):
                for metric in metrics:
                    row = paired_test(block, ["scenario_id"], "policy", first, second, metric, f"information_set_{experiment}")
                    if row:
                        tests.append(row)

    surrogate_path = data_root / "surrogate_experiments" / "surrogate_policy_metrics.csv"
    if surrogate_path.exists():
        surrogate = pd.read_csv(surrogate_path)
        base = pd.read_csv(data_root / "main_experiments" / "revised_policy_metrics.csv")
        base = base[
            ((base["experiment"].eq("normal")) & base["policy"].eq("tuned_nsga2_causal"))
            | ((base["experiment"].isin(["single_fault", "multi_fault"])) & base["policy"].eq("nsga2_dynamic_reoptimization"))
        ].copy()
        base["method"] = "Exact NSGA-II"
        surrogate["method"] = "Surrogate-assisted NSGA-II"
        combined = pd.concat([base, surrogate], ignore_index=True, sort=False)
        scenario = scenario_aggregate(combined, ["experiment", "scenario_id", "method"], metrics + ["optimization_time_s", "initial_optimization_time_s", "total_reoptimization_time_s"])
        exports["surrogate_scenario_aggregates.csv"] = scenario
        descriptive.extend(descriptive_rows(scenario, ["experiment", "method"], metrics, "surrogate_comparison", 5001))
        for experiment, block in scenario.groupby("experiment"):
            for metric in metrics:
                row = paired_test(block, ["scenario_id"], "method", "Surrogate-assisted NSGA-II", "Exact NSGA-II", metric, f"surrogate_{experiment}")
                if row:
                    tests.append(row)
    return descriptive, tests, exports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scenario-level statistical analysis for revised experiments")
    repository_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-root", type=Path, default=repository_root / "data")
    parser.add_argument("--output-dir", type=Path, default=repository_root / "outputs" / "paper_statistics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    descriptive, tests, exports = main_policy_analysis(args.data_root)
    quality_descriptive, quality_tests, global_tests, quality_scenarios = algorithm_quality_analysis(args.data_root)
    optional_descriptive, optional_tests, optional_exports = optional_analyses(args.data_root)
    descriptive.extend(quality_descriptive)
    descriptive.extend(optional_descriptive)
    tests.extend(quality_tests)
    tests.extend(optional_tests)
    exports.update(optional_exports)
    exports["algorithm_quality_scenario_aggregates.csv"] = quality_scenarios
    holm_adjust(tests)
    pd.DataFrame(descriptive).to_csv(args.output_dir / "descriptive_statistics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(tests).to_csv(args.output_dir / "paired_wilcoxon_tests.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(global_tests).to_csv(args.output_dir / "friedman_global_tests.csv", index=False, encoding="utf-8-sig")
    for filename, frame in exports.items():
        frame.to_csv(args.output_dir / filename, index=False, encoding="utf-8-sig")
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "independent_unit": "scenario/context; optimizer repeats were aggregated within scenario before inference",
        "confidence_interval": "10,000-replicate nonparametric bootstrap of scenario means",
        "paired_test": "two-sided paired Wilcoxon signed-rank",
        "multiple_testing": "Holm adjustment within analysis and metric families",
        "algorithm_global_test": "Friedman test on scenario-aggregated independent algorithm runs",
        "descriptive_row_count": len(descriptive),
        "paired_test_count": len(tests),
    }
    (args.output_dir / "statistical_analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
