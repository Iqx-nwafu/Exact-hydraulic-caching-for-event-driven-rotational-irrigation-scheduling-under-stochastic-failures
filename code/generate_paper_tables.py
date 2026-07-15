from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MAIN = DATA / "paper_tables" / "main"
SUPP = DATA / "paper_tables" / "supplementary"

METRICS = [
    "achievement_pct",
    "unmet_emitter_ratio_pct",
    "average_deficit_m3_per_emitter",
    "max_deficit_m3",
    "pressure_uniformity_index_pct",
    "feasible_group_ratio_pct",
]


def compact_summary(frame: pd.DataFrame, groups: list[str], metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(groups, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        identity = dict(zip(groups, key_values))
        for metric in metrics:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            rows.append(
                {
                    **identity,
                    "metric": metric,
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "mean_sd": f"{np.mean(values):.4f} +/- {np.std(values, ddof=1) if len(values) > 1 else 0.0:.4f}",
                    "median": float(np.median(values)),
                    "worst_min": float(np.min(values)),
                    "worst_max": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def write_settings() -> None:
    rows = [
        ("Hydraulic solver", "Closed iterative pressure-dependent analysis; Darcy-Weisbach/Colebrook-White"),
        ("Source head", "Hourly independent realization in [20, 30] m; causal future expectation 25 m"),
        ("Emitter demand", "0.013 m3/s required flow; 11.4 m required pressure; linear PDA exponent 1"),
        ("Normal scenarios", "100; random 50 repeats/scenario; NSGA-II 5 repeats/scenario"),
        ("Single fault/recovery", "100 paired scenarios"),
        ("Multiple fault/recovery", "500 paired scenarios; 1-3 faults with partial or absent recovery"),
        ("Algorithms", "Tuned canonical NSGA-II, SPEA2, RVEA, MOEA/D"),
        ("Main FE budgets", "2,400 initial; 1,200 per dynamic reoptimization"),
        ("Pareto benchmark", "30 held-out contexts x 20 runs x 4 algorithms x 2,400 FE"),
        ("Parallelism", "Exactly 20 worker processes for formal scenario calculations"),
        ("Inference unit", "Scenario/context after within-scenario repeat aggregation"),
    ]
    pd.DataFrame(rows, columns=["setting", "value"]).to_csv(MAIN / "Table1_experimental_settings.csv", index=False, encoding="utf-8-sig")


def hydraulic_table() -> None:
    validation = pd.read_csv(DATA / "hydraulic_validation" / "pda_error_metrics_by_case.csv")
    table = validation.groupby(["case_class", "metric"]).agg(
        n_cases=("case_id", "nunique"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"),
        mape_mean_pct=("mape_pct", "mean"),
        maximum_absolute_error=("max_abs_error", "max"),
    ).reset_index()
    table.to_csv(MAIN / "Table2_closed_PDA_vs_EPANET.csv", index=False, encoding="utf-8-sig")


def main_policy_tables() -> None:
    policy = pd.read_csv(DATA / "main_experiments" / "revised_policy_metrics.csv")
    normal = policy[policy["experiment"].eq("normal")].copy()
    normal["method"] = normal["policy"].replace(
        {"original_near_far": "Original near-to-far", "random_rotation": "Random", "tuned_nsga2_causal": "Tuned NSGA-II"}
    )
    normal = normal.groupby(["scenario_id", "method"], as_index=False)[METRICS].mean()
    compact_summary(normal, ["method"], METRICS).to_csv(
        MAIN / "Table3_normal_operation_agricultural_metrics.csv", index=False, encoding="utf-8-sig"
    )

    fault = policy[policy["experiment"].isin(["single_fault", "multi_fault"])].copy()
    fault["decision"] = np.where(fault["policy"].str.contains("dynamic_reoptimization"), "Dynamic", "Static")
    fault = fault.groupby(["experiment", "scenario_id", "decision"], as_index=False)[
        METRICS + ["affected_target_met_pct", "affected_average_deficit_m3"]
    ].mean()
    compact_summary(
        fault,
        ["experiment", "decision"],
        METRICS + ["affected_target_met_pct", "affected_average_deficit_m3"],
    ).to_csv(MAIN / "Table4_dynamic_reoptimization_value.csv", index=False, encoding="utf-8-sig")

    dynamic = policy[
        policy["experiment"].isin(["single_fault", "multi_fault"])
        & policy["policy"].str.contains("dynamic_reoptimization")
    ].copy()
    agriculture = dynamic.groupby(["experiment", "scenario_id", "algorithm"], as_index=False)[METRICS].mean()
    quality = pd.read_csv(DATA / "algorithm_quality" / "pareto_quality_metrics.csv")
    quality = quality.groupby(["context_id", "algorithm"], as_index=False)[
        ["hv_4d", "igd_plus_4d", "additive_epsilon_4d", "nondominated_count", "spacing_4d", "elapsed_s"]
    ].mean()
    agriculture_summary = compact_summary(agriculture, ["experiment", "algorithm"], METRICS)
    quality_summary = compact_summary(
        quality,
        ["algorithm"],
        ["hv_4d", "igd_plus_4d", "additive_epsilon_4d", "nondominated_count", "spacing_4d", "elapsed_s"],
    )
    agriculture_summary.insert(0, "evidence", "agricultural outcome")
    quality_summary.insert(0, "evidence", "Pareto quality")
    pd.concat([agriculture_summary, quality_summary], ignore_index=True, sort=False).to_csv(
        MAIN / "Table5_algorithm_agriculture_and_Pareto_quality.csv", index=False, encoding="utf-8-sig"
    )


def cache_table() -> None:
    cache = pd.read_csv(DATA / "cache_scalability" / "cache_ablation_runs.csv")
    scenario = cache.groupby(["task_class", "scenario_id", "cache_state", "cache_level"], as_index=False).agg(
        elapsed_s=("elapsed_s", "mean"),
        peak_rss_mb=("peak_rss_delta_bytes", lambda values: float(np.mean(values) / 1024**2)),
        cache_storage_mb=("cache_estimated_bytes", lambda values: float(np.mean(values) / 1024**2)),
    )
    summary = compact_summary(scenario, ["cache_state", "cache_level"], ["elapsed_s", "peak_rss_mb", "cache_storage_mb"])
    timing = scenario.groupby(["cache_state", "cache_level"]).elapsed_s.mean().unstack("cache_level")
    speedup_rows = []
    for state, row in timing.iterrows():
        for level in row.index:
            speedup_rows.append(
                {"cache_state": state, "cache_level": level, "metric": "speedup_vs_level0", "mean": row[0.0] / row[level]}
            )
    pd.concat([summary, pd.DataFrame(speedup_rows)], ignore_index=True, sort=False).to_csv(
        MAIN / "Table6_cache_ablation_runtime_memory.csv", index=False, encoding="utf-8-sig"
    )


def supplementary_tables() -> None:
    copies = {
        DATA / "algorithm_tuning" / "algorithm_tuning_summary.csv": "TableS1_algorithm_parameter_tuning.csv",
        DATA / "paper_statistics" / "paired_wilcoxon_tests.csv": "TableS2_paired_Wilcoxon_Holm.csv",
        DATA / "paper_statistics" / "friedman_global_tests.csv": "TableS3_Friedman_global_tests.csv",
        DATA / "information_set" / "information_set_policy_metrics.csv": "TableS4_information_set_raw_scenarios.csv",
        DATA / "surrogate_grouped" / "surrogate_test_metrics.csv": "TableS5_surrogate_internal_external_prediction.csv",
        DATA / "surrogate_grouped" / "surrogate_model_tuning.csv": "TableS6_surrogate_parameter_tuning.csv",
    }
    for source, destination in copies.items():
        pd.read_csv(source).to_csv(SUPP / destination, index=False, encoding="utf-8-sig")

    scale = pd.read_csv(DATA / "cache_scalability" / "network_scalability_runs.csv")
    fit_rows = []
    for level, group in scale.groupby("cache_level"):
        means = group.groupby("emitter_count").elapsed_s.mean().reset_index()
        slope, intercept = np.polyfit(np.log(means["emitter_count"]), np.log(means["elapsed_s"]), 1)
        predicted = intercept + slope * np.log(means["emitter_count"])
        observed = np.log(means["elapsed_s"])
        r2 = 1.0 - np.sum((observed - predicted) ** 2) / np.sum((observed - observed.mean()) ** 2)
        fit_rows.append({"cache_level": level, "empirical_time_exponent": slope, "log_intercept": intercept, "r2": r2})
    pd.DataFrame(fit_rows).to_csv(SUPP / "TableS7_network_scalability_power_law.csv", index=False, encoding="utf-8-sig")
    scale.to_csv(SUPP / "TableS8_network_scalability_runs.csv", index=False, encoding="utf-8-sig")

    base = pd.read_csv(DATA / "main_experiments" / "revised_policy_metrics.csv")
    surrogate = pd.read_csv(DATA / "surrogate_experiments" / "surrogate_policy_metrics.csv")
    base = base[
        ((base["experiment"].eq("normal")) & base["policy"].eq("tuned_nsga2_causal"))
        | ((base["experiment"].isin(["single_fault", "multi_fault"])) & base["policy"].eq("nsga2_dynamic_reoptimization"))
    ].copy()
    base["method"] = "Exact NSGA-II"
    surrogate["method"] = "Surrogate-assisted NSGA-II"
    combined = pd.concat([base, surrogate], ignore_index=True, sort=False)
    scenario = combined.groupby(["experiment", "scenario_id", "method"], as_index=False)[METRICS].mean()
    scales = {
        "achievement_pct": 100.0,
        "unmet_emitter_ratio_pct": 100.0,
        "average_deficit_m3_per_emitter": 187.2,
        "max_deficit_m3": 187.2,
        "pressure_uniformity_index_pct": 100.0,
        "feasible_group_ratio_pct": 100.0,
    }
    equivalence_rows = []
    for experiment, block in scenario.groupby("experiment"):
        for metric in METRICS:
            pivot = block.pivot(index="scenario_id", columns="method", values=metric).dropna()
            difference = np.abs(pivot["Surrogate-assisted NSGA-II"] - pivot["Exact NSGA-II"])
            normalized = difference / scales[metric] * 100.0
            equivalence_rows.append(
                {
                    "experiment": experiment,
                    "metric": metric,
                    "n_scenarios": len(pivot),
                    "mean_absolute_difference": float(difference.mean()),
                    "maximum_absolute_difference": float(difference.max()),
                    "mean_normalized_difference_pct": float(normalized.mean()),
                    "maximum_normalized_difference_pct": float(normalized.max()),
                    "passes_maximum_5pct_equivalence": bool(normalized.max() < 5.0),
                }
            )
    pd.DataFrame(equivalence_rows).to_csv(SUPP / "TableS9_surrogate_equivalence_gate.csv", index=False, encoding="utf-8-sig")

    runtime = pd.DataFrame(
        [
            {
                "method": "Exact NSGA-II",
                "normal_mean_optimization_time_s": base[base["experiment"].eq("normal")]["optimization_time_s"].mean(),
                "fault_initial_mean_time_s": base[~base["experiment"].eq("normal")]["initial_optimization_time_s"].mean(),
                "fault_reoptimization_mean_total_time_s": base[~base["experiment"].eq("normal")]["total_reoptimization_time_s"].mean(),
                "normal_mean_exact_evaluations": base[base["experiment"].eq("normal")]["function_evaluations"].mean(),
            },
            {
                "method": "Surrogate-assisted NSGA-II",
                "normal_mean_optimization_time_s": surrogate[surrogate["experiment"].eq("normal")]["optimization_time_s"].mean(),
                "fault_initial_mean_time_s": surrogate[~surrogate["experiment"].eq("normal")]["initial_optimization_time_s"].mean(),
                "fault_reoptimization_mean_total_time_s": surrogate[~surrogate["experiment"].eq("normal")]["total_reoptimization_time_s"].mean(),
                "normal_mean_exact_evaluations": surrogate[surrogate["experiment"].eq("normal")]["function_evaluations"].mean(),
            },
        ]
    )
    runtime.to_csv(SUPP / "TableS10_surrogate_runtime_exact_evaluations.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    MAIN.mkdir(parents=True, exist_ok=True)
    SUPP.mkdir(parents=True, exist_ok=True)
    write_settings()
    hydraulic_table()
    main_policy_tables()
    cache_table()
    supplementary_tables()
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "main_table_count": len(list(MAIN.glob("*.csv"))),
        "supplementary_table_count": len(list(SUPP.glob("*.csv"))),
        "main_table_limit": 6,
    }
    (DATA / "paper_tables" / "table_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
