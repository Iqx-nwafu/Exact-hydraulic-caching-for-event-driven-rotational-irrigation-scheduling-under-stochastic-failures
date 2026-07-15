from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV

from revision_core import (
    DEFAULT_WORKERS,
    HydraulicParams,
    evaluate_plan,
    isolated_emitters_after_fault,
    load_network,
    original_plan,
    source_profile,
)
from revised_optimization import (
    AlgorithmConfig,
    forecast_for_information_mode,
    normalized_objectives,
    optimize_plan,
)


NETWORK: Any = None
PARAMS: HydraulicParams | None = None


def init_worker(nodes_path: str, pipes_path: str) -> None:
    global NETWORK, PARAMS
    NETWORK = load_network(Path(nodes_path), Path(pipes_path))
    PARAMS = HydraulicParams()


def configuration_grid() -> dict[str, list[AlgorithmConfig]]:
    return {
        "nsga2": [
            AlgorithmConfig("nsga2", 40, 0.90, 10, 1.0, 15),
            AlgorithmConfig("nsga2", 40, 1.00, 20, 2.0, 20),
            AlgorithmConfig("nsga2", 48, 0.90, 15, 1.0, 20),
            AlgorithmConfig("nsga2", 48, 1.00, 20, 1.0, 30),
            AlgorithmConfig("nsga2", 48, 0.90, 30, 2.0, 20),
            AlgorithmConfig("nsga2", 64, 0.90, 20, 1.0, 20),
            AlgorithmConfig("nsga2", 64, 1.00, 30, 2.0, 30),
            AlgorithmConfig("nsga2", 64, 0.90, 15, 2.0, 40),
        ],
        "spea2": [
            AlgorithmConfig("spea2", 40, 0.90, 10, 1.0, 15, archive_size=40),
            AlgorithmConfig("spea2", 40, 1.00, 20, 2.0, 20, archive_size=40),
            AlgorithmConfig("spea2", 48, 0.90, 15, 1.0, 20, archive_size=48),
            AlgorithmConfig("spea2", 48, 1.00, 20, 1.0, 30, archive_size=48),
            AlgorithmConfig("spea2", 48, 0.90, 30, 2.0, 20, archive_size=48),
            AlgorithmConfig("spea2", 64, 0.90, 20, 1.0, 20, archive_size=64),
            AlgorithmConfig("spea2", 64, 1.00, 30, 2.0, 30, archive_size=64),
            AlgorithmConfig("spea2", 64, 0.90, 15, 2.0, 40, archive_size=64),
        ],
        "rvea": [
            AlgorithmConfig("rvea", 40, 0.90, 20, 1.0, 20, rvea_alpha=1.0, rvea_adapt_frequency=0.10),
            AlgorithmConfig("rvea", 40, 1.00, 30, 2.0, 30, rvea_alpha=2.0, rvea_adapt_frequency=0.20),
            AlgorithmConfig("rvea", 48, 0.90, 20, 1.0, 20, rvea_alpha=1.0, rvea_adapt_frequency=0.20),
            AlgorithmConfig("rvea", 48, 1.00, 30, 1.0, 30, rvea_alpha=2.0, rvea_adapt_frequency=0.10),
            AlgorithmConfig("rvea", 48, 0.90, 15, 2.0, 20, rvea_alpha=3.0, rvea_adapt_frequency=0.10),
            AlgorithmConfig("rvea", 64, 0.90, 20, 1.0, 20, rvea_alpha=2.0, rvea_adapt_frequency=0.10),
            AlgorithmConfig("rvea", 64, 1.00, 30, 2.0, 30, rvea_alpha=3.0, rvea_adapt_frequency=0.20),
            AlgorithmConfig("rvea", 64, 0.90, 15, 2.0, 40, rvea_alpha=4.0, rvea_adapt_frequency=0.10),
        ],
        "moead": [
            AlgorithmConfig("moead", 40, 0.90, 20, 1.0, 20, moead_neighbors=10, moead_neighbor_probability=0.70, moead_pbi_theta=3.0),
            AlgorithmConfig("moead", 40, 1.00, 30, 2.0, 30, moead_neighbors=20, moead_neighbor_probability=0.90, moead_pbi_theta=5.0),
            AlgorithmConfig("moead", 48, 0.90, 20, 1.0, 20, moead_neighbors=10, moead_neighbor_probability=0.90, moead_pbi_theta=3.0),
            AlgorithmConfig("moead", 48, 1.00, 30, 1.0, 30, moead_neighbors=20, moead_neighbor_probability=0.70, moead_pbi_theta=5.0),
            AlgorithmConfig("moead", 48, 0.90, 15, 2.0, 20, moead_neighbors=30, moead_neighbor_probability=0.90, moead_pbi_theta=5.0),
            AlgorithmConfig("moead", 64, 0.90, 20, 1.0, 20, moead_neighbors=10, moead_neighbor_probability=0.70, moead_pbi_theta=5.0),
            AlgorithmConfig("moead", 64, 1.00, 30, 2.0, 30, moead_neighbors=20, moead_neighbor_probability=0.90, moead_pbi_theta=3.0),
            AlgorithmConfig("moead", 64, 0.90, 15, 2.0, 40, moead_neighbors=30, moead_neighbor_probability=0.90, moead_pbi_theta=5.0),
        ],
    }


def build_context(context_id: int, base_seed: int) -> dict[str, Any]:
    if NETWORK is None or PARAMS is None:
        raise RuntimeError("worker not initialized")
    network, params = NETWORK, PARAMS
    seed = base_seed + context_id * 1009
    profile = source_profile(seed, params)
    phase_index = (context_id - 1) // 4
    phase = ("normal", "post_fault", "post_recovery")[phase_index]
    start_time = 0.0 if phase == "normal" else float(24 + 8 * ((context_id - 1) % 4))
    start_volumes = np.zeros(len(network.emitters), dtype=float)
    unavailable: set[int] = set()
    fault_node = ""
    if start_time > 0:
        groups_to_execute = int(math.ceil(start_time / params.target_duration_h))
        prefix = original_plan(network, params)[:groups_to_execute]
        _, start_volumes, _ = evaluate_plan(
            network,
            params,
            prefix,
            profile,
            horizon_h=start_time,
        )
        fault_node = network.irrigation_nodes[(context_id * 7) % len(network.irrigation_nodes)]
        affected = isolated_emitters_after_fault(network, fault_node)
        if phase == "post_fault":
            unavailable = affected
        else:
            # Recovered emitters retain only water delivered before isolation.
            start_volumes[list(affected)] *= 0.25
    candidates = [
        index
        for index in range(len(network.emitters))
        if index not in unavailable and start_volumes[index] < params.target_per_emitter_m3 - 1.0e-9
    ]
    forecast = forecast_for_information_mode(profile, start_time, "causal_mean")
    return {
        "context_id": context_id,
        "phase": phase,
        "scenario_seed": seed,
        "source_profile": forecast,
        "unavailable": unavailable,
        "start_time_h": start_time,
        "start_volumes": start_volumes,
        "candidates": candidates,
        "fault_node": fault_node,
    }


def run_tuning_task(task: dict[str, Any]) -> dict[str, Any]:
    if NETWORK is None or PARAMS is None:
        raise RuntimeError("worker not initialized")
    context = build_context(int(task["context_id"]), int(task["base_seed"]))
    config = AlgorithmConfig(**task["config"])
    result = optimize_plan(
        NETWORK,
        PARAMS,
        context["candidates"],
        context["source_profile"],
        context["unavailable"],
        context["start_time_h"],
        context["start_volumes"],
        PARAMS.horizon_h,
        config,
        int(task["optimizer_seed"]),
        int(task["evaluation_budget"]),
        cache_level=3,
    )
    front = normalized_objectives(
        result.population_objectives[result.nondominated_indices],
        result.population_violations[result.nondominated_indices],
    )
    hypervolume = float(HV(ref_point=np.array([1.1, 1.1, 1.1, 1.1]))(front))
    row = {
        "algorithm": config.algorithm,
        "config_id": task["config_id"],
        "context_id": context["context_id"],
        "phase": context["phase"],
        "replicate": task["replicate"],
        "optimizer_seed": task["optimizer_seed"],
        "evaluation_budget": task["evaluation_budget"],
        "actual_function_evaluations": result.function_evaluations,
        "hypervolume_4d": hypervolume,
        "selected_hydraulic_cv": result.selected_violation,
        "minimum_population_cv": float(np.min(result.population_violations)),
        "nondominated_count": len(result.nondominated_indices),
        "elapsed_s": result.elapsed_s,
        "achievement_pct": result.selected_metrics["achievement_pct"],
        "average_deficit_m3_per_emitter": result.selected_metrics["average_deficit_m3_per_emitter"],
        "pressure_uniformity_index_pct": result.selected_metrics["pressure_uniformity_index_pct"],
        "feasible_group_ratio_pct": result.selected_metrics["feasible_group_ratio_pct"],
    }
    row.update({f"param_{key}": value for key, value in asdict(config).items()})
    return row


def select_best_configs(results: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    ranked = results.copy()
    ranked["hv_rank_within_block"] = ranked.groupby(
        ["algorithm", "context_id", "replicate"]
    )["hypervolume_4d"].rank(method="average", ascending=False)
    summary = (
        ranked.groupby(["algorithm", "config_id"], as_index=False)
        .agg(
            mean_hypervolume_4d=("hypervolume_4d", "mean"),
            std_hypervolume_4d=("hypervolume_4d", "std"),
            median_hv_rank=("hv_rank_within_block", "median"),
            mean_hv_rank=("hv_rank_within_block", "mean"),
            mean_selected_cv=("selected_hydraulic_cv", "mean"),
            mean_elapsed_s=("elapsed_s", "mean"),
            run_count=("hypervolume_4d", "size"),
        )
    )
    best: dict[str, dict[str, Any]] = {}
    grid = configuration_grid()
    for algorithm, group in summary.groupby("algorithm"):
        chosen = group.sort_values(
            ["median_hv_rank", "mean_hv_rank", "mean_hypervolume_4d", "mean_selected_cv", "mean_elapsed_s"],
            ascending=[True, True, False, True, True],
        ).iloc[0]
        config_index = int(str(chosen["config_id"]).split("_")[-1])
        config = grid[algorithm][config_index]
        best[algorithm] = {
            "selection_rule": "lowest median paired HV rank, then mean rank, higher mean HV, lower CV, lower time",
            "config_id": chosen["config_id"],
            "config": asdict(config),
            "tuning_mean_hypervolume_4d": float(chosen["mean_hypervolume_4d"]),
            "tuning_mean_hv_rank": float(chosen["mean_hv_rank"]),
        }
    return summary, best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic parameter tuning for four canonical multi-objective algorithms")
    repository_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--nodes", type=Path, default=repository_root / "data" / "network" / "Nodes.xlsx")
    parser.add_argument("--pipes", type=Path, default=repository_root / "data" / "network" / "Pipes.xlsx")
    parser.add_argument("--output-dir", type=Path, default=repository_root / "outputs" / "algorithm_tuning")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--contexts", type=int, default=12)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--evaluation-budget", type=int, default=960)
    parser.add_argument("--base-seed", type=int, default=2026071500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers != DEFAULT_WORKERS:
        raise ValueError(f"all formal parallel calculations require exactly {DEFAULT_WORKERS} workers")
    if args.contexts != 12:
        raise ValueError("the preregistered tuning design uses 12 contexts: 4 normal, 4 fault, 4 recovery")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, Any]] = []
    for algorithm, configs in configuration_grid().items():
        for config_index, config in enumerate(configs):
            config_id = f"{algorithm}_{config_index}"
            for context_id in range(1, args.contexts + 1):
                for replicate in range(1, args.replicates + 1):
                    tasks.append(
                        {
                            "algorithm": algorithm,
                            "config_id": config_id,
                            "config": asdict(config),
                            "context_id": context_id,
                            "replicate": replicate,
                            "base_seed": args.base_seed,
                            "optimizer_seed": args.base_seed + 10_000_000 + context_id * 10_000 + replicate * 100 + config_index,
                            "evaluation_budget": args.evaluation_budget,
                        }
                    )
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=DEFAULT_WORKERS,
        initializer=init_worker,
        initargs=(str(args.nodes), str(args.pipes)),
    ) as executor:
        futures = [executor.submit(run_tuning_task, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed == 1 or completed % 50 == 0 or completed == len(futures):
                print(
                    f"Tuning {completed}/{len(futures)}; elapsed={(time.perf_counter() - started) / 60:.1f} min",
                    flush=True,
                )
    results = pd.DataFrame(rows)
    summary, best = select_best_configs(results)
    results.to_csv(args.output_dir / "algorithm_tuning_runs.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "algorithm_tuning_summary.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "best_algorithm_parameters.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.replace("\n", " "),
        "workers": DEFAULT_WORKERS,
        "logical_cpu_count": os.cpu_count(),
        "contexts": args.contexts,
        "context_composition": {"normal": 4, "post_fault": 4, "post_recovery": 4},
        "replicates_per_context": args.replicates,
        "candidate_configs_per_algorithm": 8,
        "evaluation_budget": args.evaluation_budget,
        "total_tasks": len(tasks),
        "wall_time_s": time.perf_counter() - started,
        "test_scenarios_used_for_tuning": False,
    }
    (args.output_dir / "algorithm_tuning_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(best, ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
