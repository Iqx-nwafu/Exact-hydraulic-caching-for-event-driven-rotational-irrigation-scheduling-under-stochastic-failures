from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus

from revision_core import DEFAULT_WORKERS, HydraulicParams, evaluate_plan, isolated_emitters_after_fault, load_network, original_plan, source_profile
from revised_optimization import AlgorithmConfig, forecast_for_information_mode, nondominated_indices, normalized_objectives, optimize_plan


NETWORK: Any = None
PARAMS: HydraulicParams | None = None
CONFIGS: dict[str, AlgorithmConfig] = {}


def init_worker(nodes_path: str, pipes_path: str, best_config_path: str) -> None:
    global NETWORK, PARAMS, CONFIGS
    NETWORK = load_network(Path(nodes_path), Path(pipes_path))
    PARAMS = HydraulicParams()
    data = json.loads(Path(best_config_path).read_text(encoding="utf-8"))
    CONFIGS = {algorithm: AlgorithmConfig(**entry["config"]) for algorithm, entry in data.items()}


def heldout_context(context_id: int, base_seed: int) -> dict[str, Any]:
    if NETWORK is None or PARAMS is None:
        raise RuntimeError("worker not initialized")
    network, params = NETWORK, PARAMS
    seed = base_seed + context_id * 7919
    profile = source_profile(seed, params)
    if context_id <= 10:
        phase = "normal"
        start_time = 0.0
    elif context_id <= 20:
        phase = "post_fault"
        start_time = float(20 + (context_id - 11) * 6)
    else:
        phase = "post_recovery"
        start_time = float(32 + (context_id - 21) * 5)
    volumes = np.zeros(len(network.emitters), dtype=float)
    unavailable: set[int] = set()
    if start_time > 0:
        prefix_count = int(math.ceil(start_time / params.target_duration_h))
        _, volumes, _ = evaluate_plan(
            network,
            params,
            original_plan(network, params)[:prefix_count],
            profile,
            horizon_h=start_time,
        )
        fault_node = network.irrigation_nodes[(context_id * 13) % len(network.irrigation_nodes)]
        affected = isolated_emitters_after_fault(network, fault_node)
        if phase == "post_fault":
            unavailable = affected
        else:
            volumes[list(affected)] *= 0.2
    candidates = [
        index
        for index in range(len(network.emitters))
        if index not in unavailable and volumes[index] < params.target_per_emitter_m3 - 1.0e-9
    ]
    return {
        "context_id": context_id,
        "phase": phase,
        "seed": seed,
        "profile": forecast_for_information_mode(profile, start_time, "causal_mean"),
        "start_time_h": start_time,
        "volumes": volumes,
        "unavailable": unavailable,
        "candidates": candidates,
    }


def run_task(task: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if NETWORK is None or PARAMS is None:
        raise RuntimeError("worker not initialized")
    context = heldout_context(int(task["context_id"]), int(task["base_seed"]))
    algorithm = str(task["algorithm"])
    result = optimize_plan(
        NETWORK,
        PARAMS,
        context["candidates"],
        context["profile"],
        context["unavailable"],
        context["start_time_h"],
        context["volumes"],
        PARAMS.horizon_h,
        CONFIGS[algorithm],
        int(task["optimizer_seed"]),
        int(task["evaluation_budget"]),
        cache_level=3,
    )
    front_rows = []
    for point_id, population_index in enumerate(result.nondominated_indices):
        objective = result.population_objectives[population_index]
        front_rows.append(
            {
                "context_id": context["context_id"],
                "phase": context["phase"],
                "algorithm": algorithm,
                "replicate": task["replicate"],
                "point_id": point_id,
                "achievement_loss": float(objective[0]),
                "average_deficit_m3_per_emitter": float(objective[1]),
                "pressure_nonuniformity": float(objective[2]),
                "hydraulic_cv": float(result.population_violations[population_index]),
            }
        )
    traces = [
        {
            **row,
            "context_id": context["context_id"],
            "phase": context["phase"],
            "algorithm": algorithm,
            "replicate": task["replicate"],
        }
        for row in result.convergence
    ]
    run = {
        "context_id": context["context_id"],
        "phase": context["phase"],
        "algorithm": algorithm,
        "replicate": task["replicate"],
        "optimizer_seed": task["optimizer_seed"],
        "requested_evaluation_budget": task["evaluation_budget"],
        "actual_function_evaluations": result.function_evaluations,
        "elapsed_s": result.elapsed_s,
        "selected_achievement_pct": result.selected_metrics["achievement_pct"],
        "selected_average_deficit": result.selected_metrics["average_deficit_m3_per_emitter"],
        "selected_pressure_uniformity": result.selected_metrics["pressure_uniformity_index_pct"],
        "selected_feasible_group_ratio": result.selected_metrics["feasible_group_ratio_pct"],
        "selected_hydraulic_cv": result.selected_violation,
    }
    return front_rows, traces, run


def additive_epsilon(reference: np.ndarray, approximation: np.ndarray) -> float:
    values = []
    for reference_point in reference:
        values.append(np.min(np.max(approximation - reference_point, axis=1)))
    return float(np.max(values))


def spacing_indicator(front: np.ndarray) -> float:
    if len(front) < 2:
        return 0.0
    distances = np.full(len(front), np.inf)
    for index in range(len(front)):
        difference = np.abs(front - front[index]).sum(axis=1)
        difference[index] = np.inf
        distances[index] = np.min(difference)
    return float(np.std(distances, ddof=1)) if len(distances) > 1 else 0.0


def compute_quality(fronts: pd.DataFrame, runs: pd.DataFrame, traces: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    quality_rows: list[dict[str, Any]] = []
    for context_id, context_fronts in fronts.groupby("context_id"):
        pooled_raw = context_fronts[["achievement_loss", "average_deficit_m3_per_emitter", "pressure_nonuniformity"]].to_numpy()
        pooled_cv = context_fronts["hydraulic_cv"].to_numpy()
        pooled = normalized_objectives(pooled_raw, pooled_cv)
        reference = pooled[nondominated_indices(pooled)]
        for (algorithm, replicate), group in context_fronts.groupby(["algorithm", "replicate"]):
            raw = group[["achievement_loss", "average_deficit_m3_per_emitter", "pressure_nonuniformity"]].to_numpy()
            approximation = normalized_objectives(raw, group["hydraulic_cv"].to_numpy())
            approximation = approximation[nondominated_indices(approximation)]
            run = runs[
                runs["context_id"].eq(context_id)
                & runs["algorithm"].eq(algorithm)
                & runs["replicate"].eq(replicate)
            ].iloc[0]
            quality_rows.append(
                {
                    "context_id": context_id,
                    "phase": run["phase"],
                    "algorithm": algorithm,
                    "replicate": replicate,
                    "hv_4d": float(HV(ref_point=np.array([1.1, 1.1, 1.1, 1.1]))(approximation)),
                    "igd_plus_4d": float(IGDPlus(reference)(approximation)),
                    "additive_epsilon_4d": additive_epsilon(reference, approximation),
                    "nondominated_count": len(approximation),
                    "spacing_4d": spacing_indicator(approximation),
                    "elapsed_s": run["elapsed_s"],
                    "function_evaluations": run["actual_function_evaluations"],
                }
            )
    quality = pd.DataFrame(quality_rows)

    equal_time_rows: list[dict[str, Any]] = []
    for (context_id, replicate), block in traces.groupby(["context_id", "replicate"]):
        max_times = block.groupby("algorithm")["elapsed_s"].max()
        if len(max_times) < 4:
            continue
        common_time = float(max_times.min())
        for algorithm, algorithm_trace in block.groupby("algorithm"):
            eligible = algorithm_trace[algorithm_trace["elapsed_s"] <= common_time + 1.0e-12]
            selected = eligible.sort_values("elapsed_s").iloc[-1] if not eligible.empty else algorithm_trace.sort_values("elapsed_s").iloc[0]
            equal_time_rows.append(
                {
                    "context_id": context_id,
                    "phase": selected["phase"],
                    "replicate": replicate,
                    "algorithm": algorithm,
                    "common_wall_time_s": common_time,
                    "observed_elapsed_s": selected["elapsed_s"],
                    "function_evaluations_at_common_time": selected["function_evaluations"],
                    "hv_4d_at_common_time": selected["hypervolume_4d"],
                }
            )
    return quality, pd.DataFrame(equal_time_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Held-out Pareto quality benchmark for tuned algorithms")
    repository_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--nodes", type=Path, default=repository_root / "data" / "network" / "Nodes.xlsx")
    parser.add_argument("--pipes", type=Path, default=repository_root / "data" / "network" / "Pipes.xlsx")
    parser.add_argument("--best-config", type=Path, default=repository_root / "data" / "algorithm_tuning" / "best_algorithm_parameters.json")
    parser.add_argument("--output-dir", type=Path, default=repository_root / "outputs" / "algorithm_quality")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--contexts", type=int, default=30)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--evaluation-budget", type=int, default=2400)
    parser.add_argument("--base-seed", type=int, default=2026071900)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers != DEFAULT_WORKERS:
        raise ValueError(f"all formal parallel calculations require exactly {DEFAULT_WORKERS} workers")
    if args.contexts != 30:
        raise ValueError("the held-out design uses 30 contexts: 10 normal, 10 fault, 10 recovery")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    algorithms = ("nsga2", "spea2", "rvea", "moead")
    tasks = [
        {
            "context_id": context_id,
            "algorithm": algorithm,
            "replicate": replicate,
            "optimizer_seed": args.base_seed + context_id * 100_000 + replicate * 1000 + algorithm_index,
            "base_seed": args.base_seed,
            "evaluation_budget": args.evaluation_budget,
        }
        for context_id in range(1, args.contexts + 1)
        for replicate in range(1, args.replicates + 1)
        for algorithm_index, algorithm in enumerate(algorithms)
    ]
    started = time.perf_counter()
    front_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=DEFAULT_WORKERS,
        initializer=init_worker,
        initargs=(str(args.nodes), str(args.pipes), str(args.best_config)),
    ) as executor:
        futures = [executor.submit(run_task, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            fronts, traces, run = future.result()
            front_rows.extend(fronts)
            trace_rows.extend(traces)
            run_rows.append(run)
            if completed == 1 or completed % 50 == 0 or completed == len(futures):
                print(f"Algorithm quality {completed}/{len(futures)}; elapsed={(time.perf_counter() - started) / 60:.1f} min", flush=True)
    fronts = pd.DataFrame(front_rows)
    traces = pd.DataFrame(trace_rows)
    runs = pd.DataFrame(run_rows)
    quality, equal_time = compute_quality(fronts, runs, traces)
    fronts.to_parquet(args.output_dir / "heldout_pareto_fronts.parquet", index=False)
    traces.to_parquet(args.output_dir / "heldout_convergence_traces.parquet", index=False)
    runs.to_csv(args.output_dir / "heldout_algorithm_runs.csv", index=False, encoding="utf-8-sig")
    quality.to_csv(args.output_dir / "pareto_quality_metrics.csv", index=False, encoding="utf-8-sig")
    equal_time.to_csv(args.output_dir / "equal_wall_time_hv.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.replace("\n", " "),
        "workers": DEFAULT_WORKERS,
        "logical_cpu_count": os.cpu_count(),
        "heldout_contexts": args.contexts,
        "context_composition": {"normal": 10, "post_fault": 10, "post_recovery": 10},
        "independent_replicates": args.replicates,
        "algorithms": list(algorithms),
        "equal_function_evaluation_budget": args.evaluation_budget,
        "empirical_reference_front": "non-dominated union of all algorithms and replicates within each held-out context",
        "equal_wall_time_rule": "minimum terminal wall time across algorithms within each paired context and replicate",
        "wall_time_s": time.perf_counter() - started,
        "run_count": len(runs),
    }
    (args.output_dir / "algorithm_quality_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
