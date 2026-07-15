from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

import run_revised_main_experiments as main_experiments
from revision_core import DEFAULT_WORKERS, Event, HydraulicParams, Network, evaluate_plan, load_network, multi_fault_scenario, single_fault_scenario, source_profile
from revised_optimization import OptimizationResult, forecast_for_information_mode
from surrogate_optimization import optimize_surrogate_assisted_nsga2


NETWORK: Network | None = None
PARAMS: HydraulicParams | None = None
MODEL: Any = None
MODEL_FAMILY = ""


def init_worker(nodes_path: str, pipes_path: str, data_dir: str) -> None:
    global NETWORK, PARAMS, MODEL, MODEL_FAMILY
    NETWORK = load_network(Path(nodes_path), Path(pipes_path))
    PARAMS = HydraulicParams()
    training = json.loads((Path(data_dir) / "surrogate_training_metadata.json").read_text(encoding="utf-8"))
    MODEL_FAMILY = str(training["selected_family"])
    MODEL = joblib.load(Path(data_dir) / f"best_{MODEL_FAMILY}_surrogate.joblib")
    if hasattr(MODEL, "set_params") and MODEL_FAMILY == "catboost":
        MODEL.set_params(thread_count=1)
    if hasattr(MODEL, "n_jobs"):
        MODEL.n_jobs = 1
    for estimator in getattr(MODEL, "estimators_", []):
        if hasattr(estimator, "set_params"):
            estimator.set_params(n_jobs=1)
    main_experiments.NETWORK = NETWORK
    main_experiments.PARAMS = PARAMS
    main_experiments.optimize_current_state = optimize_current_state


def optimize_current_state(
    network: Network,
    params: HydraulicParams,
    algorithm: str,
    realized_profile: Sequence[float],
    information_mode: str,
    candidates: list[int],
    unavailable: set[int],
    time_h: float,
    volumes: np.ndarray,
    seed: int,
    evaluation_budget: int,
) -> OptimizationResult:
    forecast = forecast_for_information_mode(realized_profile, time_h, information_mode)
    return optimize_surrogate_assisted_nsga2(
        network,
        params,
        candidates,
        forecast,
        unavailable,
        time_h,
        volumes,
        params.horizon_h,
        MODEL,
        seed,
        evaluation_budget,
        population_size=48,
        exact_fraction=0.35,
        exploration_fraction=0.20,
        exact_final_generations=4,
        cache_level=3,
    )


def normal_task(task: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if NETWORK is None or PARAMS is None:
        raise RuntimeError("worker not initialized")
    network, params = NETWORK, PARAMS
    scenario_id = int(task["scenario_id"])
    seed = int(task["seed"])
    profile = source_profile(seed, params)
    rows: list[dict[str, Any]] = []
    fronts: list[dict[str, Any]] = []
    convergence: list[dict[str, Any]] = []
    for repeat in range(1, int(task["repeats"]) + 1):
        result = optimize_current_state(
            network,
            params,
            "surrogate_nsga2",
            profile,
            "causal_mean",
            list(range(len(network.emitters))),
            set(),
            0.0,
            np.zeros(len(network.emitters)),
            seed + repeat * 1009,
            int(task["initial_budget"]),
        )
        metrics, _, _ = evaluate_plan(network, params, result.plan, profile)
        rows.append(
            {
                **metrics,
                "experiment": "normal",
                "scenario_id": scenario_id,
                "policy": "surrogate_assisted_nsga2_causal",
                "algorithm": "surrogate_nsga2",
                "repeat": repeat,
                "seed": seed + repeat * 1009,
                "information_mode": "causal_mean",
                "surrogate_model": MODEL_FAMILY,
                "optimization_time_s": result.elapsed_s,
                "function_evaluations": result.function_evaluations,
                "reference_candidate_evaluations": result.selected_metrics["surrogate_reference_evaluations"],
                "exact_evaluation_reduction_pct": result.selected_metrics["surrogate_exact_reduction_pct"],
                "selected_hydraulic_cv": result.selected_violation,
            }
        )
        new_fronts, new_convergence = main_experiments.optimization_artifacts(
            result, "normal", scenario_id, "initial", repeat
        )
        fronts.extend(new_fronts)
        convergence.extend(new_convergence)
    return rows, fronts, convergence


def fault_task(task: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if NETWORK is None or PARAMS is None:
        raise RuntimeError("worker not initialized")
    network, params = NETWORK, PARAMS
    experiment = str(task["experiment"])
    scenario_id = int(task["scenario_id"])
    seed = int(task["seed"])
    profile = source_profile(seed, params)
    events: list[Event] = (
        single_fault_scenario(seed + 333, network, params)
        if experiment == "single_fault"
        else multi_fault_scenario(seed + 777, network, params)
    )
    initial = optimize_current_state(
        network,
        params,
        "surrogate_nsga2",
        profile,
        "causal_mean",
        list(range(len(network.emitters))),
        set(),
        0.0,
        np.zeros(len(network.emitters)),
        seed + 10_009,
        int(task["initial_budget"]),
    )
    fronts, convergence = main_experiments.optimization_artifacts(
        initial, experiment, scenario_id, "initial", 1
    )
    metrics, records, new_fronts, new_convergence = main_experiments.simulate_with_events(
        network,
        params,
        initial.plan,
        profile,
        events,
        True,
        "surrogate_nsga2",
        "causal_mean",
        seed + 50_000,
        int(task["reoptimization_budget"]),
        experiment,
        scenario_id,
    )
    fronts.extend(new_fronts)
    convergence.extend(new_convergence)
    event_text = ";".join(f"{event.kind}:{event.node}@{event.time_h:.6f}" for event in events)
    row = {
        **metrics,
        "experiment": experiment,
        "scenario_id": scenario_id,
        "policy": "surrogate_nsga2_dynamic_reoptimization",
        "algorithm": "surrogate_nsga2",
        "repeat": 1,
        "seed": seed,
        "information_mode": "causal_mean",
        "surrogate_model": MODEL_FAMILY,
        "initial_optimization_time_s": initial.elapsed_s,
        "initial_function_evaluations": initial.function_evaluations,
        "initial_reference_candidate_evaluations": initial.selected_metrics["surrogate_reference_evaluations"],
        "initial_exact_evaluation_reduction_pct": initial.selected_metrics["surrogate_exact_reduction_pct"],
        "event_count": len(events),
        "fault_count": sum(event.kind == "fault" for event in events),
        "recovery_count": sum(event.kind == "recovery" for event in events),
        "events": event_text,
    }
    return [row], records, fronts, convergence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Surrogate-assisted NSGA-II experiments with exact final hydraulics")
    repository_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--nodes", type=Path, default=repository_root / "data" / "network" / "Nodes.xlsx")
    parser.add_argument("--pipes", type=Path, default=repository_root / "data" / "network" / "Pipes.xlsx")
    parser.add_argument("--surrogate-data", type=Path, default=repository_root / "data" / "surrogate_grouped")
    parser.add_argument("--output-dir", type=Path, default=repository_root / "outputs" / "surrogate_experiments")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--normal-scenarios", type=int, default=100)
    parser.add_argument("--single-fault-scenarios", type=int, default=100)
    parser.add_argument("--multi-fault-scenarios", type=int, default=500)
    parser.add_argument("--normal-repeats", type=int, default=5)
    parser.add_argument("--initial-budget", type=int, default=2400)
    parser.add_argument("--reoptimization-budget", type=int, default=1200)
    parser.add_argument("--base-seed", type=int, default=2026071600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers != DEFAULT_WORKERS:
        raise ValueError(f"all formal parallel calculations require exactly {DEFAULT_WORKERS} workers")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[str, dict[str, Any]]] = []
    tasks.extend(
        (
            "normal",
            {
                "scenario_id": scenario_id,
                "seed": args.base_seed + scenario_id,
                "repeats": args.normal_repeats,
                "initial_budget": args.initial_budget,
            },
        )
        for scenario_id in range(1, args.normal_scenarios + 1)
    )
    for experiment, count, offset in [
        ("single_fault", args.single_fault_scenarios, 100_000),
        ("multi_fault", args.multi_fault_scenarios, 200_000),
    ]:
        tasks.extend(
            (
                "fault",
                {
                    "experiment": experiment,
                    "scenario_id": scenario_id,
                    "seed": args.base_seed + offset + scenario_id,
                    "initial_budget": args.initial_budget,
                    "reoptimization_budget": args.reoptimization_budget,
                },
            )
            for scenario_id in range(1, count + 1)
        )
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    fronts: list[dict[str, Any]] = []
    convergence: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=DEFAULT_WORKERS,
        initializer=init_worker,
        initargs=(str(args.nodes), str(args.pipes), str(args.surrogate_data)),
    ) as executor:
        futures = [
            executor.submit(normal_task if kind == "normal" else fault_task, task)
            for kind, task in tasks
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if len(result) == 3:
                new_rows, new_fronts, new_convergence = result
                new_records: list[dict[str, Any]] = []
            else:
                new_rows, new_records, new_fronts, new_convergence = result
            rows.extend(new_rows)
            records.extend(new_records)
            fronts.extend(new_fronts)
            convergence.extend(new_convergence)
            if completed == 1 or completed % 25 == 0 or completed == len(futures):
                print(f"Surrogate experiments {completed}/{len(futures)}; elapsed={(time.perf_counter() - started) / 60:.1f} min", flush=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "surrogate_policy_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(records).to_parquet(args.output_dir / "surrogate_execution_records.parquet", index=False)
    pd.DataFrame(fronts).to_parquet(args.output_dir / "surrogate_pareto_fronts.parquet", index=False)
    pd.DataFrame(convergence).to_parquet(args.output_dir / "surrogate_convergence_traces.parquet", index=False)
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.replace("\n", " "),
        "workers": DEFAULT_WORKERS,
        "logical_cpu_count": os.cpu_count(),
        "normal_scenarios": args.normal_scenarios,
        "normal_repeats": args.normal_repeats,
        "single_fault_scenarios": args.single_fault_scenarios,
        "multi_fault_scenarios": args.multi_fault_scenarios,
        "initial_reference_budget": args.initial_budget,
        "reoptimization_reference_budget": args.reoptimization_budget,
        "final_metric_source": "closed iterative PDA exact hydraulic model",
        "wall_time_s": time.perf_counter() - started,
    }
    (args.output_dir / "surrogate_experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
