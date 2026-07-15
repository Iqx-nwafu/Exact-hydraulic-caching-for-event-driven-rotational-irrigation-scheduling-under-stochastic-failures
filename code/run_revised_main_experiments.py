from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from revision_core import (
    DEFAULT_WORKERS,
    EPS,
    CacheConfig,
    EvaluationCaches,
    Event,
    Group,
    HydraulicParams,
    Network,
    evaluate_plan,
    isolated_emitters_after_fault,
    load_network,
    multi_fault_scenario,
    original_plan,
    random_plan,
    single_fault_scenario,
    solve_group_pda,
    source_head_at,
    source_profile,
    unavailable_from_faults,
)
from revised_optimization import AlgorithmConfig, OptimizationResult, forecast_for_information_mode, optimize_plan


NETWORK: Network | None = None
PARAMS: HydraulicParams | None = None
BEST_CONFIGS: dict[str, AlgorithmConfig] = {}


def init_worker(nodes_path: str, pipes_path: str, best_config_path: str) -> None:
    global NETWORK, PARAMS, BEST_CONFIGS
    NETWORK = load_network(Path(nodes_path), Path(pipes_path))
    PARAMS = HydraulicParams()
    data = json.loads(Path(best_config_path).read_text(encoding="utf-8"))
    BEST_CONFIGS = {
        algorithm: AlgorithmConfig(**entry["config"])
        for algorithm, entry in data.items()
    }


def weighted_pressure_metrics(values: list[float], weights: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    values_array = np.asarray(values, dtype=float)
    weights_array = np.asarray(weights, dtype=float)
    mean = float(np.average(values_array, weights=weights_array))
    mad = float(np.average(np.abs(values_array - mean), weights=weights_array))
    variance = float(np.average(np.square(values_array - mean), weights=weights_array))
    return (1.0 - mad / max(abs(mean), EPS)) * 100.0, math.sqrt(max(variance, 0.0))


def summarize_dynamic_state(
    network: Network,
    params: HydraulicParams,
    volumes: np.ndarray,
    pressure_values: list[float],
    pressure_weights: list[float],
    executed_groups: int,
    feasible_groups: int,
    pressure_violation_groups: int,
    velocity_violation_groups: int,
    nonconverged_segments: int,
    total_segments: int,
    min_pressure_m: float,
    max_velocity_ms: float,
    end_time_h: float,
) -> dict[str, float]:
    target = params.target_per_emitter_m3
    deficit = np.maximum(target - volumes, 0.0)
    capped = np.minimum(volumes, target)
    uniformity, pressure_std = weighted_pressure_metrics(pressure_values, pressure_weights)
    return {
        "achievement_pct": float(capped.sum() / (target * len(network.emitters)) * 100.0),
        "unmet_emitter_ratio_pct": float(np.mean(deficit > 1.0e-9) * 100.0),
        "average_deficit_m3_per_emitter": float(np.mean(deficit)),
        "max_deficit_m3": float(np.max(deficit)),
        "pressure_uniformity_index_pct": uniformity,
        "pressure_std_m": pressure_std,
        "feasible_group_ratio_pct": feasible_groups / max(executed_groups, 1) * 100.0,
        "pressure_violation_group_ratio_pct": pressure_violation_groups / max(executed_groups, 1) * 100.0,
        "velocity_violation_group_ratio_pct": velocity_violation_groups / max(executed_groups, 1) * 100.0,
        "combined_violation_group_ratio_pct": (executed_groups - feasible_groups) / max(executed_groups, 1) * 100.0,
        "pda_nonconvergence_rate_pct": nonconverged_segments / max(total_segments, 1) * 100.0,
        "min_pressure_m": min_pressure_m if math.isfinite(min_pressure_m) else math.nan,
        "max_velocity_ms": max_velocity_ms,
        "executed_group_count": float(executed_groups),
        "end_time_h": end_time_h,
    }


def optimization_artifacts(
    result: OptimizationResult,
    experiment: str,
    scenario_id: int,
    stage: str,
    repeat: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fronts: list[dict[str, Any]] = []
    for population_index in result.nondominated_indices:
        objective = result.population_objectives[population_index]
        fronts.append(
            {
                "experiment": experiment,
                "scenario_id": scenario_id,
                "algorithm": result.algorithm,
                "stage": stage,
                "repeat": repeat,
                "population_index": int(population_index),
                "achievement_loss": float(objective[0]),
                "average_deficit_m3_per_emitter": float(objective[1]),
                "pressure_nonuniformity": float(objective[2]),
                "hydraulic_cv": float(result.population_violations[population_index]),
                "selected": int(population_index == int(np.argmin(np.linalg.norm(result.population_x - result.selected_x, axis=1)))),
                "function_evaluations": result.function_evaluations,
                "elapsed_s": result.elapsed_s,
            }
        )
    convergence: list[dict[str, Any]] = []
    for row in result.convergence:
        convergence.append(
            {
                **row,
                "experiment": experiment,
                "scenario_id": scenario_id,
                "algorithm": result.algorithm,
                "stage": stage,
                "repeat": repeat,
            }
        )
    return fronts, convergence


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
    return optimize_plan(
        network,
        params,
        candidates,
        forecast,
        unavailable,
        time_h,
        volumes,
        params.horizon_h,
        BEST_CONFIGS[algorithm],
        seed,
        evaluation_budget,
        cache_level=3,
    )


def simulate_with_events(
    network: Network,
    params: HydraulicParams,
    initial_plan: list[Group],
    realized_profile: Sequence[float],
    events: list[Event],
    dynamic: bool,
    algorithm: str,
    information_mode: str,
    optimizer_seed: int,
    reoptimization_budget: int,
    experiment: str,
    scenario_id: int,
    rolling_every_group: bool = False,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    events = sorted(events, key=lambda event: (event.time_h, 0 if event.kind == "fault" else 1))
    processed = np.zeros(len(events), dtype=bool)
    volumes = np.zeros(len(network.emitters), dtype=float)
    active_faults: set[str] = set()
    plan = list(initial_plan)
    plan_index = 0
    time_h = 0.0
    caches = EvaluationCaches(CacheConfig(level=2))
    pressure_values: list[float] = []
    pressure_weights: list[float] = []
    records: list[dict[str, Any]] = []
    fronts: list[dict[str, Any]] = []
    convergence: list[dict[str, Any]] = []
    executed_groups = 0
    feasible_groups = 0
    pressure_violation_groups = 0
    velocity_violation_groups = 0
    nonconverged_segments = 0
    total_segments = 0
    min_pressure = math.inf
    max_velocity = 0.0
    reoptimization_count = 0
    reoptimization_time = 0.0
    reoptimization_function_evaluations = 0
    affected_emitters: set[int] = set()
    for event in events:
        if event.kind == "fault":
            affected_emitters.update(isolated_emitters_after_fault(network, event.node))

    def apply_events_until(boundary_h: float) -> bool:
        changed = False
        for event_index, event in enumerate(events):
            if processed[event_index] or event.time_h > boundary_h + EPS:
                continue
            if event.kind == "fault":
                active_faults.add(event.node)
            else:
                active_faults.discard(event.node)
            processed[event_index] = True
            changed = True
        return changed

    while time_h < params.horizon_h - EPS:
        boundary_changed = apply_events_until(time_h)
        unavailable = unavailable_from_faults(network, active_faults)
        need_reoptimization = dynamic and (boundary_changed or plan_index >= len(plan))
        if rolling_every_group and dynamic and time_h > EPS:
            need_reoptimization = True
        if need_reoptimization:
            candidates = [
                index
                for index in range(len(network.emitters))
                if index not in unavailable and volumes[index] < params.target_per_emitter_m3 - 1.0e-9
            ]
            if not candidates:
                break
            result = optimize_current_state(
                network,
                params,
                algorithm,
                realized_profile,
                information_mode,
                candidates,
                unavailable,
                time_h,
                volumes,
                optimizer_seed + 100_003 * (reoptimization_count + 1),
                reoptimization_budget,
            )
            plan = result.plan
            plan_index = 0
            reoptimization_count += 1
            reoptimization_time += result.elapsed_s
            reoptimization_function_evaluations += result.function_evaluations
            new_fronts, new_convergence = optimization_artifacts(
                result, experiment, scenario_id, f"reopt_{reoptimization_count}", 1
            )
            fronts.extend(new_fronts)
            convergence.extend(new_convergence)
        if plan_index >= len(plan):
            break

        group = plan[plan_index]
        group_start = time_h
        group_end = min(group_start + group.duration_h, params.horizon_h)
        future_fault_times = [
            event.time_h
            for event_index, event in enumerate(events)
            if not processed[event_index]
            and event.kind == "fault"
            and time_h + EPS < event.time_h < group_end - EPS
        ]
        interrupt_time = min(future_fault_times) if future_fault_times else group_end
        group_executed = False
        group_feasible = True
        group_pressure_violation = False
        group_velocity_violation = False
        interrupted = interrupt_time < group_end - EPS
        while time_h < interrupt_time - EPS:
            next_hour = math.floor(time_h + EPS) + 1.0
            segment_end = min(interrupt_time, next_hour)
            duration_h = segment_end - time_h
            unavailable = unavailable_from_faults(network, active_faults)
            source_head = source_head_at(realized_profile, time_h)
            hydraulic = solve_group_pda(
                network,
                params,
                group.emitters,
                unavailable,
                source_head,
                caches=caches,
            )
            total_segments += 1
            nonconverged_segments += int(not hydraulic.converged)
            if hydraulic.active_emitters:
                group_executed = True
                group_feasible = group_feasible and hydraulic.feasible
                group_pressure_violation = group_pressure_violation or hydraulic.pressure_violation_count > 0
                group_velocity_violation = group_velocity_violation or hydraulic.velocity_violation_count > 0
                min_pressure = min(min_pressure, float(np.min(hydraulic.emitter_pressure_m)))
                active_velocity = np.abs(hydraulic.velocities_ms[np.abs(hydraulic.pipe_flows_m3s) > EPS])
                if len(active_velocity):
                    max_velocity = max(max_velocity, float(np.max(active_velocity)))
                for local_index, emitter_index in enumerate(hydraulic.active_emitters):
                    volumes[emitter_index] += hydraulic.delivered_flow_m3s[local_index] * duration_h * 3600.0
                    pressure_values.append(float(hydraulic.emitter_pressure_m[local_index]))
                    pressure_weights.append(duration_h)
            records.append(
                {
                    "experiment": experiment,
                    "scenario_id": scenario_id,
                    "algorithm": algorithm,
                    "policy": "dynamic_reoptimization" if dynamic else "static_no_reoptimization",
                    "segment_start_h": time_h,
                    "segment_end_h": segment_end,
                    "source_head_m": source_head,
                    "active_faults": ";".join(sorted(active_faults)),
                    "emitters": ";".join(network.emitters[e] for e in group.emitters),
                    "active_emitters": ";".join(network.emitters[e] for e in hydraulic.active_emitters),
                    "pda_converged": hydraulic.converged,
                    "pda_iterations": hydraulic.iterations,
                    "min_pressure_m": float(np.min(hydraulic.emitter_pressure_m)) if len(hydraulic.emitter_pressure_m) else math.nan,
                    "max_velocity_ms": float(np.max(active_velocity)) if len(active_velocity) else 0.0,
                    "feasible": hydraulic.feasible,
                }
            )
            time_h = segment_end
        if group_executed:
            executed_groups += 1
            feasible_groups += int(group_feasible)
            pressure_violation_groups += int(group_pressure_violation)
            velocity_violation_groups += int(group_velocity_violation)

        if interrupted:
            changed = apply_events_until(time_h)
            if dynamic and changed:
                plan_index = len(plan)
            else:
                plan_index += 1
            continue

        time_h = group_end
        plan_index += 1
        recovery_or_boundary_event = apply_events_until(time_h)
        if dynamic and recovery_or_boundary_event:
            plan_index = len(plan)

    metrics = summarize_dynamic_state(
        network,
        params,
        volumes,
        pressure_values,
        pressure_weights,
        executed_groups,
        feasible_groups,
        pressure_violation_groups,
        velocity_violation_groups,
        nonconverged_segments,
        total_segments,
        min_pressure,
        max_velocity,
        time_h,
    )
    target = params.target_per_emitter_m3
    if affected_emitters:
        affected = np.asarray(sorted(affected_emitters), dtype=int)
        affected_deficit = np.maximum(target - volumes[affected], 0.0)
        metrics.update(
            {
                "affected_emitter_count": float(len(affected)),
                "affected_target_met_pct": float(np.mean(affected_deficit <= 1.0e-9) * 100.0),
                "affected_mean_completion_pct": float(np.mean(np.minimum(volumes[affected], target) / target) * 100.0),
                "affected_average_deficit_m3": float(np.mean(affected_deficit)),
                "affected_max_deficit_m3": float(np.max(affected_deficit)),
            }
        )
    else:
        metrics.update(
            {
                "affected_emitter_count": 0.0,
                "affected_target_met_pct": math.nan,
                "affected_mean_completion_pct": math.nan,
                "affected_average_deficit_m3": math.nan,
                "affected_max_deficit_m3": math.nan,
            }
        )
    metrics.update(
        {
            "reoptimization_count": float(reoptimization_count),
            "total_reoptimization_time_s": reoptimization_time,
            "mean_reoptimization_time_s": reoptimization_time / max(reoptimization_count, 1),
            "total_reoptimization_function_evaluations": float(reoptimization_function_evaluations),
        }
    )
    return metrics, records, fronts, convergence


def run_normal_task(task: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if NETWORK is None or PARAMS is None:
        raise RuntimeError("worker not initialized")
    network, params = NETWORK, PARAMS
    scenario_id = int(task["scenario_id"])
    seed = int(task["seed"])
    realized_profile = source_profile(seed, params)
    rows: list[dict[str, Any]] = []
    fronts: list[dict[str, Any]] = []
    convergence: list[dict[str, Any]] = []

    original = original_plan(network, params)
    metrics, _, _ = evaluate_plan(network, params, original, realized_profile)
    rows.append(
        {
            **metrics,
            "experiment": "normal",
            "scenario_id": scenario_id,
            "policy": "original_near_far",
            "algorithm": "none",
            "repeat": 0,
            "seed": seed,
            "information_mode": "causal_execution_only",
        }
    )
    for repeat in range(1, int(task["random_repeats"]) + 1):
        plan = random_plan(network, params, random.Random(seed + repeat * 101))
        metrics, _, _ = evaluate_plan(network, params, plan, realized_profile)
        rows.append(
            {
                **metrics,
                "experiment": "normal",
                "scenario_id": scenario_id,
                "policy": "random_rotation",
                "algorithm": "none",
                "repeat": repeat,
                "seed": seed + repeat * 101,
                "information_mode": "causal_execution_only",
            }
        )
    for repeat in range(1, int(task["nsga_repeats"]) + 1):
        result = optimize_current_state(
            network,
            params,
            "nsga2",
            realized_profile,
            "causal_mean",
            list(range(len(network.emitters))),
            set(),
            0.0,
            np.zeros(len(network.emitters)),
            seed + repeat * 1009,
            int(task["initial_budget"]),
        )
        metrics, _, _ = evaluate_plan(network, params, result.plan, realized_profile)
        rows.append(
            {
                **metrics,
                "experiment": "normal",
                "scenario_id": scenario_id,
                "policy": "tuned_nsga2_causal",
                "algorithm": "nsga2",
                "repeat": repeat,
                "seed": seed + repeat * 1009,
                "information_mode": "causal_mean",
                "optimization_time_s": result.elapsed_s,
                "function_evaluations": result.function_evaluations,
                "selected_hydraulic_cv": result.selected_violation,
            }
        )
        new_fronts, new_convergence = optimization_artifacts(
            result, "normal", scenario_id, "initial", repeat
        )
        fronts.extend(new_fronts)
        convergence.extend(new_convergence)
    return rows, fronts, convergence


def run_fault_task(
    task: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if NETWORK is None or PARAMS is None:
        raise RuntimeError("worker not initialized")
    network, params = NETWORK, PARAMS
    experiment = str(task["experiment"])
    scenario_id = int(task["scenario_id"])
    seed = int(task["seed"])
    realized_profile = source_profile(seed, params)
    events = (
        single_fault_scenario(seed + 333, network, params)
        if experiment == "single_fault"
        else multi_fault_scenario(seed + 777, network, params)
    )
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    fronts: list[dict[str, Any]] = []
    convergence: list[dict[str, Any]] = []
    event_text = ";".join(f"{event.kind}:{event.node}@{event.time_h:.6f}" for event in events)
    for algorithm_index, algorithm in enumerate(sorted(BEST_CONFIGS)):
        initial = optimize_current_state(
            network,
            params,
            algorithm,
            realized_profile,
            "causal_mean",
            list(range(len(network.emitters))),
            set(),
            0.0,
            np.zeros(len(network.emitters)),
            seed + 10_009 + algorithm_index * 1_000_003,
            int(task["initial_budget"]),
        )
        initial_fronts, initial_convergence = optimization_artifacts(
            initial, experiment, scenario_id, "initial", 1
        )
        fronts.extend(initial_fronts)
        convergence.extend(initial_convergence)
        for dynamic, suffix in ((False, "static_no_reoptimization"), (True, "dynamic_reoptimization")):
            metrics, execution, new_fronts, new_convergence = simulate_with_events(
                network,
                params,
                initial.plan,
                realized_profile,
                events,
                dynamic,
                algorithm,
                "causal_mean",
                seed + (50_000 if dynamic else 60_000) + algorithm_index * 1_000_003,
                int(task["reoptimization_budget"]),
                experiment,
                scenario_id,
            )
            rows.append(
                {
                    **metrics,
                    "experiment": experiment,
                    "scenario_id": scenario_id,
                    "policy": f"{algorithm}_{suffix}",
                    "algorithm": algorithm,
                    "repeat": 1,
                    "seed": seed,
                    "information_mode": "causal_mean",
                    "initial_optimization_time_s": initial.elapsed_s,
                    "initial_function_evaluations": initial.function_evaluations,
                    "event_count": len(events),
                    "fault_count": sum(event.kind == "fault" for event in events),
                    "recovery_count": sum(event.kind == "recovery" for event in events),
                    "events": event_text,
                }
            )
            records.extend(execution)
            fronts.extend(new_fronts)
            convergence.extend(new_convergence)
    return rows, records, fronts, convergence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Revised closed-PDA irrigation experiments")
    repository_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--nodes", type=Path, default=repository_root / "data" / "network" / "Nodes.xlsx")
    parser.add_argument("--pipes", type=Path, default=repository_root / "data" / "network" / "Pipes.xlsx")
    parser.add_argument("--best-config", type=Path, default=repository_root / "data" / "algorithm_tuning" / "best_algorithm_parameters.json")
    parser.add_argument("--output-dir", type=Path, default=repository_root / "outputs" / "main_experiments")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--normal-scenarios", type=int, default=100)
    parser.add_argument("--single-fault-scenarios", type=int, default=100)
    parser.add_argument("--multi-fault-scenarios", type=int, default=500)
    parser.add_argument("--random-repeats", type=int, default=50)
    parser.add_argument("--nsga-repeats", type=int, default=5)
    parser.add_argument("--initial-budget", type=int, default=2400)
    parser.add_argument("--reoptimization-budget", type=int, default=1200)
    parser.add_argument("--base-seed", type=int, default=2026071600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers != DEFAULT_WORKERS:
        raise ValueError(f"all formal parallel calculations require exactly {DEFAULT_WORKERS} workers")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    normal_tasks = [
        {
            "scenario_id": scenario_id,
            "seed": args.base_seed + scenario_id * 100,
            "random_repeats": args.random_repeats,
            "nsga_repeats": args.nsga_repeats,
            "initial_budget": args.initial_budget,
        }
        for scenario_id in range(1, args.normal_scenarios + 1)
    ]
    fault_tasks = [
        {
            "experiment": "single_fault",
            "scenario_id": scenario_id,
            "seed": args.base_seed + 1_000_000 + scenario_id * 100,
            "initial_budget": args.initial_budget,
            "reoptimization_budget": args.reoptimization_budget,
        }
        for scenario_id in range(1, args.single_fault_scenarios + 1)
    ] + [
        {
            "experiment": "multi_fault",
            "scenario_id": scenario_id,
            "seed": args.base_seed + 2_000_000 + scenario_id * 100,
            "initial_budget": args.initial_budget,
            "reoptimization_budget": args.reoptimization_budget,
        }
        for scenario_id in range(1, args.multi_fault_scenarios + 1)
    ]
    started = time.perf_counter()
    policy_rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []
    front_rows: list[dict[str, Any]] = []
    convergence_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=DEFAULT_WORKERS,
        initializer=init_worker,
        initargs=(str(args.nodes), str(args.pipes), str(args.best_config)),
    ) as executor:
        futures = [executor.submit(run_normal_task, task) for task in normal_tasks]
        futures.extend(executor.submit(run_fault_task, task) for task in fault_tasks)
        total = len(futures)
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if len(result) == 3:
                policies, fronts, convergence = result
                records = []
            else:
                policies, records, fronts, convergence = result
            policy_rows.extend(policies)
            record_rows.extend(records)
            front_rows.extend(fronts)
            convergence_rows.extend(convergence)
            if completed == 1 or completed % 10 == 0 or completed == total:
                print(
                    f"Main experiments {completed}/{total}; elapsed={(time.perf_counter() - started) / 60:.1f} min",
                    flush=True,
                )

    policies = pd.DataFrame(policy_rows)
    records = pd.DataFrame(record_rows)
    fronts = pd.DataFrame(front_rows)
    convergence = pd.DataFrame(convergence_rows)
    policies.to_csv(args.output_dir / "revised_policy_metrics.csv", index=False, encoding="utf-8-sig")
    records.to_parquet(args.output_dir / "revised_execution_records.parquet", index=False)
    fronts.to_parquet(args.output_dir / "revised_pareto_fronts.parquet", index=False)
    convergence.to_parquet(args.output_dir / "revised_convergence_traces.parquet", index=False)
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.replace("\n", " "),
        "workers": DEFAULT_WORKERS,
        "logical_cpu_count": os.cpu_count(),
        "normal_scenarios": args.normal_scenarios,
        "single_fault_scenarios": args.single_fault_scenarios,
        "multi_fault_scenarios": args.multi_fault_scenarios,
        "random_repeats": args.random_repeats,
        "nsga2_repeats": args.nsga_repeats,
        "initial_evaluation_budget": args.initial_budget,
        "reoptimization_evaluation_budget": args.reoptimization_budget,
        "information_set": "causal: observed current/past source head, current faults, cumulative volumes; no future head or event times",
        "future_head_assumption": "25 m expectation beyond decision time",
        "best_config_path": str(args.best_config),
        "wall_time_s": time.perf_counter() - started,
        "policy_row_count": len(policies),
        "execution_record_count": len(records),
        "pareto_point_count": len(fronts),
    }
    (args.output_dir / "revised_main_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
