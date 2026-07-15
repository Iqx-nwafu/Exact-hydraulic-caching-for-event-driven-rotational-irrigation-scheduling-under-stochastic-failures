from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_revised_main_experiments as main_exp
from revision_core import DEFAULT_WORKERS, multi_fault_scenario, single_fault_scenario, source_profile


def init_worker(nodes_path: str, pipes_path: str, best_config_path: str) -> None:
    main_exp.init_worker(nodes_path, pipes_path, best_config_path)


def run_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    network = main_exp.NETWORK
    params = main_exp.PARAMS
    if network is None or params is None:
        raise RuntimeError("worker not initialized")
    experiment = str(task["experiment"])
    scenario_id = int(task["scenario_id"])
    seed = int(task["seed"])
    profile = source_profile(seed, params)
    events = (
        single_fault_scenario(seed + 333, network, params)
        if experiment == "single_fault"
        else multi_fault_scenario(seed + 777, network, params)
    )
    modes = [
        ("oracle_event_driven", "oracle", False),
        ("causal_mean_event_driven", "causal_mean", False),
        ("causal_persistence_event_driven", "causal_persistence", False),
        ("causal_mean_rolling_group", "causal_mean", True),
    ]
    rows: list[dict[str, Any]] = []
    for mode_index, (policy, information_mode, rolling) in enumerate(modes):
        initial = main_exp.optimize_current_state(
            network,
            params,
            "nsga2",
            profile,
            information_mode,
            list(range(len(network.emitters))),
            set(),
            0.0,
            np.zeros(len(network.emitters)),
            seed + 10_000_003,
            int(task["initial_budget"]),
        )
        metrics, _, _, _ = main_exp.simulate_with_events(
            network,
            params,
            initial.plan,
            profile,
            events,
            True,
            "nsga2",
            information_mode,
            seed + 20_000_003,
            int(task["reoptimization_budget"]),
            experiment,
            scenario_id,
            rolling_every_group=rolling,
        )
        rows.append(
            {
                **metrics,
                "experiment": experiment,
                "scenario_id": scenario_id,
                "policy": policy,
                "information_mode": information_mode,
                "rolling_every_group": rolling,
                "initial_optimization_time_s": initial.elapsed_s,
                "initial_function_evaluations": initial.function_evaluations,
                "total_optimization_time_s": initial.elapsed_s + metrics["total_reoptimization_time_s"],
                "total_function_evaluations": initial.function_evaluations
                + metrics["total_reoptimization_function_evaluations"],
                "event_count": len(events),
                "fault_count": sum(event.kind == "fault" for event in events),
                "recovery_count": sum(event.kind == "recovery" for event in events),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle versus causal and rolling information-set comparison")
    repository_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--nodes", type=Path, default=repository_root / "data" / "network" / "Nodes.xlsx")
    parser.add_argument("--pipes", type=Path, default=repository_root / "data" / "network" / "Pipes.xlsx")
    parser.add_argument("--best-config", type=Path, default=repository_root / "data" / "algorithm_tuning" / "best_algorithm_parameters.json")
    parser.add_argument("--output-dir", type=Path, default=repository_root / "outputs" / "information_set")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--single-scenarios", type=int, default=30)
    parser.add_argument("--multi-scenarios", type=int, default=30)
    parser.add_argument("--initial-budget", type=int, default=960)
    parser.add_argument("--reoptimization-budget", type=int, default=480)
    parser.add_argument("--base-seed", type=int, default=2026071800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers != DEFAULT_WORKERS:
        raise ValueError(f"all formal parallel calculations require exactly {DEFAULT_WORKERS} workers")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        {
            "experiment": experiment,
            "scenario_id": scenario_id,
            "seed": args.base_seed + offset + scenario_id * 100,
            "initial_budget": args.initial_budget,
            "reoptimization_budget": args.reoptimization_budget,
        }
        for experiment, count, offset in (
            ("single_fault", args.single_scenarios, 1_000_000),
            ("multi_fault", args.multi_scenarios, 2_000_000),
        )
        for scenario_id in range(1, count + 1)
    ]
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=DEFAULT_WORKERS,
        initializer=init_worker,
        initargs=(str(args.nodes), str(args.pipes), str(args.best_config)),
    ) as executor:
        futures = [executor.submit(run_task, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            if completed == 1 or completed % 5 == 0 or completed == len(futures):
                print(f"Information-set {completed}/{len(futures)}; elapsed={(time.perf_counter() - started) / 60:.1f} min", flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "information_set_policy_metrics.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.replace("\n", " "),
        "workers": DEFAULT_WORKERS,
        "logical_cpu_count": os.cpu_count(),
        "single_fault_scenarios": args.single_scenarios,
        "multi_fault_scenarios": args.multi_scenarios,
        "paired_modes": [
            "oracle full future source-head trajectory; no future events",
            "causal current/past observations with 25 m future expectation",
            "causal current/past observations with persistence future expectation",
            "causal 25 m expectation reoptimized at every group boundary",
        ],
        "initial_budget": args.initial_budget,
        "reoptimization_budget": args.reoptimization_budget,
        "wall_time_s": time.perf_counter() - started,
        "row_count": len(results),
    }
    (args.output_dir / "information_set_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
