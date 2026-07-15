from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from revision_core import (
    DEFAULT_WORKERS,
    CacheConfig,
    EvaluationCaches,
    HydraulicParams,
    Network,
    evaluate_plan,
    isolated_emitters_after_fault,
    keys_to_plan,
    load_network,
    original_plan,
    source_profile,
)
from revised_optimization import AlgorithmConfig, forecast_for_information_mode, optimize_plan


NETWORK: Network | None = None
PARAMS: HydraulicParams | None = None
NSGA2_CONFIG: AlgorithmConfig | None = None


def init_worker(nodes_path: str, pipes_path: str, best_config_path: str) -> None:
    global NETWORK, PARAMS, NSGA2_CONFIG
    NETWORK = load_network(Path(nodes_path), Path(pipes_path))
    PARAMS = HydraulicParams()
    data = json.loads(Path(best_config_path).read_text(encoding="utf-8"))
    NSGA2_CONFIG = AlgorithmConfig(**data["nsga2"]["config"])


class PeakMemorySampler:
    def __init__(self, interval_s: float = 0.01) -> None:
        self.interval_s = interval_s
        self.process = psutil.Process()
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval_s):
            try:
                self.peak = max(self.peak, self.process.memory_info().rss)
            except psutil.Error:
                return

    def __enter__(self) -> "PeakMemorySampler":
        self.thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        try:
            self.peak = max(self.peak, self.process.memory_info().rss)
        except psutil.Error:
            pass

    @property
    def peak_delta_bytes(self) -> int:
        return max(self.peak - self.baseline, 0)


def benchmark_context(task_class: str, scenario_id: int, base_seed: int) -> dict[str, Any]:
    if NETWORK is None or PARAMS is None:
        raise RuntimeError("worker not initialized")
    network, params = NETWORK, PARAMS
    seed = base_seed + scenario_id * 1009 + {"normal": 0, "single_fault": 1_000_000, "multi_fault": 2_000_000}[task_class]
    profile = source_profile(seed, params)
    start_time = 0.0
    start_volumes = np.zeros(len(network.emitters), dtype=float)
    unavailable: set[int] = set()
    if task_class != "normal":
        start_time = float(24 + (scenario_id * 7) % 56)
        prefix_count = int(math.ceil(start_time / params.target_duration_h))
        _, start_volumes, _ = evaluate_plan(
            network,
            params,
            original_plan(network, params)[:prefix_count],
            profile,
            horizon_h=start_time,
        )
        node_count = 1 if task_class == "single_fault" else 1 + scenario_id % 3
        nodes = [network.irrigation_nodes[(scenario_id * 11 + offset * 17) % len(network.irrigation_nodes)] for offset in range(node_count)]
        for node in nodes:
            unavailable.update(isolated_emitters_after_fault(network, node))
    candidates = [
        index
        for index in range(len(network.emitters))
        if index not in unavailable and start_volumes[index] < params.target_per_emitter_m3 - 1.0e-9
    ]
    return {
        "seed": seed,
        "profile": forecast_for_information_mode(profile, start_time, "causal_mean"),
        "start_time_h": start_time,
        "start_volumes": start_volumes,
        "unavailable": unavailable,
        "candidates": candidates,
    }


def run_cache_task(task: dict[str, Any]) -> dict[str, Any]:
    if NETWORK is None or PARAMS is None or NSGA2_CONFIG is None:
        raise RuntimeError("worker not initialized")
    context = benchmark_context(task["task_class"], int(task["scenario_id"]), int(task["base_seed"]))
    level = int(task["cache_level"])
    cache_state = str(task["cache_state"])
    caches = EvaluationCaches(CacheConfig(level=level))
    if cache_state in {"warm_cross_seed", "hot_exact_replay"}:
        warm_seed = (
            int(task["optimizer_seed"]) + 1
            if cache_state == "warm_cross_seed"
            else int(task["optimizer_seed"])
        )
        optimize_plan(
            NETWORK,
            PARAMS,
            context["candidates"],
            context["profile"],
            context["unavailable"],
            context["start_time_h"],
            context["start_volumes"],
            PARAMS.horizon_h,
            NSGA2_CONFIG,
            warm_seed,
            int(task["evaluation_budget"]),
            cache_level=level,
            external_caches=caches,
        )
        caches.stats.clear()
    with PeakMemorySampler() as memory:
        result = optimize_plan(
            NETWORK,
            PARAMS,
            context["candidates"],
            context["profile"],
            context["unavailable"],
            context["start_time_h"],
            context["start_volumes"],
            PARAMS.horizon_h,
            NSGA2_CONFIG,
            int(task["optimizer_seed"]),
            int(task["evaluation_budget"]),
            cache_level=level,
            external_caches=caches,
        )
    row = {
        **task,
        "actual_function_evaluations": result.function_evaluations,
        "elapsed_s": result.elapsed_s,
        "seconds_per_1000_evaluations": result.elapsed_s / result.function_evaluations * 1000.0,
        "peak_rss_delta_bytes": memory.peak_delta_bytes,
        "achievement_pct": result.selected_metrics["achievement_pct"],
        "average_deficit_m3_per_emitter": result.selected_metrics["average_deficit_m3_per_emitter"],
        "pressure_uniformity_index_pct": result.selected_metrics["pressure_uniformity_index_pct"],
        "feasible_group_ratio_pct": result.selected_metrics["feasible_group_ratio_pct"],
        "selected_hydraulic_cv": result.selected_violation,
    }
    row.update(result.cache_metrics)
    return row


def synthetic_network(branches: int) -> Network:
    nodes_rows: list[dict[str, Any]] = [{"node_id": "J0", "X": 0.0, "Y": 0.0, "Z": 0.0}]
    pipes_rows: list[dict[str, Any]] = []
    parent_pipe: dict[str, int] = {}
    previous_main = "J0"
    for branch in range(1, branches + 1):
        main = f"M{branch}"
        nodes_rows.append({"node_id": main, "X": branch * 50.0, "Y": 0.0, "Z": -0.05 * branch})
        pipes_rows.append(
            {"Pipe ID": f"G{len(pipes_rows)}", "FromNode": previous_main, "ToNode": main, "Length_m": 50.0, "Diameter_m": 0.25, "Material": "UPVC"}
        )
        parent_pipe[main] = len(pipes_rows) - 1
        previous_lateral = main
        for position in range(1, 7):
            node = f"B{branch}_{position}"
            nodes_rows.append(
                {"node_id": node, "X": branch * 50.0, "Y": position * 100.0, "Z": -0.05 * branch - 0.08 * position}
            )
            pipes_rows.append(
                {"Pipe ID": f"G{len(pipes_rows)}", "FromNode": previous_lateral, "ToNode": node, "Length_m": 100.0, "Diameter_m": 0.16, "Material": "PE"}
            )
            parent_pipe[node] = len(pipes_rows) - 1
            previous_lateral = node
    nodes = pd.DataFrame(nodes_rows)
    pipes = pd.DataFrame(pipes_rows)
    node_ids = tuple(nodes["node_id"])
    node_index = {node: index for index, node in enumerate(node_ids)}
    irrigation_nodes = tuple(node for node in node_ids if node.startswith("B"))
    emitters = tuple(f"{node}{side}" for node in irrigation_nodes for side in ("L", "R"))
    emitter_to_node = tuple(name[:-1] for name in emitters)
    from_nodes = tuple(pipes["FromNode"])
    to_nodes = tuple(pipes["ToNode"])
    node_to_path: dict[str, tuple[int, ...]] = {}
    for node in irrigation_nodes:
        current = node
        path: list[int] = []
        while current != "J0":
            pipe_index = parent_pipe[current]
            path.append(pipe_index)
            current = from_nodes[pipe_index]
        node_to_path[node] = tuple(reversed(path))
    path_matrix = np.zeros((len(emitters), len(pipes)), dtype=float)
    for emitter_index, node in enumerate(emitter_to_node):
        path_matrix[emitter_index, list(node_to_path[node])] = 1.0
    diameters = pipes["Diameter_m"].to_numpy(dtype=float)
    roughness = np.where(pipes["Material"].eq("UPVC"), 0.018, 0.020).astype(float)
    fingerprint = hashlib.blake2b(
        f"synthetic-{branches}-v1".encode("ascii"), digest_size=16
    ).hexdigest()
    return Network(
        nodes=nodes,
        pipes=pipes,
        node_ids=node_ids,
        node_index=node_index,
        node_z=nodes["Z"].to_numpy(dtype=float),
        pipe_ids=tuple(pipes["Pipe ID"]),
        from_nodes=from_nodes,
        to_nodes=to_nodes,
        lengths_m=pipes["Length_m"].to_numpy(dtype=float),
        diameters_m=diameters,
        areas_m2=math.pi * diameters**2 / 4.0,
        roughness_mm=roughness,
        parent_pipe=parent_pipe,
        irrigation_nodes=irrigation_nodes,
        emitters=emitters,
        emitter_to_node=emitter_to_node,
        emitter_node_indices=np.array([node_index[node] for node in emitter_to_node], dtype=int),
        emitter_index={name: index for index, name in enumerate(emitters)},
        node_to_path=node_to_path,
        emitter_path_matrix=path_matrix,
        fingerprint=fingerprint,
    )


def run_scalability_task(task: dict[str, Any]) -> dict[str, Any]:
    params = HydraulicParams(horizon_h=float(task["emitter_count"]))
    network = synthetic_network(int(task["branches"]))
    rng = np.random.default_rng(int(task["seed"]))
    keys = rng.random(len(network.emitters))
    plan = keys_to_plan(keys, list(range(len(network.emitters))), params, stage="scalability")
    profile = [25.0] * (int(params.horizon_h) + 1)
    caches = EvaluationCaches(CacheConfig(level=int(task["cache_level"])))
    with PeakMemorySampler() as memory:
        started = time.perf_counter()
        metrics, _, _ = evaluate_plan(network, params, plan, profile, caches=caches)
        elapsed = time.perf_counter() - started
    return {
        **task,
        "node_count": len(network.node_ids),
        "pipe_count": len(network.pipe_ids),
        "actual_emitter_count": len(network.emitters),
        "elapsed_s": elapsed,
        "peak_rss_delta_bytes": memory.peak_delta_bytes,
        "achievement_pct": metrics["achievement_pct"],
        "pda_nonconvergence_rate_pct": metrics["pda_nonconvergence_rate_pct"],
        **{key: value for key, value in __import__("revision_core").cache_statistics(caches).items()},
    }


def bootstrap_mean_ci(values: np.ndarray, seed: int, n_boot: int = 2000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_cache(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in results.groupby(["cache_level", "cache_state", "task_class"]):
        for metric in ("elapsed_s", "seconds_per_1000_evaluations", "peak_rss_delta_bytes"):
            values = group[metric].to_numpy(dtype=float)
            low, high = bootstrap_mean_ci(values, 20260715 + int(keys[0]) * 100)
            rows.append(
                {
                    "cache_level": keys[0],
                    "cache_state": keys[1],
                    "task_class": keys[2],
                    "metric": metric,
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "median": float(np.median(values)),
                    "p95": float(np.quantile(values, 0.95)),
                    "worst": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fair cache ablation and network scalability experiments")
    repository_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--nodes", type=Path, default=repository_root / "data" / "network" / "Nodes.xlsx")
    parser.add_argument("--pipes", type=Path, default=repository_root / "data" / "network" / "Pipes.xlsx")
    parser.add_argument("--best-config", type=Path, default=repository_root / "data" / "algorithm_tuning" / "best_algorithm_parameters.json")
    parser.add_argument("--output-dir", type=Path, default=repository_root / "outputs" / "cache_scalability")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--evaluation-budget", type=int, default=480)
    parser.add_argument("--base-seed", type=int, default=2026071700)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers != DEFAULT_WORKERS:
        raise ValueError(f"all formal parallel calculations require exactly {DEFAULT_WORKERS} workers")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_tasks: list[dict[str, Any]] = []
    composition = {"normal": 20, "single_fault": 20, "multi_fault": 50}
    for task_class, count in composition.items():
        for scenario_id in range(1, count + 1):
            for cache_level in range(4):
                for cache_state in ("cold", "warm_cross_seed", "hot_exact_replay"):
                    for repeat in range(1, args.benchmark_repeats + 1):
                        cache_tasks.append(
                            {
                                "task_class": task_class,
                                "scenario_id": scenario_id,
                                "cache_level": cache_level,
                                "cache_state": cache_state,
                                "repeat": repeat,
                                "base_seed": args.base_seed,
                                "optimizer_seed": args.base_seed + scenario_id * 1000 + repeat,
                                "evaluation_budget": args.evaluation_budget,
                            }
                        )
    scalability_tasks = [
        {
            "branches": branches,
            "emitter_count": branches * 12,
            "cache_level": cache_level,
            "repeat": repeat,
            "seed": args.base_seed + branches * 100 + cache_level * 10 + repeat,
        }
        for branches in (5, 10, 20, 40)
        for cache_level in range(4)
        for repeat in range(1, 11)
    ]
    started = time.perf_counter()
    cache_rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=DEFAULT_WORKERS,
        initializer=init_worker,
        initargs=(str(args.nodes), str(args.pipes), str(args.best_config)),
    ) as executor:
        futures = {executor.submit(run_cache_task, task): "cache" for task in cache_tasks}
        futures.update({executor.submit(run_scalability_task, task): "scale" for task in scalability_tasks})
        total = len(futures)
        for completed, future in enumerate(as_completed(futures), start=1):
            if futures[future] == "cache":
                cache_rows.append(future.result())
            else:
                scale_rows.append(future.result())
            if completed == 1 or completed % 50 == 0 or completed == total:
                print(f"Cache/scalability {completed}/{total}; elapsed={(time.perf_counter() - started) / 60:.1f} min", flush=True)
    cache_results = pd.DataFrame(cache_rows)
    scale_results = pd.DataFrame(scale_rows)
    cache_summary = summarize_cache(cache_results)
    cache_results.to_csv(args.output_dir / "cache_ablation_runs.csv", index=False, encoding="utf-8-sig")
    cache_summary.to_csv(args.output_dir / "cache_ablation_summary.csv", index=False, encoding="utf-8-sig")
    scale_results.to_csv(args.output_dir / "network_scalability_runs.csv", index=False, encoding="utf-8-sig")
    complexity = """# Complexity analysis\n\n- One closed-PDA group evaluation is O(I_pda * (A*D + E_a*I_cw)), where A is active emitter count, D is mean path depth, E_a is active pipes, I_pda is pressure-flow iterations, and I_cw is Colebrook iterations.\n- A complete plan with K groups is O(K * I_pda * (A*D + E_a*I_cw)).\n- NSGA-II non-dominated sorting is O(M*N^2) per generation for M objectives and population N; the hydraulic evaluator is usually the dominant measured cost.\n- L1 pipe caching removes repeated Colebrook work, L2 group caching removes repeated converged PDA solves for identical group/head states, and L3 plan caching removes repeated complete-plan evaluations in the same immutable context.\n- Caches reduce expected constants and repeated work; they do not improve worst-case asymptotic complexity.\n"""
    (args.output_dir / "complexity_analysis.md").write_text(complexity, encoding="utf-8")
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.replace("\n", " "),
        "workers": DEFAULT_WORKERS,
        "logical_cpu_count": os.cpu_count(),
        "cache_task_composition": composition,
        "cache_levels": {"0": "none", "1": "pipe", "2": "pipe+group", "3": "pipe+group+plan"},
        "cache_states": {
            "cold": "empty cache",
            "warm_cross_seed": "same scenario and budget pre-run with a different optimizer seed",
            "hot_exact_replay": "identical scenario, budget, task, and optimizer seed pre-run",
        },
        "benchmark_repeats": args.benchmark_repeats,
        "evaluation_budget": args.evaluation_budget,
        "scalability_emitter_counts": [60, 120, 240, 480],
        "wall_time_s": time.perf_counter() - started,
        "cache_run_count": len(cache_results),
        "scalability_run_count": len(scale_results),
    }
    (args.output_dir / "cache_scalability_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
