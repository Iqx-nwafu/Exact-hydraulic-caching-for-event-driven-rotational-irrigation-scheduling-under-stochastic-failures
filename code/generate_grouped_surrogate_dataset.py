from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from revision_core import (
    DEFAULT_WORKERS,
    CacheConfig,
    EvaluationCaches,
    Group,
    HydraulicParams,
    Network,
    evaluate_plan,
    isolated_emitters_after_fault,
    load_network,
    objectives_from_metrics,
    source_profile,
)
from revised_optimization import forecast_for_information_mode


TOPOLOGIES: dict[str, Network] = {}
PARAMS: HydraulicParams | None = None


def synthetic_tree_network(branches: int, nodes_per_branch: int, variant_seed: int) -> Network:
    if branches * nodes_per_branch != 60:
        raise ValueError("all surrogate topologies must contain 60 irrigation nodes and 120 emitters")
    rng = np.random.default_rng(variant_seed)
    nodes_rows: list[dict[str, Any]] = [{"node_id": "J0", "X": 0.0, "Y": 0.0, "Z": 0.0}]
    pipes_rows: list[dict[str, Any]] = []
    parent_pipe: dict[str, int] = {}
    previous_main = "J0"
    for branch in range(1, branches + 1):
        main = f"M{branch}"
        main_length = float(rng.uniform(42.0, 58.0))
        nodes_rows.append(
            {
                "node_id": main,
                "X": branch * main_length,
                "Y": 0.0,
                "Z": float(-0.04 * branch + rng.normal(0.0, 0.15)),
            }
        )
        pipes_rows.append(
            {
                "Pipe ID": f"G{len(pipes_rows)}",
                "FromNode": previous_main,
                "ToNode": main,
                "Length_m": main_length,
                "Diameter_m": float(rng.choice([0.225, 0.25, 0.28])),
                "Material": "UPVC",
            }
        )
        parent_pipe[main] = len(pipes_rows) - 1
        previous_main = main
        previous_lateral = main
        for position in range(1, nodes_per_branch + 1):
            node = f"B{branch}_{position}"
            lateral_length = float(rng.uniform(85.0, 115.0))
            nodes_rows.append(
                {
                    "node_id": node,
                    "X": branch * main_length,
                    "Y": position * lateral_length,
                    "Z": float(-0.04 * branch - 0.07 * position + rng.normal(0.0, 0.25)),
                }
            )
            pipes_rows.append(
                {
                    "Pipe ID": f"G{len(pipes_rows)}",
                    "FromNode": previous_lateral,
                    "ToNode": node,
                    "Length_m": lateral_length,
                    "Diameter_m": float(rng.choice([0.14, 0.16, 0.18])),
                    "Material": "PE",
                }
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
        path: list[int] = []
        current = node
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
        f"synthetic-{branches}x{nodes_per_branch}-seed{variant_seed}".encode("ascii"), digest_size=16
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


def init_worker(nodes_path: str, pipes_path: str) -> None:
    global TOPOLOGIES, PARAMS
    actual = load_network(Path(nodes_path), Path(pipes_path))
    TOPOLOGIES = {
        "actual_10x6": actual,
        "development_10x6": synthetic_tree_network(10, 6, 101),
        "development_12x5": synthetic_tree_network(12, 5, 102),
        "development_15x4": synthetic_tree_network(15, 4, 103),
        "external_6x10": synthetic_tree_network(6, 10, 201),
        "external_20x3": synthetic_tree_network(20, 3, 202),
        "external_5x12": synthetic_tree_network(5, 12, 203),
    }
    PARAMS = HydraulicParams()


def emitter_branch(network: Network, emitter_index: int) -> str:
    node = network.emitter_to_node[emitter_index]
    if node.startswith("B"):
        return node.split("_")[0]
    number = int(node[1:])
    return f"B{10 if number >= 100 else number // 10}"


def context_for_scenario(network: Network, params: HydraulicParams, scenario_seed: int, context_type: str) -> dict[str, Any]:
    rng = random.Random(scenario_seed)
    profile = source_profile(scenario_seed + 17, params)
    start_time = 0.0
    volumes = np.zeros(len(network.emitters), dtype=float)
    unavailable: set[int] = set()
    if context_type in {"fault", "recovery"}:
        fault_count = rng.randint(1, 3)
        fault_nodes = rng.sample(list(network.irrigation_nodes), k=fault_count)
        affected: set[int] = set()
        for node in fault_nodes:
            if node.startswith("J"):
                affected.update(isolated_emitters_after_fault(network, node))
            else:
                branch = node.split("_")[0]
                position = int(node.split("_")[1])
                for emitter_index, emitter_node in enumerate(network.emitter_to_node):
                    if emitter_node.startswith(branch + "_") and int(emitter_node.split("_")[1]) >= position:
                        affected.add(emitter_index)
        start_time = rng.uniform(12.0, 96.0)
        progress = rng.betavariate(1.5, 2.5)
        volumes = params.target_per_emitter_m3 * np.minimum(
            np.asarray([rng.random() for _ in network.emitters]) * progress * 1.5,
            1.0,
        )
        if context_type == "fault":
            unavailable = affected
        else:
            volumes[list(affected)] *= 0.25
    candidates = [
        index
        for index in range(len(network.emitters))
        if index not in unavailable and volumes[index] < params.target_per_emitter_m3 - 1.0e-9
    ]
    if not candidates:
        candidates = [index for index in range(len(network.emitters)) if index not in unavailable]
    return {
        "profile": forecast_for_information_mode(profile, start_time, "causal_mean"),
        "start_time_h": start_time,
        "volumes": volumes,
        "unavailable": unavailable,
        "candidates": candidates,
    }


def random_plan_and_assignment(
    rng: random.Random,
    candidates: Sequence[int],
    params: HydraulicParams,
    emitter_count: int,
) -> tuple[list[Group], np.ndarray]:
    order = list(candidates)
    rng.shuffle(order)
    plan = [
        Group(tuple(order[index : index + params.group_size]), params.target_duration_h, "surrogate")
        for index in range(0, len(order), params.group_size)
    ]
    assignment = np.full(emitter_count, -1, dtype=np.int16)
    for group_index, group in enumerate(plan):
        assignment[list(group.emitters)] = group_index
    return plan, assignment


def group_features(
    network: Network,
    params: HydraulicParams,
    group: Group | None,
    group_index: int,
    source_profile_values: Sequence[float],
    start_time_h: float,
) -> list[float]:
    if group is None or not group.emitters:
        return [-1.0] * 14
    emitters = np.asarray(group.emitters, dtype=int)
    paths = network.emitter_path_matrix[emitters]
    path_lengths = paths @ network.lengths_m
    path_depths = paths.sum(axis=1)
    elevations = network.node_z[network.emitter_node_indices[emitters]]
    mean_diameters = np.array(
        [float(np.mean(network.diameters_m[np.flatnonzero(path)])) for path in paths]
    )
    shared_values: list[float] = []
    for left in range(len(paths)):
        for right in range(left + 1, len(paths)):
            union = np.count_nonzero((paths[left] + paths[right]) > 0)
            shared = np.count_nonzero((paths[left] * paths[right]) > 0)
            shared_values.append(shared / max(union, 1))
    branches = {emitter_branch(network, int(emitter)) for emitter in emitters}
    scheduled_time = start_time_h + group_index * params.target_duration_h
    profile_index = min(int(math.floor(scheduled_time)), len(source_profile_values) - 1)
    return [
        len(emitters) / params.group_size,
        len(set(network.emitter_to_node[int(emitter)] for emitter in emitters)) / len(emitters),
        len(branches) / len(emitters),
        float(np.mean(path_lengths) / 2000.0),
        float(np.max(path_lengths) / 2000.0),
        float(np.std(path_lengths) / 1000.0),
        float(np.mean(path_depths) / 30.0),
        float(np.max(path_depths) / 50.0),
        float(np.mean(elevations) / 20.0),
        float(np.std(elevations) / 10.0),
        float(np.mean(mean_diameters) / 0.3),
        float(np.mean(shared_values)) if shared_values else 1.0,
        float(np.max(shared_values)) if shared_values else 1.0,
        float(source_profile_values[profile_index] / 30.0),
    ]


def plan_features(
    network: Network,
    params: HydraulicParams,
    plan: Sequence[Group],
    context: dict[str, Any],
) -> np.ndarray:
    features: list[float] = []
    maximum_groups = len(network.emitters) // params.group_size
    for group_index in range(maximum_groups):
        group = plan[group_index] if group_index < len(plan) else None
        features.extend(
            group_features(
                network,
                params,
                group,
                group_index,
                context["profile"],
                context["start_time_h"],
            )
        )
    paths = network.emitter_path_matrix
    path_lengths = paths @ network.lengths_m
    path_depths = paths.sum(axis=1)
    profile = np.asarray(context["profile"], dtype=float)
    volumes = np.asarray(context["volumes"], dtype=float) / params.target_per_emitter_m3
    features.extend(
        [
            context["start_time_h"] / params.horizon_h,
            (params.horizon_h - context["start_time_h"]) / params.horizon_h,
            len(context["candidates"]) / len(network.emitters),
            len(context["unavailable"]) / len(network.emitters),
            len(plan) / maximum_groups,
            float(np.mean(profile) / 30.0),
            float(np.min(profile) / 30.0),
            float(np.max(profile) / 30.0),
            float(np.std(profile) / 10.0),
            float(np.mean(volumes)),
            float(np.std(volumes)),
            float(np.mean(volumes >= 1.0 - 1.0e-9)),
            len(network.node_ids) / 100.0,
            len(network.pipe_ids) / 100.0,
            float(np.mean(path_lengths) / 2000.0),
            float(np.max(path_lengths) / 3000.0),
            float(np.mean(path_depths) / 30.0),
            float(np.max(path_depths) / 50.0),
            float(np.mean(network.diameters_m) / 0.3),
            float(np.std(network.diameters_m) / 0.1),
            float(np.mean(network.node_z) / 20.0),
            float(np.std(network.node_z) / 10.0),
        ]
    )
    return np.asarray(features, dtype=np.float32)


def feature_names(emitter_count: int, group_size: int) -> list[str]:
    group_names = [
        "active_ratio",
        "unique_node_ratio",
        "branch_diversity",
        "path_length_mean",
        "path_length_max",
        "path_length_std",
        "path_depth_mean",
        "path_depth_max",
        "elevation_mean",
        "elevation_std",
        "path_diameter_mean",
        "shared_path_mean",
        "shared_path_max",
        "scheduled_source_head",
    ]
    names = [
        f"group_{group_index:02d}_{name}"
        for group_index in range(emitter_count // group_size)
        for name in group_names
    ]
    names.extend(
        [
            "start_time_norm",
            "remaining_time_norm",
            "candidate_ratio",
            "unavailable_ratio",
            "group_count_norm",
            "source_mean_norm",
            "source_min_norm",
            "source_max_norm",
            "source_std_norm",
            "volume_mean_fraction",
            "volume_std_fraction",
            "volume_met_ratio",
            "node_count_norm",
            "pipe_count_norm",
            "topology_path_length_mean",
            "topology_path_length_max",
            "topology_path_depth_mean",
            "topology_path_depth_max",
            "topology_diameter_mean",
            "topology_diameter_std",
            "topology_elevation_mean",
            "topology_elevation_std",
        ]
    )
    return names


def plan_hash(topology_id: str, assignment: np.ndarray) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(topology_id.encode("utf-8"))
    digest.update(np.asarray(assignment, dtype=np.int16).tobytes())
    return digest.hexdigest()


def run_scenario(task: dict[str, Any]) -> tuple[list[np.ndarray], list[dict[str, Any]], list[dict[str, Any]], list[np.ndarray]]:
    if PARAMS is None:
        raise RuntimeError("worker not initialized")
    topology_id = str(task["topology_id"])
    network = TOPOLOGIES[topology_id]
    params = PARAMS
    scenario_id = int(task["scenario_id"])
    scenario_seed = int(task["scenario_seed"])
    context_type = str(task["context_type"])
    context = context_for_scenario(network, params, scenario_seed, context_type)
    rng = random.Random(scenario_seed + 100_000_003)
    caches = EvaluationCaches(CacheConfig(level=2))
    features: list[np.ndarray] = []
    labels: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    assignments: list[np.ndarray] = []
    for plan_repeat in range(1, int(task["plans_per_scenario"]) + 1):
        plan, assignment = random_plan_and_assignment(
            rng,
            context["candidates"],
            params,
            len(network.emitters),
        )
        metrics, _, _ = evaluate_plan(
            network,
            params,
            plan,
            context["profile"],
            unavailable=context["unavailable"],
            start_time_h=context["start_time_h"],
            start_volumes=context["volumes"],
            horizon_h=params.horizon_h,
            caches=caches,
        )
        objectives, violation = objectives_from_metrics(metrics)
        features.append(plan_features(network, params, plan, context))
        labels.append(
            {
                "obj_achievement_loss": float(objectives[0]),
                "obj_average_deficit": float(objectives[1]),
                "obj_pressure_nonuniformity": float(objectives[2]),
                "constraint_violation": violation,
                "achievement_pct": metrics["achievement_pct"],
                "unmet_emitter_ratio_pct": metrics["unmet_emitter_ratio_pct"],
                "average_deficit_m3_per_emitter": metrics["average_deficit_m3_per_emitter"],
                "max_deficit_m3": metrics["max_deficit_m3"],
                "pressure_uniformity_index_pct": metrics["pressure_uniformity_index_pct"],
                "feasible_group_ratio_pct": metrics["feasible_group_ratio_pct"],
            }
        )
        metadata.append(
            {
                "sample_id": f"{topology_id}_s{scenario_id:05d}_p{plan_repeat:02d}",
                "scenario_id": f"{topology_id}_s{scenario_id:05d}",
                "topology_id": topology_id,
                "context_type": context_type,
                "plan_repeat": plan_repeat,
                "split": task["split"],
                "is_external_topology": bool(task["is_external_topology"]),
                "plan_hash": plan_hash(topology_id, assignment),
                "planned_emitters": len(context["candidates"]),
                "unavailable_emitters": len(context["unavailable"]),
                "start_time_h": context["start_time_h"],
            }
        )
        assignments.append(assignment)
    return features, labels, metadata, assignments


def internal_split(index: int) -> str:
    remainder = index % 20
    if remainder < 14:
        return "train"
    if remainder < 17:
        return "validation"
    return "test"


def build_tasks(
    development_scenarios: int,
    external_scenarios_per_topology: int,
    plans_per_scenario: int,
    base_seed: int,
) -> list[dict[str, Any]]:
    development_topologies = ["actual_10x6", "development_10x6", "development_12x5", "development_15x4"]
    external_topologies = ["external_6x10", "external_20x3", "external_5x12"]
    contexts = ["normal", "fault", "recovery"]
    tasks: list[dict[str, Any]] = []
    for scenario_id in range(1, development_scenarios + 1):
        scenario_index = scenario_id - 1
        split_block = scenario_index // 20
        tasks.append(
            {
                "scenario_id": scenario_id,
                "topology_id": development_topologies[(scenario_index + split_block) % len(development_topologies)],
                "context_type": contexts[scenario_index % len(contexts)],
                "scenario_seed": base_seed + scenario_id * 1009,
                "plans_per_scenario": plans_per_scenario,
                "split": internal_split(scenario_id - 1),
                "is_external_topology": False,
            }
        )
    external_id = development_scenarios
    for topology_index, topology_id in enumerate(external_topologies):
        for local_id in range(1, external_scenarios_per_topology + 1):
            external_id += 1
            tasks.append(
                {
                    "scenario_id": external_id,
                    "topology_id": topology_id,
                    "context_type": contexts[(local_id - 1) % len(contexts)],
                    "scenario_seed": base_seed + 10_000_000 + topology_index * 1_000_000 + local_id * 1009,
                    "plans_per_scenario": plans_per_scenario,
                    "split": "external_test",
                    "is_external_topology": True,
                }
            )
    return tasks


def leakage_audit(metadata: pd.DataFrame, assignments: np.ndarray) -> dict[str, Any]:
    scenario_split_counts = metadata.groupby("scenario_id")["split"].nunique()
    exact_hash_split_counts = metadata.groupby("plan_hash")["split"].nunique()
    development = metadata[~metadata["is_external_topology"]].copy()
    train_indices = development.index[development["split"].eq("train")].to_numpy()
    audit_indices = development.index[development["split"].isin(["validation", "test"])].to_numpy()
    nearest = NearestNeighbors(n_neighbors=1, metric="hamming", algorithm="brute", n_jobs=DEFAULT_WORKERS)
    nearest.fit(assignments[train_indices])
    distances, _ = nearest.kneighbors(assignments[audit_indices])
    distances = distances[:, 0]
    return {
        "scenario_ids_present_in_multiple_splits": int(np.sum(scenario_split_counts > 1)),
        "exact_plan_hashes_present_in_multiple_splits": int(np.sum(exact_hash_split_counts > 1)),
        "nearest_train_plan_hamming_distance_for_validation_test": {
            "minimum": float(np.min(distances)),
            "p01": float(np.quantile(distances, 0.01)),
            "p05": float(np.quantile(distances, 0.05)),
            "median": float(np.median(distances)),
            "near_duplicate_count_below_0p05": int(np.sum(distances < 0.05)),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate grouped, topology-aware surrogate data with leakage audit")
    repository_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--nodes", type=Path, default=repository_root / "data" / "network" / "Nodes.xlsx")
    parser.add_argument("--pipes", type=Path, default=repository_root / "data" / "network" / "Pipes.xlsx")
    parser.add_argument("--output-dir", type=Path, default=repository_root / "outputs" / "surrogate_grouped")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--development-scenarios", type=int, default=1500)
    parser.add_argument("--external-scenarios-per-topology", type=int, default=200)
    parser.add_argument("--plans-per-scenario", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=2026072000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers != DEFAULT_WORKERS:
        raise ValueError(f"all formal parallel calculations require exactly {DEFAULT_WORKERS} workers")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(
        args.development_scenarios,
        args.external_scenarios_per_topology,
        args.plans_per_scenario,
        args.base_seed,
    )
    started = time.perf_counter()
    feature_rows: list[np.ndarray] = []
    label_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    assignment_rows: list[np.ndarray] = []
    with ProcessPoolExecutor(
        max_workers=DEFAULT_WORKERS,
        initializer=init_worker,
        initargs=(str(args.nodes), str(args.pipes)),
    ) as executor:
        futures = [executor.submit(run_scenario, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            features, labels, metadata, assignments = future.result()
            feature_rows.extend(features)
            label_rows.extend(labels)
            metadata_rows.extend(metadata)
            assignment_rows.extend(assignments)
            if completed == 1 or completed % 100 == 0 or completed == len(futures):
                print(f"Surrogate scenarios {completed}/{len(futures)}; elapsed={(time.perf_counter() - started) / 60:.1f} min", flush=True)
    names = feature_names(120, 4)
    features = pd.DataFrame(np.vstack(feature_rows), columns=names)
    labels = pd.DataFrame(label_rows)
    metadata = pd.DataFrame(metadata_rows)
    assignments = np.vstack(assignment_rows).astype(np.int16)
    audit = leakage_audit(metadata, assignments)
    features.to_parquet(args.output_dir / "surrogate_features.parquet", index=False)
    labels.to_parquet(args.output_dir / "surrogate_labels.parquet", index=False)
    metadata.to_csv(args.output_dir / "surrogate_sample_metadata.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(args.output_dir / "surrogate_plan_assignments.npz", assignments=assignments)
    (args.output_dir / "surrogate_leakage_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    split_counts = metadata.groupby(["split", "topology_id", "context_type"]).size().reset_index(name="samples")
    split_counts.to_csv(args.output_dir / "surrogate_split_counts.csv", index=False, encoding="utf-8-sig")
    metadata_output = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.replace("\n", " "),
        "workers": DEFAULT_WORKERS,
        "logical_cpu_count": os.cpu_count(),
        "development_scenarios": args.development_scenarios,
        "development_samples": args.development_scenarios * args.plans_per_scenario,
        "external_scenarios": args.external_scenarios_per_topology * 3,
        "external_samples": args.external_scenarios_per_topology * 3 * args.plans_per_scenario,
        "plans_per_scenario": args.plans_per_scenario,
        "group_split": "scenario_id; 70% train, 15% validation, 15% internal test",
        "external_topologies": ["6 branches x 10 nodes", "20 branches x 3 nodes", "5 branches x 12 nodes"],
        "feature_count": len(features.columns),
        "wall_time_s": time.perf_counter() - started,
        "leakage_audit": audit,
    }
    (args.output_dir / "surrogate_dataset_metadata.json").write_text(
        json.dumps(metadata_output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata_output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
