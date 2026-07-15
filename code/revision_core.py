from __future__ import annotations

import hashlib
import math
import pickle
import random
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


G = 9.80665
EPS = 1.0e-12
MODEL_VERSION = "closed-pda-v1"
DEFAULT_WORKERS = 20


@dataclass(frozen=True)
class HydraulicParams:
    design_head_m: float = 11.4
    design_flow_m3s: float = 0.013
    target_duration_h: float = 4.0
    horizon_h: float = 120.0
    group_size: int = 4
    min_velocity_ms: float = 0.3
    max_velocity_ms: float = 2.5
    kinematic_viscosity_m2s: float = 1.0e-6
    pda_flow_tolerance_m3s: float = 1.0e-10
    pda_head_tolerance_m: float = 1.0e-8
    pda_max_iterations: int = 200
    pda_relaxation: float = 0.45

    @property
    def target_per_emitter_m3(self) -> float:
        return self.design_flow_m3s * self.target_duration_h * 3600.0


@dataclass(frozen=True)
class Network:
    nodes: pd.DataFrame
    pipes: pd.DataFrame
    node_ids: tuple[str, ...]
    node_index: dict[str, int]
    node_z: np.ndarray
    pipe_ids: tuple[str, ...]
    from_nodes: tuple[str, ...]
    to_nodes: tuple[str, ...]
    lengths_m: np.ndarray
    diameters_m: np.ndarray
    areas_m2: np.ndarray
    roughness_mm: np.ndarray
    parent_pipe: dict[str, int]
    irrigation_nodes: tuple[str, ...]
    emitters: tuple[str, ...]
    emitter_to_node: tuple[str, ...]
    emitter_node_indices: np.ndarray
    emitter_index: dict[str, int]
    node_to_path: dict[str, tuple[int, ...]]
    emitter_path_matrix: np.ndarray
    fingerprint: str


@dataclass(frozen=True)
class Group:
    emitters: tuple[int, ...]
    duration_h: float = 4.0
    stage: str = "scheduled"


@dataclass(frozen=True)
class Event:
    time_h: float
    kind: str
    node: str


@dataclass(frozen=True)
class PDAResult:
    active_emitters: tuple[int, ...]
    node_pressure: dict[str, float]
    emitter_pressure_m: np.ndarray
    delivered_flow_m3s: np.ndarray
    pipe_flows_m3s: np.ndarray
    velocities_ms: np.ndarray
    friction_factors: np.ndarray
    friction_loss_m: np.ndarray
    converged: bool
    iterations: int
    flow_residual_m3s: float
    head_residual_m: float
    pressure_violation_count: int
    pressure_violation_m: float
    velocity_violation_count: int
    velocity_violation_ms: float
    feasible: bool


@dataclass(frozen=True)
class CacheConfig:
    level: int = 3
    pipe_max_items: int = 100_000
    group_max_items: int = 100_000
    plan_max_items: int = 30_000

    def __post_init__(self) -> None:
        if self.level not in (0, 1, 2, 3):
            raise ValueError("cache level must be 0, 1, 2, or 3")


@dataclass
class EvaluationCaches:
    config: CacheConfig
    pipe: OrderedDict[tuple[Any, ...], tuple[float, float]] = field(default_factory=OrderedDict)
    group: OrderedDict[tuple[Any, ...], PDAResult] = field(default_factory=OrderedDict)
    plan: OrderedDict[tuple[Any, ...], tuple[dict[str, float], np.ndarray]] = field(default_factory=OrderedDict)
    stats: Counter[str] = field(default_factory=Counter)

    def clear(self) -> None:
        self.pipe.clear()
        self.group.clear()
        self.plan.clear()
        self.stats.clear()

    def estimated_bytes(self) -> int:
        return len(pickle.dumps((self.pipe, self.group, self.plan), protocol=pickle.HIGHEST_PROTOCOL))


def _put_lru(cache: OrderedDict, key: Any, value: Any, max_items: int) -> None:
    if max_items <= 0:
        return
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_items:
        cache.popitem(last=False)


def _node_key(node_id: str) -> tuple[int, str]:
    if node_id.startswith("J") and node_id[1:].isdigit():
        return int(node_id[1:]), node_id
    return 10**9, node_id


def branch_position(node_id: str) -> tuple[int, int]:
    number = int(node_id[1:])
    if number >= 100:
        return 10, number - 100
    return number // 10, number % 10


def _network_fingerprint(nodes: pd.DataFrame, pipes: pd.DataFrame, roughness_mm: np.ndarray) -> str:
    payload = (
        nodes[["node_id", "X", "Y", "Z"]].to_csv(index=False)
        + pipes[["Pipe ID", "FromNode", "ToNode", "Length_m", "Diameter_m", "Material"]].to_csv(index=False)
        + np.asarray(roughness_mm, dtype=np.float64).tobytes().hex()
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def load_network(
    nodes_path: Path,
    pipes_path: Path,
    roughness_mm_by_material: dict[str, float] | None = None,
) -> Network:
    roughness_mm_by_material = roughness_mm_by_material or {"UPVC": 0.018, "PE": 0.020}
    nodes = pd.read_excel(nodes_path)
    pipes = pd.read_excel(pipes_path)
    nodes.columns = [str(c).strip() for c in nodes.columns]
    pipes.columns = [str(c).strip() for c in pipes.columns]
    nodes = nodes.rename(columns={nodes.columns[0]: "node_id"})
    nodes["node_id"] = nodes["node_id"].astype(str).str.strip()
    for column in ("Pipe ID", "FromNode", "ToNode", "Material"):
        pipes[column] = pipes[column].astype(str).str.strip()
    pipes["Material"] = pipes["Material"].str.upper()

    required_nodes = {"node_id", "X", "Y", "Z"}
    required_pipes = {"Pipe ID", "FromNode", "ToNode", "Length_m", "Diameter_m", "Material"}
    if not required_nodes.issubset(nodes.columns):
        raise ValueError(f"Nodes.xlsx missing columns: {sorted(required_nodes - set(nodes.columns))}")
    if not required_pipes.issubset(pipes.columns):
        raise ValueError(f"Pipes.xlsx missing columns: {sorted(required_pipes - set(pipes.columns))}")

    node_ids = tuple(nodes["node_id"].tolist())
    node_index = {node: i for i, node in enumerate(node_ids)}
    missing_nodes = (set(pipes["FromNode"]) | set(pipes["ToNode"])) - set(node_ids)
    if missing_nodes:
        raise ValueError(f"pipes reference missing nodes: {sorted(missing_nodes)}")
    missing_materials = set(pipes["Material"]) - set(roughness_mm_by_material)
    if missing_materials:
        raise ValueError(f"roughness missing for materials: {sorted(missing_materials)}")

    from_nodes = tuple(pipes["FromNode"].tolist())
    to_nodes = tuple(pipes["ToNode"].tolist())
    parent_pipe: dict[str, int] = {}
    for pipe_idx, node in enumerate(to_nodes):
        if node in parent_pipe:
            raise ValueError(f"tree topology required; node {node} has multiple parents")
        parent_pipe[node] = pipe_idx

    main_nodes = {f"J{i}" for i in range(11)}
    irrigation_nodes = tuple(node for node in sorted(node_ids, key=_node_key) if node not in main_nodes)
    emitters = tuple(f"{node}{side}" for node in irrigation_nodes for side in ("L", "R"))
    emitter_to_node = tuple(name[:-1] for name in emitters)
    emitter_index = {name: i for i, name in enumerate(emitters)}
    node_to_path: dict[str, tuple[int, ...]] = {}
    for node in irrigation_nodes:
        current = node
        path: list[int] = []
        seen: set[str] = set()
        while current != "J0":
            if current in seen:
                raise ValueError(f"cycle detected while tracing {node}")
            seen.add(current)
            if current not in parent_pipe:
                raise ValueError(f"node {current} has no upstream pipe")
            pipe_idx = parent_pipe[current]
            path.append(pipe_idx)
            current = from_nodes[pipe_idx]
        node_to_path[node] = tuple(reversed(path))

    n_emitters = len(emitters)
    n_pipes = len(pipes)
    path_matrix = np.zeros((n_emitters, n_pipes), dtype=np.float64)
    for emitter_idx, node in enumerate(emitter_to_node):
        path_matrix[emitter_idx, list(node_to_path[node])] = 1.0

    diameters = pipes["Diameter_m"].astype(float).to_numpy()
    roughness = np.array([roughness_mm_by_material[m] for m in pipes["Material"]], dtype=float)
    return Network(
        nodes=nodes,
        pipes=pipes,
        node_ids=node_ids,
        node_index=node_index,
        node_z=nodes["Z"].astype(float).to_numpy(),
        pipe_ids=tuple(pipes["Pipe ID"].tolist()),
        from_nodes=from_nodes,
        to_nodes=to_nodes,
        lengths_m=pipes["Length_m"].astype(float).to_numpy(),
        diameters_m=diameters,
        areas_m2=math.pi * np.square(diameters) / 4.0,
        roughness_mm=roughness,
        parent_pipe=parent_pipe,
        irrigation_nodes=irrigation_nodes,
        emitters=emitters,
        emitter_to_node=emitter_to_node,
        emitter_node_indices=np.array([node_index[n] for n in emitter_to_node], dtype=int),
        emitter_index=emitter_index,
        node_to_path=node_to_path,
        emitter_path_matrix=path_matrix,
        fingerprint=_network_fingerprint(nodes, pipes, roughness),
    )


def colebrook_white_friction_factor(reynolds: float, roughness_m: float, diameter_m: float) -> float:
    if reynolds <= EPS:
        return 0.0
    if reynolds < 2000.0:
        return 64.0 / reynolds

    def turbulent(re_value: float) -> float:
        relative_roughness = max(roughness_m / max(diameter_m, EPS), 0.0)
        f = 1.0 / max(
            (-1.8 * math.log10((relative_roughness / 3.7) ** 1.11 + 6.9 / max(re_value, EPS))) ** 2,
            EPS,
        )
        for _ in range(50):
            inverse_sqrt = -2.0 * math.log10(
                relative_roughness / 3.7 + 2.51 / (re_value * math.sqrt(f))
            )
            updated = 1.0 / max(inverse_sqrt * inverse_sqrt, EPS)
            if abs(updated - f) <= 1.0e-12:
                return float(updated)
            f = updated
        return float(f)

    if reynolds >= 4000.0:
        return turbulent(reynolds)
    fraction = (reynolds - 2000.0) / 2000.0
    smooth = fraction * fraction * (3.0 - 2.0 * fraction)
    return float((1.0 - smooth) * (64.0 / reynolds) + smooth * turbulent(4000.0))


def source_head_at(source_profile: Sequence[float], time_h: float, default: float = 25.0) -> float:
    if not source_profile:
        return default
    index = int(min(max(math.floor(time_h + 1.0e-12), 0), len(source_profile) - 1))
    return float(source_profile[index])


def _pipe_hydraulics(
    network: Network,
    params: HydraulicParams,
    pipe_flows: np.ndarray,
    caches: EvaluationCaches | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    velocities = np.divide(pipe_flows, network.areas_m2, out=np.zeros_like(pipe_flows), where=network.areas_m2 > EPS)
    active = np.abs(pipe_flows) > EPS
    reynolds = np.zeros_like(pipe_flows)
    reynolds[active] = (
        np.abs(velocities[active]) * network.diameters_m[active] / params.kinematic_viscosity_m2s
    )
    factors = np.zeros_like(pipe_flows)
    losses = np.zeros_like(pipe_flows)
    roughness_m = network.roughness_mm / 1000.0
    for pipe_idx in np.flatnonzero(active):
        flow_hex = np.float64(pipe_flows[pipe_idx]).tobytes()
        key = (MODEL_VERSION, network.fingerprint, int(pipe_idx), flow_hex)
        cached = caches.pipe.get(key) if caches is not None and caches.config.level >= 1 else None
        if cached is not None:
            caches.stats["pipe_hits"] += 1
            caches.pipe.move_to_end(key)
            factors[pipe_idx], losses[pipe_idx] = cached
            continue
        if caches is not None and caches.config.level >= 1:
            caches.stats["pipe_misses"] += 1
        factor = colebrook_white_friction_factor(
            float(reynolds[pipe_idx]), float(roughness_m[pipe_idx]), float(network.diameters_m[pipe_idx])
        )
        loss = (
            factor
            * network.lengths_m[pipe_idx]
            / network.diameters_m[pipe_idx]
            * velocities[pipe_idx] ** 2
            / (2.0 * G)
        )
        factors[pipe_idx] = factor
        losses[pipe_idx] = loss
        if caches is not None and caches.config.level >= 1:
            _put_lru(caches.pipe, key, (float(factor), float(loss)), caches.config.pipe_max_items)
    return velocities, reynolds, factors, losses


def solve_group_pda(
    network: Network,
    params: HydraulicParams,
    emitters: Sequence[int],
    unavailable: set[int],
    source_head_m: float,
    caches: EvaluationCaches | None = None,
) -> PDAResult:
    active_emitters = tuple(sorted(int(e) for e in emitters if int(e) not in unavailable))
    group_key = (
        MODEL_VERSION,
        network.fingerprint,
        active_emitters,
        np.float64(source_head_m).tobytes(),
        params.design_head_m,
        params.design_flow_m3s,
        params.pda_flow_tolerance_m3s,
        params.pda_head_tolerance_m,
        params.pda_relaxation,
    )
    cached = caches.group.get(group_key) if caches is not None and caches.config.level >= 2 else None
    if cached is not None:
        caches.stats["group_hits"] += 1
        caches.group.move_to_end(group_key)
        return cached
    if caches is not None and caches.config.level >= 2:
        caches.stats["group_misses"] += 1

    n_pipes = len(network.pipe_ids)
    if not active_emitters:
        empty = np.zeros(0, dtype=float)
        result = PDAResult(
            active_emitters=(),
            node_pressure={},
            emitter_pressure_m=empty,
            delivered_flow_m3s=empty,
            pipe_flows_m3s=np.zeros(n_pipes),
            velocities_ms=np.zeros(n_pipes),
            friction_factors=np.zeros(n_pipes),
            friction_loss_m=np.zeros(n_pipes),
            converged=True,
            iterations=0,
            flow_residual_m3s=0.0,
            head_residual_m=0.0,
            pressure_violation_count=0,
            pressure_violation_m=0.0,
            velocity_violation_count=0,
            velocity_violation_ms=0.0,
            feasible=False,
        )
        if caches is not None and caches.config.level >= 2:
            _put_lru(caches.group, group_key, result, caches.config.group_max_items)
        return result

    active_array = np.asarray(active_emitters, dtype=int)
    path_matrix = network.emitter_path_matrix[active_array]
    emitter_z = network.node_z[network.emitter_node_indices[active_array]]
    required_flow = np.full(len(active_emitters), params.design_flow_m3s, dtype=float)
    flow = required_flow.copy()
    previous_head = np.full(len(active_emitters), np.nan)
    converged = False
    flow_residual = math.inf
    head_residual = math.inf
    velocities = np.zeros(n_pipes)
    factors = np.zeros(n_pipes)
    losses = np.zeros(n_pipes)
    pipe_flows = np.zeros(n_pipes)
    pressure = np.full(len(active_emitters), source_head_m) - emitter_z

    for iteration in range(1, params.pda_max_iterations + 1):
        pipe_flows = path_matrix.T @ flow
        velocities, _, factors, losses = _pipe_hydraulics(network, params, pipe_flows, caches)
        pressure = source_head_m - emitter_z - path_matrix @ losses
        target_flow = required_flow * np.clip(pressure / params.design_head_m, 0.0, 1.0)
        updated_flow = (1.0 - params.pda_relaxation) * flow + params.pda_relaxation * target_flow
        flow_residual = float(np.max(np.abs(updated_flow - flow)))
        if np.all(np.isfinite(previous_head)):
            head_residual = float(np.max(np.abs(pressure - previous_head)))
        previous_head = pressure.copy()
        flow = updated_flow
        if flow_residual <= params.pda_flow_tolerance_m3s and head_residual <= params.pda_head_tolerance_m:
            converged = True
            break

    pipe_flows = path_matrix.T @ flow
    velocities, _, factors, losses = _pipe_hydraulics(network, params, pipe_flows, caches)
    pressure = source_head_m - emitter_z - path_matrix @ losses
    fixed_point_flow = required_flow * np.clip(pressure / params.design_head_m, 0.0, 1.0)
    flow_residual = float(np.max(np.abs(fixed_point_flow - flow)))
    flow = fixed_point_flow
    pipe_flows = path_matrix.T @ flow
    velocities, _, factors, losses = _pipe_hydraulics(network, params, pipe_flows, caches)
    pressure = source_head_m - emitter_z - path_matrix @ losses
    final_target_flow = required_flow * np.clip(pressure / params.design_head_m, 0.0, 1.0)
    flow_residual = float(np.max(np.abs(final_target_flow - flow)))
    if flow_residual <= params.pda_flow_tolerance_m3s and head_residual <= params.pda_head_tolerance_m:
        converged = True
    else:
        converged = False

    pressure_deficit = np.maximum(params.design_head_m - pressure, 0.0)
    pressure_violation_count = int(np.count_nonzero(pressure_deficit > 1.0e-9))
    active_pipe = np.abs(pipe_flows) > EPS
    active_velocities = np.abs(velocities[active_pipe])
    low_velocity = np.maximum(params.min_velocity_ms - active_velocities, 0.0)
    high_velocity = np.maximum(active_velocities - params.max_velocity_ms, 0.0)
    velocity_violation_count = int(np.count_nonzero((low_velocity + high_velocity) > 1.0e-9))
    node_pressure: dict[str, float] = {}
    for local_idx, emitter_idx in enumerate(active_emitters):
        node = network.emitter_to_node[emitter_idx]
        node_pressure[node] = float(pressure[local_idx])

    result = PDAResult(
        active_emitters=active_emitters,
        node_pressure=node_pressure,
        emitter_pressure_m=pressure.astype(float),
        delivered_flow_m3s=flow.astype(float),
        pipe_flows_m3s=pipe_flows.astype(float),
        velocities_ms=velocities.astype(float),
        friction_factors=factors.astype(float),
        friction_loss_m=losses.astype(float),
        converged=converged,
        iterations=iteration,
        flow_residual_m3s=flow_residual,
        head_residual_m=head_residual,
        pressure_violation_count=pressure_violation_count,
        pressure_violation_m=float(pressure_deficit.sum()),
        velocity_violation_count=velocity_violation_count,
        velocity_violation_ms=float((low_velocity + high_velocity).sum()),
        feasible=bool(converged and pressure_violation_count == 0 and velocity_violation_count == 0),
    )
    if caches is not None and caches.config.level >= 2:
        _put_lru(caches.group, group_key, result, caches.config.group_max_items)
    return result


def isolated_emitters_after_fault(network: Network, fault_node: str) -> set[int]:
    branch, position = branch_position(fault_node)
    nodes = [f"J{branch}{p}" if branch < 10 else f"J10{p}" for p in range(position, 7)]
    isolated: set[int] = set()
    for node in nodes:
        for side in ("L", "R"):
            emitter = f"{node}{side}"
            if emitter in network.emitter_index:
                isolated.add(network.emitter_index[emitter])
    return isolated


def unavailable_from_faults(network: Network, active_faults: Iterable[str]) -> set[int]:
    unavailable: set[int] = set()
    for node in active_faults:
        unavailable.update(isolated_emitters_after_fault(network, node))
    return unavailable


def original_plan(network: Network, params: HydraulicParams) -> list[Group]:
    order = list(range(len(network.emitters)))
    return [
        Group(tuple(order[i : i + params.group_size]), params.target_duration_h, "original")
        for i in range(0, len(order), params.group_size)
    ]


def random_plan(network: Network, params: HydraulicParams, rng: random.Random) -> list[Group]:
    order = list(range(len(network.emitters)))
    rng.shuffle(order)
    return [
        Group(tuple(order[i : i + params.group_size]), params.target_duration_h, "random")
        for i in range(0, len(order), params.group_size)
    ]


def keys_to_plan(
    keys: np.ndarray,
    candidates: Sequence[int],
    params: HydraulicParams,
    stage: str = "optimized",
) -> list[Group]:
    if len(keys) != len(candidates):
        raise ValueError("random-key vector and candidate list have different lengths")
    order = [int(candidates[i]) for i in np.argsort(np.asarray(keys), kind="mergesort")]
    return [
        Group(tuple(order[i : i + params.group_size]), params.target_duration_h, stage)
        for i in range(0, len(order), params.group_size)
    ]


def _hash_context(
    source_profile: Sequence[float],
    unavailable: set[int],
    start_time_h: float,
    start_volumes: np.ndarray,
    horizon_h: float,
) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.asarray(source_profile, dtype=np.float64).tobytes())
    digest.update(np.asarray(sorted(unavailable), dtype=np.int32).tobytes())
    digest.update(np.float64(start_time_h).tobytes())
    digest.update(np.asarray(start_volumes, dtype=np.float64).tobytes())
    digest.update(np.float64(horizon_h).tobytes())
    return digest.hexdigest()


def _next_hour_boundary(time_h: float) -> float:
    return math.floor(time_h + 1.0e-12) + 1.0


def _weighted_pressure_uniformity(values: list[float], weights: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    value_array = np.asarray(values, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    mean = float(np.average(value_array, weights=weight_array))
    mad = float(np.average(np.abs(value_array - mean), weights=weight_array))
    uniformity = (1.0 - mad / max(abs(mean), EPS)) * 100.0
    variance = float(np.average(np.square(value_array - mean), weights=weight_array))
    return uniformity, math.sqrt(max(variance, 0.0))


def evaluate_plan(
    network: Network,
    params: HydraulicParams,
    plan: Sequence[Group],
    source_profile: Sequence[float],
    unavailable: set[int] | None = None,
    start_time_h: float = 0.0,
    start_volumes: np.ndarray | None = None,
    horizon_h: float | None = None,
    caches: EvaluationCaches | None = None,
    collect_records: bool = False,
) -> tuple[dict[str, float], np.ndarray, list[dict[str, Any]]]:
    unavailable = set() if unavailable is None else set(unavailable)
    horizon = params.horizon_h if horizon_h is None else float(horizon_h)
    volumes = (
        np.zeros(len(network.emitters), dtype=float)
        if start_volumes is None
        else np.asarray(start_volumes, dtype=float).copy()
    )
    context_hash = _hash_context(source_profile, unavailable, start_time_h, volumes, horizon)
    plan_key = (
        MODEL_VERSION,
        network.fingerprint,
        context_hash,
        tuple((tuple(group.emitters), np.float64(group.duration_h).tobytes()) for group in plan),
    )
    cached_plan = caches.plan.get(plan_key) if caches is not None and caches.config.level >= 3 else None
    if cached_plan is not None and not collect_records:
        caches.stats["plan_hits"] += 1
        caches.plan.move_to_end(plan_key)
        metrics, cached_volumes = cached_plan
        return dict(metrics), cached_volumes.copy(), []
    if caches is not None and caches.config.level >= 3 and not collect_records:
        caches.stats["plan_misses"] += 1

    records: list[dict[str, Any]] = []
    pressure_values: list[float] = []
    pressure_weights: list[float] = []
    executed_groups = 0
    feasible_groups = 0
    pressure_violation_groups = 0
    velocity_violation_groups = 0
    nonconverged_segments = 0
    total_segments = 0
    max_pda_iterations = 0
    min_pressure = math.inf
    max_velocity = 0.0
    time_h = float(start_time_h)

    for group_index, group in enumerate(plan, start=1):
        if time_h >= horizon - EPS:
            break
        group_end = min(time_h + group.duration_h, horizon)
        group_active = tuple(e for e in group.emitters if e not in unavailable)
        group_executed = bool(group_active)
        group_feasible = True
        group_pressure_violation = False
        group_velocity_violation = False
        if group_executed:
            executed_groups += 1
        while time_h < group_end - EPS:
            segment_end = min(group_end, _next_hour_boundary(time_h))
            duration_h = segment_end - time_h
            source_head = source_head_at(source_profile, time_h)
            result = solve_group_pda(
                network, params, group.emitters, unavailable, source_head, caches=caches
            )
            total_segments += 1
            max_pda_iterations = max(max_pda_iterations, result.iterations)
            nonconverged_segments += int(not result.converged)
            if result.active_emitters:
                group_feasible = group_feasible and result.feasible
                group_pressure_violation = group_pressure_violation or result.pressure_violation_count > 0
                group_velocity_violation = group_velocity_violation or result.velocity_violation_count > 0
                min_pressure = min(min_pressure, float(np.min(result.emitter_pressure_m)))
                active_velocity = np.abs(result.velocities_ms[np.abs(result.pipe_flows_m3s) > EPS])
                if len(active_velocity):
                    max_velocity = max(max_velocity, float(np.max(active_velocity)))
                for local_idx, emitter_idx in enumerate(result.active_emitters):
                    volumes[emitter_idx] += result.delivered_flow_m3s[local_idx] * duration_h * 3600.0
                    pressure_values.append(float(result.emitter_pressure_m[local_idx]))
                    pressure_weights.append(duration_h)
            if collect_records:
                records.append(
                    {
                        "group_index": group_index,
                        "segment_start_h": time_h,
                        "segment_end_h": segment_end,
                        "duration_h": duration_h,
                        "source_head_m": source_head,
                        "stage": group.stage,
                        "emitters": ";".join(network.emitters[e] for e in group.emitters),
                        "active_emitters": ";".join(network.emitters[e] for e in result.active_emitters),
                        "min_pressure_m": float(np.min(result.emitter_pressure_m)) if len(result.emitter_pressure_m) else math.nan,
                        "max_velocity_ms": float(np.max(np.abs(result.velocities_ms))) if len(result.velocities_ms) else 0.0,
                        "pda_iterations": result.iterations,
                        "pda_converged": result.converged,
                        "feasible": result.feasible,
                    }
                )
            time_h = segment_end
        if group_executed:
            feasible_groups += int(group_feasible)
            pressure_violation_groups += int(group_pressure_violation)
            velocity_violation_groups += int(group_velocity_violation)

    target = params.target_per_emitter_m3
    capped = np.minimum(volumes, target)
    deficit = np.maximum(target - volumes, 0.0)
    total_target = target * len(network.emitters)
    uniformity, pressure_std = _weighted_pressure_uniformity(pressure_values, pressure_weights)
    metrics: dict[str, float] = {
        "achievement_pct": float(capped.sum() / total_target * 100.0),
        "unmet_emitter_ratio_pct": float(np.mean(deficit > 1.0e-9) * 100.0),
        "average_deficit_m3_per_emitter": float(np.mean(deficit)),
        "max_deficit_m3": float(np.max(deficit)),
        "pressure_uniformity_index_pct": uniformity,
        "pressure_std_m": pressure_std,
        "feasible_group_ratio_pct": feasible_groups / max(executed_groups, 1) * 100.0,
        "pressure_violation_group_ratio_pct": pressure_violation_groups / max(executed_groups, 1) * 100.0,
        "velocity_violation_group_ratio_pct": velocity_violation_groups / max(executed_groups, 1) * 100.0,
        "combined_violation_group_ratio_pct": (
            executed_groups - feasible_groups
        ) / max(executed_groups, 1) * 100.0,
        "min_pressure_m": min_pressure if math.isfinite(min_pressure) else math.nan,
        "max_velocity_ms": max_velocity,
        "executed_group_count": float(executed_groups),
        "end_time_h": time_h,
        "pda_nonconvergence_rate_pct": nonconverged_segments / max(total_segments, 1) * 100.0,
        "pda_max_iterations": float(max_pda_iterations),
    }
    if caches is not None and caches.config.level >= 3 and not collect_records:
        _put_lru(caches.plan, plan_key, (dict(metrics), volumes.copy()), caches.config.plan_max_items)
    return metrics, volumes, records


def objectives_from_metrics(metrics: dict[str, float]) -> tuple[np.ndarray, float]:
    uniformity = metrics["pressure_uniformity_index_pct"]
    if not math.isfinite(uniformity):
        uniformity = -100.0
    objectives = np.array(
        [
            100.0 - metrics["achievement_pct"],
            metrics["average_deficit_m3_per_emitter"],
            100.0 - uniformity,
        ],
        dtype=float,
    )
    constraint_violation = (
        metrics["combined_violation_group_ratio_pct"] / 100.0
        + metrics["pda_nonconvergence_rate_pct"] / 100.0
    )
    return objectives, float(constraint_violation)


def source_profile(seed: int, params: HydraulicParams, hours: int = 121) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(20.0, 30.0) for _ in range(hours)]


def causal_expected_profile(
    realized_profile: Sequence[float],
    decision_time_h: float,
    expected_head_m: float = 25.0,
) -> list[float]:
    output = [expected_head_m for _ in realized_profile]
    observed_until = int(min(max(math.floor(decision_time_h), 0), len(output) - 1))
    for index in range(observed_until + 1):
        output[index] = float(realized_profile[index])
    return output


def single_fault_scenario(seed: int, network: Network, params: HydraulicParams) -> list[Event]:
    rng = random.Random(seed)
    node = rng.choice(network.irrigation_nodes)
    fault_time = rng.uniform(8.0, 88.0)
    recovery_time = rng.uniform(fault_time + 4.0, params.horizon_h)
    return [Event(fault_time, "fault", node), Event(min(recovery_time, params.horizon_h), "recovery", node)]


def multi_fault_scenario(seed: int, network: Network, params: HydraulicParams) -> list[Event]:
    rng = random.Random(seed)
    count = rng.randint(1, 3)
    nodes = rng.sample(list(network.irrigation_nodes), k=count)
    events: list[Event] = []
    for node in nodes:
        fault_time = rng.uniform(0.0, 110.0)
        events.append(Event(fault_time, "fault", node))
        if rng.random() < 0.75:
            events.append(Event(rng.uniform(fault_time, params.horizon_h), "recovery", node))
    return sorted(events, key=lambda event: (event.time_h, 0 if event.kind == "fault" else 1))


def cache_statistics(caches: EvaluationCaches) -> dict[str, float]:
    output: dict[str, float] = {
        "cache_level": float(caches.config.level),
        "cache_estimated_bytes": float(caches.estimated_bytes()),
        "pipe_cache_items": float(len(caches.pipe)),
        "group_cache_items": float(len(caches.group)),
        "plan_cache_items": float(len(caches.plan)),
    }
    for name in ("pipe", "group", "plan"):
        hits = int(caches.stats[f"{name}_hits"])
        misses = int(caches.stats[f"{name}_misses"])
        output[f"{name}_cache_hits"] = float(hits)
        output[f"{name}_cache_misses"] = float(misses)
        output[f"{name}_cache_hit_rate_pct"] = hits / max(hits + misses, 1) * 100.0
    return output


def timed_call(function: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - started
