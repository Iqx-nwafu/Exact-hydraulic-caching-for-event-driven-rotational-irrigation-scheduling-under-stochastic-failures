from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


G = 9.80665
EPS = 1e-12
WATER_KINEMATIC_VISCOSITY_M2S = 1.0e-6


@dataclass(frozen=True)
class Params:
    design_head_m: float = 11.4
    design_flow_m3s: float = 0.013
    target_duration_h: float = 4.0
    horizon_h: float = 120.0
    group_size: int = 4
    min_velocity_ms: float = 0.3
    max_velocity_ms: float = 2.5
    source_head_min_m: float = 20.0
    source_head_max_m: float = 30.0
    local_loss_ratio: float = 0.1
    kinematic_viscosity_m2s: float = WATER_KINEMATIC_VISCOSITY_M2S

    @property
    def target_per_emitter_m3(self) -> float:
        return self.design_flow_m3s * self.target_duration_h * 3600.0


@dataclass(frozen=True)
class Network:
    nodes: pd.DataFrame
    pipes: pd.DataFrame
    node_z: dict[str, float]
    pipe_ids: list[str]
    from_nodes: list[str]
    to_nodes: list[str]
    lengths_m: np.ndarray
    diameters_m: np.ndarray
    areas_m2: np.ndarray
    roughness_mm: np.ndarray
    parent_pipe: dict[str, int]
    irrigation_nodes: list[str]
    emitters: list[str]
    emitter_to_node: list[str]
    emitter_index: dict[str, int]
    node_to_path: dict[str, tuple[int, ...]]


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


@dataclass
class Individual:
    x: np.ndarray
    objectives: tuple[float, float, float] | None = None
    cv: float = math.inf
    rank: int = 10**9
    crowding: float = 0.0
    metrics: dict[str, Any] | None = None

    @property
    def feasible(self) -> bool:
        return self.cv <= 1e-9


WORKER_NETWORK: Network | None = None
WORKER_PARAMS: Params | None = None
WORKER_ROUGHNESS: dict[str, float] | None = None
WORKER_CACHE_ENABLED = True
WORKER_GROUP_CACHE_MAX_ITEMS = 50000
WORKER_PLAN_CACHE_MAX_ITEMS = 20000
WORKER_PIPE_LOSS_CACHE_MAX_ITEMS = 20000
WORKER_GROUP_CACHE: OrderedDict[tuple[int, ...], dict[str, Any]] = OrderedDict()
WORKER_PIPE_LOSS_CACHE: OrderedDict[tuple[int, float], tuple[float, float]] = OrderedDict()
WORKER_CACHE_STATS: Counter[str] = Counter()
CACHE_STAT_KEYS = (
    "group_cache_hits",
    "group_cache_misses",
    "pipe_loss_cache_hits",
    "pipe_loss_cache_misses",
    "plan_cache_hits",
    "plan_cache_misses",
)


def node_key(node_id: str) -> tuple[int, str]:
    return (int(node_id[1:]), node_id) if node_id.startswith("J") and node_id[1:].isdigit() else (10**9, node_id)


def branch_position(node_id: str) -> tuple[int, int]:
    num = int(node_id[1:])
    if num >= 100:
        return 10, num - 100
    return num // 10, num % 10


def load_network(nodes_path: Path, pipes_path: Path, roughness_mm_by_material: dict[str, float]) -> Network:
    nodes = pd.read_excel(nodes_path)
    pipes = pd.read_excel(pipes_path)
    nodes.columns = [str(c).strip() for c in nodes.columns]
    pipes.columns = [str(c).strip() for c in pipes.columns]
    nodes = nodes.rename(columns={nodes.columns[0]: "node_id"})
    nodes["node_id"] = nodes["node_id"].astype(str).str.strip()
    pipes["Pipe ID"] = pipes["Pipe ID"].astype(str).str.strip()
    pipes["FromNode"] = pipes["FromNode"].astype(str).str.strip()
    pipes["ToNode"] = pipes["ToNode"].astype(str).str.strip()
    pipes["Material"] = pipes["Material"].astype(str).str.upper().str.strip()

    required_nodes = {"node_id", "X", "Y", "Z"}
    required_pipes = {"Pipe ID", "FromNode", "ToNode", "Length_m", "Diameter_m", "Material"}
    if not required_nodes.issubset(nodes.columns):
        raise ValueError(f"Nodes.xlsx missing columns: {sorted(required_nodes - set(nodes.columns))}")
    if not required_pipes.issubset(pipes.columns):
        raise ValueError(f"Pipes.xlsx missing columns: {sorted(required_pipes - set(pipes.columns))}")

    node_set = set(nodes["node_id"])
    missing_nodes = (set(pipes["FromNode"]) | set(pipes["ToNode"])) - node_set
    if missing_nodes:
        raise ValueError(f"Pipes reference missing nodes: {sorted(missing_nodes)}")
    missing_materials = sorted(set(pipes["Material"]) - set(roughness_mm_by_material))
    if missing_materials:
        raise ValueError(f"Missing roughness for materials: {missing_materials}")

    parent_pipe: dict[str, int] = {}
    for idx, to_node in enumerate(pipes["ToNode"].tolist()):
        if to_node in parent_pipe:
            raise ValueError(f"Node {to_node} has multiple upstream pipes; tree topology is assumed.")
        parent_pipe[to_node] = idx

    from_nodes = pipes["FromNode"].tolist()
    main_nodes = {f"J{i}" for i in range(11)}
    irrigation_nodes = [n for n in sorted(node_set, key=node_key) if n not in main_nodes]
    emitter_names = [f"{node}{side}" for node in irrigation_nodes for side in ("L", "R")]
    emitter_to_node = [name[:-1] for name in emitter_names]
    emitter_index = {name: i for i, name in enumerate(emitter_names)}
    node_to_path: dict[str, tuple[int, ...]] = {}
    for node in irrigation_nodes:
        cur = node
        path: list[int] = []
        seen: set[str] = set()
        while cur != "J0":
            if cur in seen:
                raise ValueError(f"Cycle detected while tracing {node}.")
            seen.add(cur)
            idx = parent_pipe[cur]
            path.append(idx)
            cur = from_nodes[idx]
        node_to_path[node] = tuple(reversed(path))

    diameters = pipes["Diameter_m"].astype(float).to_numpy()
    return Network(
        nodes=nodes,
        pipes=pipes,
        node_z=nodes.set_index("node_id")["Z"].astype(float).to_dict(),
        pipe_ids=pipes["Pipe ID"].tolist(),
        from_nodes=from_nodes,
        to_nodes=pipes["ToNode"].tolist(),
        lengths_m=pipes["Length_m"].astype(float).to_numpy(),
        diameters_m=diameters,
        areas_m2=math.pi * np.square(diameters) / 4.0,
        roughness_mm=np.array([roughness_mm_by_material[m] for m in pipes["Material"].tolist()], dtype=float),
        parent_pipe=parent_pipe,
        irrigation_nodes=irrigation_nodes,
        emitters=emitter_names,
        emitter_to_node=emitter_to_node,
        emitter_index=emitter_index,
        node_to_path=node_to_path,
    )


def colebrook_white_friction_factor(reynolds: float, roughness_m: float, diameter_m: float) -> float:
    if reynolds <= EPS:
        return 0.0
    if reynolds < 2000.0:
        return 64.0 / reynolds

    def turbulent(re_value: float) -> float:
        rr = max(roughness_m / max(diameter_m, EPS), 0.0)
        f = 1.0 / max((-1.8 * math.log10((rr / 3.7) ** 1.11 + 6.9 / max(re_value, EPS))) ** 2, EPS)
        for _ in range(30):
            inv = -2.0 * math.log10(rr / 3.7 + 2.51 / (re_value * math.sqrt(f)))
            new_f = 1.0 / max(inv * inv, EPS)
            if abs(new_f - f) < 1e-12:
                return float(new_f)
            f = new_f
        return float(f)

    if reynolds >= 4000.0:
        return turbulent(reynolds)

    t = (reynolds - 2000.0) / 2000.0
    smooth = t * t * (3.0 - 2.0 * t)
    return float((1.0 - smooth) * (64.0 / reynolds) + smooth * turbulent(4000.0))


def source_head_at(source_profile: list[float], time_h: float, default: float = 25.0) -> float:
    if not source_profile:
        return default
    idx = int(min(max(math.floor(time_h), 0), len(source_profile) - 1))
    return float(source_profile[idx])


def reset_worker_caches() -> None:
    WORKER_GROUP_CACHE.clear()
    WORKER_PIPE_LOSS_CACHE.clear()
    WORKER_CACHE_STATS.clear()


def cache_stats_snapshot() -> Counter[str]:
    return Counter({key: WORKER_CACHE_STATS.get(key, 0) for key in CACHE_STAT_KEYS})


def cache_stats_delta(start: Counter[str]) -> dict[str, Any]:
    out = {
        "cache_group_hits": int(WORKER_CACHE_STATS.get("group_cache_hits", 0) - start.get("group_cache_hits", 0)),
        "cache_group_misses": int(WORKER_CACHE_STATS.get("group_cache_misses", 0) - start.get("group_cache_misses", 0)),
        "cache_pipe_loss_hits": int(WORKER_CACHE_STATS.get("pipe_loss_cache_hits", 0) - start.get("pipe_loss_cache_hits", 0)),
        "cache_pipe_loss_misses": int(WORKER_CACHE_STATS.get("pipe_loss_cache_misses", 0) - start.get("pipe_loss_cache_misses", 0)),
        "cache_plan_hits": int(WORKER_CACHE_STATS.get("plan_cache_hits", 0) - start.get("plan_cache_hits", 0)),
        "cache_plan_misses": int(WORKER_CACHE_STATS.get("plan_cache_misses", 0) - start.get("plan_cache_misses", 0)),
    }
    group_total = out["cache_group_hits"] + out["cache_group_misses"]
    pipe_total = out["cache_pipe_loss_hits"] + out["cache_pipe_loss_misses"]
    plan_total = out["cache_plan_hits"] + out["cache_plan_misses"]
    out.update(
        {
            "cache_enabled": int(WORKER_CACHE_ENABLED),
            "cache_group_hit_rate_pct": out["cache_group_hits"] / max(group_total, 1) * 100.0,
            "cache_pipe_loss_hit_rate_pct": out["cache_pipe_loss_hits"] / max(pipe_total, 1) * 100.0,
            "cache_plan_hit_rate_pct": out["cache_plan_hits"] / max(plan_total, 1) * 100.0,
            "cache_group_size": len(WORKER_GROUP_CACHE),
            "cache_pipe_loss_size": len(WORKER_PIPE_LOSS_CACHE),
        }
    )
    return out


def put_lru(cache: OrderedDict, key: Any, value: Any, max_items: int) -> None:
    if max_items <= 0:
        return
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_items:
        cache.popitem(last=False)


def compute_hydraulic_base(
    network: Network,
    params: Params,
    active_emitters_key: tuple[int, ...],
) -> dict[str, Any]:
    node_counts = Counter(network.emitter_to_node[e] for e in active_emitters_key)
    node_demands = {node: count * params.design_flow_m3s for node, count in node_counts.items()}
    pipe_flows = np.zeros(len(network.pipe_ids), dtype=float)
    for node, demand in node_demands.items():
        for pipe_idx in network.node_to_path[node]:
            pipe_flows[pipe_idx] += demand
    active_mask = pipe_flows > EPS
    velocities = np.zeros_like(pipe_flows)
    velocities[active_mask] = pipe_flows[active_mask] / network.areas_m2[active_mask]
    reynolds = np.zeros_like(pipe_flows)
    reynolds[active_mask] = (
        np.abs(velocities[active_mask]) * network.diameters_m[active_mask] / params.kinematic_viscosity_m2s
    )
    factors = np.zeros_like(pipe_flows)
    roughness_m = network.roughness_mm / 1000.0
    friction_loss = np.zeros_like(pipe_flows)
    for idx in np.flatnonzero(active_mask):
        flow_key = round(float(pipe_flows[idx]), 12)
        pipe_cache_key = (int(idx), flow_key)
        cached_loss = WORKER_PIPE_LOSS_CACHE.get(pipe_cache_key) if WORKER_CACHE_ENABLED else None
        if cached_loss is not None:
            WORKER_CACHE_STATS["pipe_loss_cache_hits"] += 1
            WORKER_PIPE_LOSS_CACHE.move_to_end(pipe_cache_key)
            factors[idx], friction_loss[idx] = cached_loss
            continue
        if WORKER_CACHE_ENABLED:
            WORKER_CACHE_STATS["pipe_loss_cache_misses"] += 1
        factor = colebrook_white_friction_factor(
            float(reynolds[idx]), float(roughness_m[idx]), float(network.diameters_m[idx])
        )
        loss = (
            factor
            * (network.lengths_m[idx] / network.diameters_m[idx])
            * velocities[idx] ** 2
            / (2.0 * G)
        )
        factors[idx] = factor
        friction_loss[idx] = loss
        if WORKER_CACHE_ENABLED:
            put_lru(WORKER_PIPE_LOSS_CACHE, pipe_cache_key, (float(factor), float(loss)), WORKER_PIPE_LOSS_CACHE_MAX_ITEMS)

    node_path_loss = {
        node: float(friction_loss[list(network.node_to_path[node])].sum())
        for node in node_counts
    }
    velocity_violation_ms = 0.0
    velocity_violation_count = 0
    active_velocities = velocities[active_mask]
    for value in active_velocities:
        if value < params.min_velocity_ms:
            velocity_violation_ms += params.min_velocity_ms - float(value)
            velocity_violation_count += 1
        elif value > params.max_velocity_ms:
            velocity_violation_ms += float(value) - params.max_velocity_ms
            velocity_violation_count += 1

    return {
        "node_counts": dict(node_counts),
        "node_path_loss": node_path_loss,
        "pipe_flows": pipe_flows,
        "velocities": velocities,
        "friction_loss": friction_loss,
        "active_velocities": active_velocities,
        "max_velocity_ms": float(active_velocities.max()) if len(active_velocities) else 0.0,
        "min_velocity_ms": float(active_velocities.min()) if len(active_velocities) else 0.0,
        "velocity_violation_ms": velocity_violation_ms,
        "velocity_violation_count": velocity_violation_count,
    }


def hydraulic_base(
    network: Network,
    params: Params,
    active_emitters_key: tuple[int, ...],
) -> dict[str, Any]:
    if not WORKER_CACHE_ENABLED:
        return compute_hydraulic_base(network, params, active_emitters_key)
    cached = WORKER_GROUP_CACHE.get(active_emitters_key)
    if cached is not None:
        WORKER_CACHE_STATS["group_cache_hits"] += 1
        WORKER_GROUP_CACHE.move_to_end(active_emitters_key)
        return cached
    WORKER_CACHE_STATS["group_cache_misses"] += 1
    base = compute_hydraulic_base(network, params, active_emitters_key)
    put_lru(WORKER_GROUP_CACHE, active_emitters_key, base, WORKER_GROUP_CACHE_MAX_ITEMS)
    return base


def isolated_emitters_after_fault(network: Network, fault_node: str) -> set[int]:
    branch, position = branch_position(fault_node)
    nodes = [f"J{branch}{p}" if branch < 10 else f"J10{p}" for p in range(position, 7)]
    out: set[int] = set()
    for node in nodes:
        for side in ("L", "R"):
            name = f"{node}{side}"
            if name in network.emitter_index:
                out.add(network.emitter_index[name])
    return out


def hydraulic_group(
    network: Network,
    params: Params,
    emitters: tuple[int, ...],
    unavailable: set[int],
    source_head_m: float,
) -> dict[str, Any]:
    active_emitters = tuple(e for e in emitters if e not in unavailable)
    active_emitters_key = tuple(sorted(active_emitters))
    base = hydraulic_base(network, params, active_emitters_key)
    node_counts = base["node_counts"]
    node_pressure: dict[str, float] = {}
    delivered_flow: dict[int, float] = {}
    pressure_violation_m = 0.0
    pressure_violation_count = 0
    for node, count in node_counts.items():
        path_loss = float(base["node_path_loss"][node])
        pressure = float(source_head_m - path_loss - network.node_z[node])
        node_pressure[node] = pressure
        if pressure < params.design_head_m:
            pressure_violation_m += params.design_head_m - pressure
            pressure_violation_count += 1
    pressure_values = [node_pressure[network.emitter_to_node[e]] for e in active_emitters]
    for emitter in active_emitters:
        pressure = node_pressure[network.emitter_to_node[emitter]]
        delivered_flow[emitter] = params.design_flow_m3s * min(max(pressure / params.design_head_m, 0.0), 1.0)

    feasible = (
        bool(active_emitters)
        and pressure_violation_count == 0
        and int(base["velocity_violation_count"]) == 0
    )
    return {
        "active_emitters": list(active_emitters),
        "node_pressure": node_pressure,
        "pressure_values": pressure_values,
        "delivered_flow": delivered_flow,
        "pipe_flows": base["pipe_flows"],
        "velocities": base["velocities"],
        "friction_loss": base["friction_loss"],
        "min_pressure_m": min(node_pressure.values()) if node_pressure else math.nan,
        "mean_pressure_m": float(np.mean(pressure_values)) if pressure_values else math.nan,
        "pressure_std_m": float(np.std(pressure_values)) if len(pressure_values) > 1 else 0.0,
        "max_velocity_ms": base["max_velocity_ms"],
        "min_velocity_ms": base["min_velocity_ms"],
        "pressure_violation_m": pressure_violation_m,
        "pressure_violation_count": pressure_violation_count,
        "velocity_violation_ms": base["velocity_violation_ms"],
        "velocity_violation_count": base["velocity_violation_count"],
        "feasible": feasible,
    }


def original_plan(network: Network, params: Params) -> list[Group]:
    groups: list[Group] = []
    for branch in range(1, 11):
        nodes = [f"J{branch}{i}" for i in range(1, 7)] if branch < 10 else [f"J10{i}" for i in range(1, 7)]
        for start in (0, 2, 4):
            emitters = tuple(network.emitter_index[f"{node}{side}"] for node in nodes[start : start + 2] for side in ("L", "R"))
            groups.append(Group(emitters, params.target_duration_h, "original"))
    return groups


def groups_from_order(order: list[int], params: Params, stage: str) -> list[Group]:
    groups = []
    for start in range(0, len(order), params.group_size):
        chunk = tuple(order[start : start + params.group_size])
        if chunk:
            groups.append(Group(chunk, params.target_duration_h, stage))
    return groups


def random_plan(network: Network, params: Params, rng: random.Random) -> list[Group]:
    order = list(range(len(network.emitters)))
    rng.shuffle(order)
    return groups_from_order(order, params, "random")


def keys_to_plan(keys: np.ndarray, candidates: list[int], params: Params, stage: str) -> list[Group]:
    order_idx = np.argsort(keys, kind="mergesort").tolist()
    order = [candidates[i] for i in order_idx]
    return groups_from_order(order, params, stage)


def plan_cache_key(plan: list[Group]) -> tuple[tuple[tuple[int, ...], int], ...]:
    return tuple((tuple(group.emitters), int(round(group.duration_h * 3600000.0))) for group in plan)


def evaluate_plan(
    network: Network,
    params: Params,
    plan: list[Group],
    source_profile: list[float],
    unavailable: set[int] | None = None,
    start_time_h: float = 0.0,
    start_volumes: np.ndarray | None = None,
    horizon_h: float | None = None,
    collect_records: bool = False,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    unavailable = set() if unavailable is None else set(unavailable)
    horizon = params.horizon_h if horizon_h is None else horizon_h
    volumes = np.zeros(len(network.emitters), dtype=float) if start_volumes is None else start_volumes.copy()
    records: list[dict[str, Any]] = []
    pressure_values: list[float] = []
    feasible_groups = 0
    executed_groups = 0
    pressure_violation_groups = 0
    velocity_violation_groups = 0
    min_pressure = math.inf
    max_velocity = 0.0
    t = start_time_h
    for seq, group in enumerate(plan, start=1):
        if t >= horizon - EPS:
            break
        duration = min(group.duration_h, horizon - t)
        if duration <= EPS:
            break
        source_head = source_head_at(source_profile, t)
        result = hydraulic_group(network, params, group.emitters, unavailable, source_head)
        if result["active_emitters"]:
            executed_groups += 1
            feasible_groups += int(result["feasible"])
            pressure_violation_groups += int(result["pressure_violation_count"] > 0)
            velocity_violation_groups += int(result["velocity_violation_count"] > 0)
            min_pressure = min(min_pressure, float(result["min_pressure_m"]))
            max_velocity = max(max_velocity, float(result["max_velocity_ms"]))
            pressure_values.extend(result["pressure_values"])
            for emitter, flow in result["delivered_flow"].items():
                volumes[emitter] += flow * duration * 3600.0
        if collect_records:
            records.append(
                {
                    "seq": seq,
                    "stage": group.stage,
                    "start_h": t,
                    "end_h": t + duration,
                    "duration_h": duration,
                    "source_head_m": source_head,
                    "emitters": ";".join(network.emitters[e] for e in group.emitters),
                    "active_emitters": ";".join(network.emitters[e] for e in result["active_emitters"]),
                    "min_pressure_m": result["min_pressure_m"],
                    "mean_pressure_m": result["mean_pressure_m"],
                    "pressure_std_m": result["pressure_std_m"],
                    "max_velocity_ms": result["max_velocity_ms"],
                    "feasible": result["feasible"],
                }
            )
        t += duration
    metrics = summarize_volumes(network, params, volumes, pressure_values, feasible_groups, executed_groups)
    metrics.update(
        {
            "executed_group_count": executed_groups,
            "feasible_group_ratio_pct": feasible_groups / max(executed_groups, 1) * 100.0,
            "pressure_violation_group_ratio_pct": pressure_violation_groups / max(executed_groups, 1) * 100.0,
            "velocity_violation_group_ratio_pct": velocity_violation_groups / max(executed_groups, 1) * 100.0,
            "min_pressure_m": min_pressure if math.isfinite(min_pressure) else math.nan,
            "max_velocity_ms": max_velocity,
            "end_time_h": t,
        }
    )
    return metrics, volumes, records


def summarize_volumes(
    network: Network,
    params: Params,
    volumes: np.ndarray,
    pressure_values: list[float],
    feasible_groups: int,
    executed_groups: int,
) -> dict[str, Any]:
    target = params.target_per_emitter_m3
    capped = np.minimum(volumes, target)
    deficit = np.maximum(target - volumes, 0.0)
    unmet = deficit > 1e-9
    total_target = target * len(volumes)
    if pressure_values:
        p = np.array(pressure_values, dtype=float)
        p_mean = float(np.mean(p))
        pressure_uniformity = (1.0 - float(np.mean(np.abs(p - p_mean))) / max(abs(p_mean), EPS)) * 100.0
        pressure_std = float(np.std(p, ddof=1)) if len(p) > 1 else 0.0
    else:
        pressure_uniformity = math.nan
        pressure_std = math.nan
    return {
        "achievement_pct": float(capped.sum() / total_target * 100.0),
        "unmet_emitter_ratio_pct": float(unmet.mean() * 100.0),
        "unmet_emitters": int(unmet.sum()),
        "average_deficit_m3_per_emitter": float(deficit.mean()),
        "max_deficit_m3": float(deficit.max()),
        "total_deficit_m3": float(deficit.sum()),
        "pressure_uniformity_index_pct": pressure_uniformity,
        "pressure_std_m": pressure_std,
        "feasible_group_ratio_pct": feasible_groups / max(executed_groups, 1) * 100.0,
    }


def plan_objectives(
    network: Network,
    params: Params,
    plan: list[Group],
    source_profile: list[float],
    unavailable: set[int] | None = None,
    start_time_h: float = 0.0,
    start_volumes: np.ndarray | None = None,
    horizon_h: float | None = None,
) -> tuple[tuple[float, float, float], float, dict[str, Any]]:
    metrics, _, _ = evaluate_plan(
        network,
        params,
        plan,
        source_profile,
        unavailable=unavailable,
        start_time_h=start_time_h,
        start_volumes=start_volumes,
        horizon_h=horizon_h,
        collect_records=False,
    )
    cv = (
        metrics["pressure_violation_group_ratio_pct"] / 100.0
        + metrics["velocity_violation_group_ratio_pct"] / 100.0
        + max(0.0, 100.0 - metrics["feasible_group_ratio_pct"]) / 100.0
    )
    objectives = (
        100.0 - metrics["achievement_pct"],
        metrics["average_deficit_m3_per_emitter"],
        100.0 - metrics["pressure_uniformity_index_pct"] if not math.isnan(metrics["pressure_uniformity_index_pct"]) else 100.0,
    )
    return objectives, cv, metrics


def dominates(a: Individual, b: Individual) -> bool:
    if a.feasible and not b.feasible:
        return True
    if b.feasible and not a.feasible:
        return False
    if not a.feasible and not b.feasible:
        return a.cv < b.cv - 1e-12
    assert a.objectives is not None and b.objectives is not None
    return all(x <= y + 1e-12 for x, y in zip(a.objectives, b.objectives)) and any(
        x < y - 1e-12 for x, y in zip(a.objectives, b.objectives)
    )


def fast_non_dominated_sort(pop: list[Individual]) -> list[list[int]]:
    dominates_set: list[list[int]] = [[] for _ in pop]
    dominated_count = [0 for _ in pop]
    fronts: list[list[int]] = [[]]
    for i, a in enumerate(pop):
        for j, b in enumerate(pop):
            if i == j:
                continue
            if dominates(a, b):
                dominates_set[i].append(j)
            elif dominates(b, a):
                dominated_count[i] += 1
        if dominated_count[i] == 0:
            a.rank = 0
            fronts[0].append(i)
    k = 0
    while fronts[k]:
        next_front: list[int] = []
        for i in fronts[k]:
            for j in dominates_set[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    pop[j].rank = k + 1
                    next_front.append(j)
        k += 1
        fronts.append(next_front)
    return fronts[:-1]


def assign_crowding(pop: list[Individual], front: list[int]) -> None:
    if not front:
        return
    for idx in front:
        pop[idx].crowding = 0.0
    if len(front) <= 2:
        for idx in front:
            pop[idx].crowding = math.inf
        return
    for m in range(3):
        front_sorted = sorted(front, key=lambda idx: pop[idx].objectives[m] if pop[idx].objectives else math.inf)
        pop[front_sorted[0]].crowding = math.inf
        pop[front_sorted[-1]].crowding = math.inf
        min_v = pop[front_sorted[0]].objectives[m]
        max_v = pop[front_sorted[-1]].objectives[m]
        span = max(max_v - min_v, EPS)
        for pos in range(1, len(front_sorted) - 1):
            prev_v = pop[front_sorted[pos - 1]].objectives[m]
            next_v = pop[front_sorted[pos + 1]].objectives[m]
            pop[front_sorted[pos]].crowding += (next_v - prev_v) / span


def nsga2_select(pop: list[Individual], size: int) -> list[Individual]:
    fronts = fast_non_dominated_sort(pop)
    selected: list[Individual] = []
    for front in fronts:
        assign_crowding(pop, front)
        if len(selected) + len(front) <= size:
            selected.extend(pop[i] for i in front)
        else:
            ranked = sorted((pop[i] for i in front), key=lambda ind: ind.crowding, reverse=True)
            selected.extend(ranked[: size - len(selected)])
            break
    return selected


def objective_matrix(pop: list[Individual]) -> np.ndarray:
    return np.array([ind.objectives if ind.objectives is not None else (math.inf, math.inf, math.inf) for ind in pop], dtype=float)


def feasible_pool(pop: list[Individual]) -> list[Individual]:
    return sorted(pop, key=lambda ind: (not ind.feasible, ind.cv))


def spea2_select(pop: list[Individual], size: int) -> list[Individual]:
    n = len(pop)
    strength = np.zeros(n, dtype=float)
    raw = np.zeros(n, dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j and dominates(pop[i], pop[j]):
                strength[i] += 1.0
    for j in range(n):
        for i in range(n):
            if i != j and dominates(pop[i], pop[j]):
                raw[j] += strength[i]
    obj = objective_matrix(pop)
    finite = np.isfinite(obj).all(axis=1)
    density = np.ones(n, dtype=float)
    if finite.sum() > 1:
        norm = obj.copy()
        mins = np.nanmin(norm[finite], axis=0)
        maxs = np.nanmax(norm[finite], axis=0)
        norm = (norm - mins) / np.maximum(maxs - mins, EPS)
        k = max(1, int(math.sqrt(n)))
        for i in range(n):
            if not finite[i]:
                density[i] = 1.0
                continue
            dist = np.linalg.norm(norm[finite] - norm[i], axis=1)
            dist.sort()
            sigma = dist[min(k, len(dist) - 1)]
            density[i] = 1.0 / (sigma + 2.0)
    fitness = raw + density + np.array([0.0 if ind.feasible else 1e6 + ind.cv for ind in pop])
    return [pop[i] for i in np.argsort(fitness, kind="mergesort")[:size]]


def reference_vectors_3d(divisions: int = 8) -> np.ndarray:
    vecs = []
    for i in range(divisions + 1):
        for j in range(divisions + 1 - i):
            k = divisions - i - j
            v = np.array([i, j, k], dtype=float) / max(divisions, 1)
            if np.linalg.norm(v) > EPS:
                vecs.append(v / np.linalg.norm(v))
    return np.array(vecs, dtype=float)


def rvea_select(pop: list[Individual], size: int) -> list[Individual]:
    pool = feasible_pool(pop)
    feasible = [ind for ind in pool if ind.feasible and ind.objectives is not None]
    if not feasible:
        return pool[:size]
    obj = objective_matrix(feasible)
    mins = obj.min(axis=0)
    norm = (obj - mins) / np.maximum(obj.max(axis=0) - mins, EPS)
    refs = reference_vectors_3d(8)
    selected_idx: set[int] = set()
    for ref in refs:
        dots = norm @ ref / np.maximum(np.linalg.norm(norm, axis=1) * np.linalg.norm(ref), EPS)
        angles = np.arccos(np.clip(dots, -1.0, 1.0))
        distances = np.linalg.norm(norm, axis=1)
        apd = distances * (1.0 + angles)
        selected_idx.add(int(np.argmin(apd)))
        if len(selected_idx) >= size:
            break
    ranked = [feasible[i] for i in selected_idx]
    if len(ranked) < size:
        remaining = [ind for i, ind in enumerate(feasible) if i not in selected_idx]
        remaining.sort(key=lambda ind: ind.objectives)
        ranked.extend(remaining[: size - len(ranked)])
    if len(ranked) < size:
        ranked_ids = {id(ind) for ind in ranked}
        ranked.extend([ind for ind in pool if id(ind) not in ranked_ids][: size - len(ranked)])
    return ranked[:size]


def moead_select(pop: list[Individual], size: int) -> list[Individual]:
    pool = feasible_pool(pop)
    feasible = [ind for ind in pool if ind.feasible and ind.objectives is not None]
    if not feasible:
        return pool[:size]
    obj = objective_matrix(feasible)
    ideal = obj.min(axis=0)
    weights = reference_vectors_3d(10)
    selected_idx: set[int] = set()
    for weight in weights:
        w = np.maximum(weight, 1e-6)
        score = np.max(w * np.abs(obj - ideal), axis=1)
        selected_idx.add(int(np.argmin(score)))
        if len(selected_idx) >= size:
            break
    ranked = [feasible[i] for i in selected_idx]
    if len(ranked) < size:
        remaining = [ind for i, ind in enumerate(feasible) if i not in selected_idx]
        remaining.sort(key=lambda ind: ind.objectives)
        ranked.extend(remaining[: size - len(ranked)])
    if len(ranked) < size:
        ranked_ids = {id(ind) for ind in ranked}
        ranked.extend([ind for ind in pool if id(ind) not in ranked_ids][: size - len(ranked)])
    return ranked[:size]


def select_population(pop: list[Individual], size: int, algorithm: str) -> list[Individual]:
    alg = algorithm.lower()
    if alg == "nsga2":
        return nsga2_select(pop, size)
    if alg == "spea2":
        return spea2_select(pop, size)
    if alg == "rvea":
        return rvea_select(pop, size)
    if alg == "moead":
        return moead_select(pop, size)
    raise ValueError(f"Unsupported optimizer algorithm: {algorithm}")


def evaluate_population(
    pop: list[Individual],
    candidates: list[int],
    params: Params,
    source_profile: list[float],
    unavailable: set[int],
    start_time_h: float,
    start_volumes: np.ndarray | None,
    horizon_h: float,
    plan_cache: OrderedDict[tuple[tuple[tuple[int, ...], int], ...], tuple[tuple[float, float, float], float, dict[str, Any]]] | None = None,
) -> None:
    assert WORKER_NETWORK is not None
    network = WORKER_NETWORK
    for ind in pop:
        plan = keys_to_plan(ind.x, candidates, params, "nsga2")
        cache_key = plan_cache_key(plan) if WORKER_CACHE_ENABLED and plan_cache is not None else None
        cached = plan_cache.get(cache_key) if cache_key is not None else None
        if cached is not None:
            WORKER_CACHE_STATS["plan_cache_hits"] += 1
            plan_cache.move_to_end(cache_key)
            objectives, cv, metrics = cached
            metrics = dict(metrics)
        else:
            if cache_key is not None:
                WORKER_CACHE_STATS["plan_cache_misses"] += 1
            objectives, cv, metrics = plan_objectives(
                network,
                params,
                plan,
                source_profile,
                unavailable,
                start_time_h,
                start_volumes,
                horizon_h,
            )
            if cache_key is not None:
                put_lru(plan_cache, cache_key, (objectives, cv, dict(metrics)), WORKER_PLAN_CACHE_MAX_ITEMS)
        ind.objectives = objectives
        ind.cv = cv
        ind.metrics = metrics


def choose_solution(pop: list[Individual]) -> Individual:
    fronts = fast_non_dominated_sort(pop)
    first = [pop[i] for i in fronts[0]]
    return sorted(
        first,
        key=lambda ind: (
            -(ind.metrics or {}).get("achievement_pct", -math.inf),
            (ind.metrics or {}).get("average_deficit_m3_per_emitter", math.inf),
            -((ind.metrics or {}).get("pressure_uniformity_index_pct", -math.inf)),
            ind.cv,
        ),
    )[0]


def optimize_nsga2_plan(
    candidates: list[int],
    params: Params,
    source_profile: list[float],
    seed: int,
    population: int,
    generations: int,
    unavailable: set[int] | None = None,
    start_time_h: float = 0.0,
    start_volumes: np.ndarray | None = None,
    horizon_h: float | None = None,
    initial_orders: list[list[int]] | None = None,
    algorithm: str = "nsga2",
) -> tuple[list[Group], dict[str, Any], float, list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    unavailable = set() if unavailable is None else set(unavailable)
    horizon = params.horizon_h if horizon_h is None else horizon_h
    dim = len(candidates)
    if dim == 0:
        return [], {}, 0.0, []
    cache_stats_start = cache_stats_snapshot()
    plan_cache: OrderedDict[tuple[tuple[tuple[int, ...], int], ...], tuple[tuple[float, float, float], float, dict[str, Any]]] | None = (
        OrderedDict() if WORKER_CACHE_ENABLED else None
    )
    t0 = time.perf_counter()
    pop: list[Individual] = []
    initial_orders = initial_orders or []
    for order in initial_orders:
        rank = {em: i for i, em in enumerate(order) if em in candidates}
        keys = np.array([(rank.get(em, dim) + rng.random() * 0.01) / max(dim, 1) for em in candidates], dtype=float)
        pop.append(Individual(keys))
    while len(pop) < population:
        pop.append(Individual(rng.random(dim)))
    evaluate_population(pop, candidates, params, source_profile, unavailable, start_time_h, start_volumes, horizon, plan_cache)
    trace: list[dict[str, Any]] = []
    for gen in range(generations):
        fronts = fast_non_dominated_sort(pop)
        for front in fronts:
            assign_crowding(pop, front)
        offspring: list[Individual] = []
        while len(offspring) < population:
            a, b = rng.choice(pop, size=2, replace=False)
            parent1 = a if (a.rank, -a.crowding) < (b.rank, -b.crowding) else b
            c, d = rng.choice(pop, size=2, replace=False)
            parent2 = c if (c.rank, -c.crowding) < (d.rank, -d.crowding) else d
            alpha = rng.random(dim)
            child_x = alpha * parent1.x + (1.0 - alpha) * parent2.x
            mutation_mask = rng.random(dim) < 0.08
            child_x[mutation_mask] += rng.normal(0.0, 0.12, size=int(mutation_mask.sum()))
            child_x = np.clip(child_x, 0.0, 1.0)
            offspring.append(Individual(child_x))
        evaluate_population(offspring, candidates, params, source_profile, unavailable, start_time_h, start_volumes, horizon, plan_cache)
        pop = select_population(pop + offspring, population, algorithm)
        best = choose_solution(pop)
        trace.append(
            {
                "generation": gen,
                "algorithm": algorithm,
                "best_achievement_pct": best.metrics.get("achievement_pct", math.nan) if best.metrics else math.nan,
                "best_avg_deficit_m3": best.metrics.get("average_deficit_m3_per_emitter", math.nan) if best.metrics else math.nan,
                "best_pressure_uniformity_index_pct": best.metrics.get("pressure_uniformity_index_pct", math.nan) if best.metrics else math.nan,
                "best_cv": best.cv,
            }
        )
    best = choose_solution(pop)
    elapsed = time.perf_counter() - t0
    plan = keys_to_plan(best.x, candidates, params, "nsga2")
    metrics_out = dict(best.metrics or {})
    metrics_out.update(cache_stats_delta(cache_stats_start))
    metrics_out["cache_plan_size"] = len(plan_cache) if plan_cache is not None else 0
    return plan, metrics_out, elapsed, trace


def process_events_until(events: list[Event], idx: int, time_h: float, active_faults: set[str]) -> tuple[int, bool]:
    changed = False
    while idx < len(events) and events[idx].time_h <= time_h + EPS:
        if events[idx].kind == "fault":
            active_faults.add(events[idx].node)
            changed = True
        elif events[idx].kind == "recovery":
            active_faults.discard(events[idx].node)
            changed = True
        idx += 1
    return idx, changed


def unavailable_from_faults(network: Network, active_faults: set[str]) -> set[int]:
    out: set[int] = set()
    for node in active_faults:
        out.update(isolated_emitters_after_fault(network, node))
    return out


def simulate_plan_with_events(
    network: Network,
    params: Params,
    initial_plan: list[Group],
    source_profile: list[float],
    events: list[Event],
    dynamic: bool,
    optimizer_seed: int,
    population: int,
    generations: int,
    optimizer_algorithm: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    events = sorted(events, key=lambda e: (e.time_h, 0 if e.kind == "fault" else 1))
    volumes = np.zeros(len(network.emitters), dtype=float)
    active_faults: set[str] = set()
    event_idx = 0
    t = 0.0
    plan = initial_plan
    plan_idx = 0
    records: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    pressure_values: list[float] = []
    feasible_groups = 0
    executed_groups = 0
    reopt_count = 0
    reopt_time_s = 0.0
    reopt_cache_counts: Counter[str] = Counter()
    affected_emitters: set[int] = set()
    for event in events:
        if event.kind == "fault":
            affected_emitters.update(isolated_emitters_after_fault(network, event.node))

    while t < params.horizon_h - EPS:
        event_idx, event_changed = process_events_until(events, event_idx, t, active_faults)
        if dynamic and event_changed:
            plan_idx = len(plan)
        unavailable = unavailable_from_faults(network, active_faults)
        if dynamic and (plan_idx >= len(plan)):
            candidates = [i for i in range(len(network.emitters)) if i not in unavailable and volumes[i] < params.target_per_emitter_m3 - 1e-9]
            plan, opt_metrics, elapsed, trace = optimize_nsga2_plan(
                candidates,
                params,
                source_profile,
                optimizer_seed + 1000 * (reopt_count + 1),
                population,
                max(10, generations // 2),
                unavailable=unavailable,
                start_time_h=t,
                start_volumes=volumes,
                horizon_h=params.horizon_h,
                algorithm=optimizer_algorithm,
            )
            for key in (
                "cache_group_hits",
                "cache_group_misses",
                "cache_pipe_loss_hits",
                "cache_pipe_loss_misses",
                "cache_plan_hits",
                "cache_plan_misses",
            ):
                reopt_cache_counts[key] += int(opt_metrics.get(key, 0))
            for row in trace:
                row.update({"reopt_round": reopt_count + 1, "reopt_time_h": t, "optimizer_algorithm": optimizer_algorithm})
            traces.extend(trace)
            reopt_time_s += elapsed
            reopt_count += 1
            plan_idx = 0
            if not plan:
                break
        if plan_idx >= len(plan):
            break
        group = plan[plan_idx]
        next_event = events[event_idx] if event_idx < len(events) else None
        group_end = min(t + group.duration_h, params.horizon_h)
        if next_event and next_event.time_h < group_end - EPS and next_event.kind == "fault":
            duration = next_event.time_h - t
            stop_after = True
        else:
            duration = group_end - t
            stop_after = False
        if duration <= EPS:
            event_idx, event_changed = process_events_until(events, event_idx, t + EPS, active_faults)
            if dynamic and event_changed:
                plan_idx = len(plan)
            continue

        source_head = source_head_at(source_profile, t)
        result = hydraulic_group(network, params, group.emitters, unavailable, source_head)
        if result["active_emitters"]:
            executed_groups += 1
            feasible_groups += int(result["feasible"])
            pressure_values.extend(result["pressure_values"])
            for emitter, flow in result["delivered_flow"].items():
                volumes[emitter] += flow * duration * 3600.0
        records.append(
            {
                "start_h": t,
                "end_h": t + duration,
                "duration_h": duration,
                "stage": "dynamic" if dynamic else "static",
                "optimizer_algorithm": optimizer_algorithm,
                "source_head_m": source_head,
                "active_faults": ";".join(sorted(active_faults)),
                "emitters": ";".join(network.emitters[e] for e in group.emitters),
                "active_emitters": ";".join(network.emitters[e] for e in result["active_emitters"]),
                "min_pressure_m": result["min_pressure_m"],
                "mean_pressure_m": result["mean_pressure_m"],
                "feasible": result["feasible"],
            }
        )
        t += duration
        if stop_after:
            event_idx, _ = process_events_until(events, event_idx, t, active_faults)
            if dynamic:
                plan_idx = len(plan)
            continue
        plan_idx += 1
        if next_event and next_event.time_h < group_end - EPS and next_event.kind == "recovery":
            event_idx, _ = process_events_until(events, event_idx, group_end, active_faults)
            if dynamic:
                plan_idx = len(plan)

    metrics = summarize_volumes(network, params, volumes, pressure_values, feasible_groups, executed_groups)
    affected = sorted(affected_emitters)
    if affected:
        target = params.target_per_emitter_m3
        affected_deficit = np.maximum(target - volumes[affected], 0.0)
        affected_completion = np.minimum(volumes[affected], target) / target
        metrics.update(
            {
                "affected_emitters": len(affected),
                "affected_target_met_pct": float((affected_deficit <= 1e-9).mean() * 100.0),
                "affected_mean_completion_pct": float(affected_completion.mean() * 100.0),
                "affected_average_deficit_m3": float(affected_deficit.mean()),
                "affected_max_deficit_m3": float(affected_deficit.max()),
            }
        )
    else:
        metrics.update(
            {
                "affected_emitters": 0,
                "affected_target_met_pct": math.nan,
                "affected_mean_completion_pct": math.nan,
                "affected_average_deficit_m3": math.nan,
                "affected_max_deficit_m3": math.nan,
            }
        )
    metrics.update(
        {
            "executed_group_count": executed_groups,
            "feasible_group_ratio_pct": feasible_groups / max(executed_groups, 1) * 100.0,
            "reoptimization_count": reopt_count,
            "total_reoptimization_time_s": reopt_time_s,
            "mean_reoptimization_time_s": reopt_time_s / max(reopt_count, 1),
            "end_time_h": t,
            "optimizer_algorithm": optimizer_algorithm,
            "cache_enabled": int(WORKER_CACHE_ENABLED),
            "cache_group_hits": int(reopt_cache_counts["cache_group_hits"]),
            "cache_group_misses": int(reopt_cache_counts["cache_group_misses"]),
            "cache_pipe_loss_hits": int(reopt_cache_counts["cache_pipe_loss_hits"]),
            "cache_pipe_loss_misses": int(reopt_cache_counts["cache_pipe_loss_misses"]),
            "cache_plan_hits": int(reopt_cache_counts["cache_plan_hits"]),
            "cache_plan_misses": int(reopt_cache_counts["cache_plan_misses"]),
            "cache_group_hit_rate_pct": reopt_cache_counts["cache_group_hits"]
            / max(reopt_cache_counts["cache_group_hits"] + reopt_cache_counts["cache_group_misses"], 1)
            * 100.0,
            "cache_pipe_loss_hit_rate_pct": reopt_cache_counts["cache_pipe_loss_hits"]
            / max(reopt_cache_counts["cache_pipe_loss_hits"] + reopt_cache_counts["cache_pipe_loss_misses"], 1)
            * 100.0,
            "cache_plan_hit_rate_pct": reopt_cache_counts["cache_plan_hits"]
            / max(reopt_cache_counts["cache_plan_hits"] + reopt_cache_counts["cache_plan_misses"], 1)
            * 100.0,
        }
    )
    return metrics, records, traces


def source_profile(seed: int, params: Params, hours: int = 121) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(params.source_head_min_m, params.source_head_max_m) for _ in range(hours)]


def single_fault_scenario(seed: int, network: Network, params: Params) -> list[Event]:
    rng = random.Random(seed)
    node = rng.choice(network.irrigation_nodes)
    fault_t = rng.uniform(8.0, 88.0)
    recovery_t = rng.uniform(fault_t + 4.0, params.horizon_h)
    return [Event(fault_t, "fault", node), Event(min(recovery_t, params.horizon_h), "recovery", node)]


def multi_fault_scenario(seed: int, network: Network, params: Params) -> list[Event]:
    rng = random.Random(seed)
    n_faults = rng.randint(1, 3)
    nodes = rng.sample(network.irrigation_nodes, k=n_faults)
    events: list[Event] = []
    for node in nodes:
        fault_t = rng.uniform(0.0, 110.0)
        events.append(Event(fault_t, "fault", node))
        if rng.random() < 0.75:
            events.append(Event(rng.uniform(fault_t, params.horizon_h), "recovery", node))
    return sorted(events, key=lambda e: (e.time_h, 0 if e.kind == "fault" else 1))


def add_identity(row: dict[str, Any], experiment: str, scenario_id: int, policy: str, repeat: int, seed: int) -> dict[str, Any]:
    out = dict(row)
    out.update({"experiment": experiment, "scenario_id": scenario_id, "policy": policy, "repeat": repeat, "seed": seed})
    return out


def copy_cache_metrics(target: dict[str, Any], source: dict[str, Any], prefix: str = "") -> None:
    for key, value in source.items():
        if key.startswith("cache_"):
            target[f"{prefix}{key}"] = value


def run_normal_task(task: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assert WORKER_NETWORK is not None and WORKER_PARAMS is not None
    network, params = WORKER_NETWORK, WORKER_PARAMS
    sid = int(task["scenario_id"])
    seed = int(task["seed"])
    profile = source_profile(seed, params)
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    original = original_plan(network, params)
    metrics, _, _ = evaluate_plan(network, params, original, profile)
    metrics["optimizer_algorithm"] = "none"
    rows.append(add_identity(metrics, "normal", sid, "original_near_far", 0, seed))
    for repeat in range(1, int(task["random_repeats"]) + 1):
        plan = random_plan(network, params, random.Random(seed + repeat * 101))
        metrics, _, _ = evaluate_plan(network, params, plan, profile)
        metrics["optimizer_algorithm"] = "none"
        rows.append(add_identity(metrics, "normal", sid, "random_rotation", repeat, seed + repeat * 101))
    original_order = [e for g in original for e in g.emitters]
    for repeat in range(1, int(task["nsga_repeats"]) + 1):
        plan, metrics0, elapsed, trace = optimize_nsga2_plan(
            list(range(len(network.emitters))),
            params,
            profile,
            seed + repeat * 1009,
            int(task["population"]),
            int(task["generations"]),
            initial_orders=[original_order],
            algorithm="nsga2",
        )
        metrics, _, _ = evaluate_plan(network, params, plan, profile)
        copy_cache_metrics(metrics, metrics0)
        metrics["total_reoptimization_time_s"] = elapsed
        metrics["mean_reoptimization_time_s"] = elapsed
        metrics["reoptimization_count"] = 1
        metrics["optimizer_algorithm"] = "nsga2"
        rows.append(add_identity(metrics, "normal", sid, "nsga2_verified", repeat, seed + repeat * 1009))
        for row in trace:
            row.update({"experiment": "normal", "scenario_id": sid, "policy": "nsga2_verified", "repeat": repeat})
            traces.append(row)
    return rows, traces


def run_fault_task(task: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    assert WORKER_NETWORK is not None and WORKER_PARAMS is not None
    network, params = WORKER_NETWORK, WORKER_PARAMS
    sid = int(task["scenario_id"])
    seed = int(task["seed"])
    experiment = str(task["experiment"])
    algorithms = [str(a).strip().lower() for a in task["dynamic_algorithms"] if str(a).strip()]
    profile = source_profile(seed, params)
    events = single_fault_scenario(seed + 333, network, params) if experiment == "single_fault" else multi_fault_scenario(seed + 777, network, params)
    original = original_plan(network, params)
    original_order = [e for g in original for e in g.emitters]
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for alg_idx, algorithm in enumerate(algorithms):
        base_plan, base_metrics, base_elapsed, trace = optimize_nsga2_plan(
            list(range(len(network.emitters))),
            params,
            profile,
            seed + 1009 + alg_idx * 100003,
            int(task["population"]),
            int(task["generations"]),
            initial_orders=[original_order],
            algorithm=algorithm,
        )
        for row in trace:
            row.update(
                {
                    "experiment": experiment,
                    "scenario_id": sid,
                    "policy": f"{algorithm}_initial",
                    "repeat": 1,
                    "optimizer_algorithm": algorithm,
                }
            )
            traces.append(row)
        for dynamic, suffix in [(False, "static_no_reoptimization"), (True, "dynamic_reoptimization")]:
            policy = f"{algorithm}_{suffix}"
            metrics, recs, tr = simulate_plan_with_events(
                network,
                params,
                base_plan,
                profile,
                events,
                dynamic,
                seed + (5000 if dynamic else 6000) + alg_idx * 100003,
                int(task["population"]),
                int(task["generations"]),
                optimizer_algorithm=algorithm,
            )
            metrics["initial_optimization_time_s"] = base_elapsed
            copy_cache_metrics(metrics, base_metrics, prefix="initial_")
            metrics["event_count"] = len(events)
            metrics["fault_count"] = sum(1 for e in events if e.kind == "fault")
            metrics["recovery_count"] = sum(1 for e in events if e.kind == "recovery")
            metrics["events"] = ";".join(f"{e.kind}:{e.node}@{e.time_h:.2f}" for e in events)
            metrics["optimizer_algorithm"] = algorithm
            rows.append(add_identity(metrics, experiment, sid, policy, 1, seed))
            for rec in recs:
                rec.update({"experiment": experiment, "scenario_id": sid, "policy": policy, "optimizer_algorithm": algorithm})
                records.append(rec)
            for row in tr:
                row.update({"experiment": experiment, "scenario_id": sid, "policy": policy, "repeat": 1, "optimizer_algorithm": algorithm})
                traces.append(row)
    return rows, records, traces


def init_worker(
    nodes: str,
    pipes: str,
    params_dict: dict[str, Any],
    roughness: dict[str, float],
    cache_enabled: bool,
    group_cache_max_items: int,
    plan_cache_max_items: int,
    pipe_loss_cache_max_items: int,
) -> None:
    global WORKER_NETWORK, WORKER_PARAMS, WORKER_ROUGHNESS
    global WORKER_CACHE_ENABLED, WORKER_GROUP_CACHE_MAX_ITEMS, WORKER_PLAN_CACHE_MAX_ITEMS, WORKER_PIPE_LOSS_CACHE_MAX_ITEMS
    WORKER_PARAMS = Params(**params_dict)
    WORKER_ROUGHNESS = roughness
    WORKER_NETWORK = load_network(Path(nodes), Path(pipes), roughness)
    WORKER_CACHE_ENABLED = bool(cache_enabled)
    WORKER_GROUP_CACHE_MAX_ITEMS = int(group_cache_max_items)
    WORKER_PLAN_CACHE_MAX_ITEMS = int(plan_cache_max_items)
    WORKER_PIPE_LOSS_CACHE_MAX_ITEMS = int(pipe_loss_cache_max_items)
    reset_worker_caches()


def worker_count(value: str) -> int:
    cpu = os.cpu_count() or 1
    if value.lower() == "auto":
        return max(1, math.floor(cpu * 0.80))
    return max(1, int(value))


def aggregate_by_policy(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "achievement_pct",
        "unmet_emitter_ratio_pct",
        "average_deficit_m3_per_emitter",
        "max_deficit_m3",
        "pressure_uniformity_index_pct",
        "feasible_group_ratio_pct",
        "affected_target_met_pct",
        "affected_mean_completion_pct",
        "total_reoptimization_time_s",
        "mean_reoptimization_time_s",
        "cache_group_hits",
        "cache_group_misses",
        "cache_group_hit_rate_pct",
        "cache_pipe_loss_hits",
        "cache_pipe_loss_misses",
        "cache_pipe_loss_hit_rate_pct",
        "cache_plan_hits",
        "cache_plan_misses",
        "cache_plan_hit_rate_pct",
        "initial_cache_group_hits",
        "initial_cache_group_misses",
        "initial_cache_group_hit_rate_pct",
        "initial_cache_pipe_loss_hits",
        "initial_cache_pipe_loss_misses",
        "initial_cache_pipe_loss_hit_rate_pct",
        "initial_cache_plan_hits",
        "initial_cache_plan_misses",
        "initial_cache_plan_hit_rate_pct",
    ]
    rows: list[dict[str, Any]] = []
    for (experiment, policy), group in df.groupby(["experiment", "policy"], dropna=False):
        row: dict[str, Any] = {"experiment": experiment, "policy": policy, "n": len(group)}
        for metric in metrics:
            if metric in group.columns:
                values = pd.to_numeric(group[metric], errors="coerce")
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0
                row[f"{metric}_p05"] = float(values.quantile(0.05))
                row[f"{metric}_p50"] = float(values.quantile(0.50))
                row[f"{metric}_p95"] = float(values.quantile(0.95))
        rows.append(row)
    return pd.DataFrame(rows)


AGRICULTURAL_METRICS = [
    "achievement_pct",
    "unmet_emitter_ratio_pct",
    "average_deficit_m3_per_emitter",
    "max_deficit_m3",
    "pressure_uniformity_index_pct",
    "feasible_group_ratio_pct",
    "affected_mean_completion_pct",
    "affected_target_met_pct",
    "total_reoptimization_time_s",
]

BENEFICIAL_HIGHER = {
    "achievement_pct",
    "pressure_uniformity_index_pct",
    "feasible_group_ratio_pct",
    "affected_mean_completion_pct",
    "affected_target_met_pct",
}


def bootstrap_mean_ci(values: pd.Series, rng: np.random.Generator, n_boot: int = 2000) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return math.nan, math.nan
    if len(clean) == 1:
        return float(clean[0]), float(clean[0])
    idx = rng.integers(0, len(clean), size=(n_boot, len(clean)))
    means = clean[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def agricultural_summary_with_ci(df: pd.DataFrame, seed: int = 20260709) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    group_cols = ["experiment", "policy"]
    if "optimizer_algorithm" in df.columns:
        group_cols.append("optimizer_algorithm")
    for keys, group in df.groupby(group_cols, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group_cols, key_values))
        for metric in AGRICULTURAL_METRICS:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            ci_low, ci_high = bootstrap_mean_ci(values, rng)
            worst = values.min() if metric in BENEFICIAL_HIGHER else values.max()
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "n": int(len(values)),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "mean_pm_std": f"{values.mean():.4f} +/- {(values.std(ddof=1) if len(values) > 1 else 0.0):.4f}",
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "p05": float(values.quantile(0.05)),
                    "p50": float(values.quantile(0.50)),
                    "p95": float(values.quantile(0.95)),
                    "best": float(values.max() if metric in BENEFICIAL_HIGHER else values.min()),
                    "worst": float(worst),
                }
            )
    return pd.DataFrame(rows)


def scenario_policy_means(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["experiment", "scenario_id", "policy"]
    if "optimizer_algorithm" in df.columns:
        cols.append("optimizer_algorithm")
    value_cols = [c for c in AGRICULTURAL_METRICS if c in df.columns]
    return df.groupby(cols, dropna=False, as_index=False)[value_cols].mean()


def safe_wilcoxon(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    a = pd.to_numeric(x, errors="coerce")
    b = pd.to_numeric(y, errors="coerce")
    mask = a.notna() & b.notna()
    a = a[mask].to_numpy(dtype=float)
    b = b[mask].to_numpy(dtype=float)
    if len(a) == 0:
        return math.nan, math.nan, 0
    diff = a - b
    if np.all(np.abs(diff) <= 1e-12):
        return 0.0, 1.0, int(len(a))
    stat, p_value = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return float(stat), float(p_value), int(len(a))


def paired_wilcoxon_tests(df: pd.DataFrame) -> pd.DataFrame:
    means = scenario_policy_means(df)
    rows: list[dict[str, Any]] = []
    normal = means[means["experiment"].eq("normal")].copy()
    for baseline in ["original_near_far", "random_rotation"]:
        left = normal[normal["policy"].eq("nsga2_verified")]
        right = normal[normal["policy"].eq(baseline)]
        merged = left.merge(right, on=["experiment", "scenario_id"], suffixes=("_left", "_right"))
        for metric in [m for m in AGRICULTURAL_METRICS if f"{m}_left" in merged.columns]:
            stat, p_value, n = safe_wilcoxon(merged[f"{metric}_left"], merged[f"{metric}_right"])
            rows.append(
                {
                    "experiment": "normal",
                    "comparison": f"nsga2_verified_vs_{baseline}",
                    "optimizer_algorithm": "nsga2",
                    "metric": metric,
                    "n_pairs": n,
                    "wilcoxon_statistic": stat,
                    "p_value": p_value,
                    "mean_difference_left_minus_right": float((merged[f"{metric}_left"] - merged[f"{metric}_right"]).mean()) if n else math.nan,
                }
            )

    for experiment in ["single_fault", "multi_fault"]:
        sub = means[means["experiment"].eq(experiment)].copy()
        for algorithm in sorted(a for a in sub["optimizer_algorithm"].dropna().unique() if a != "none"):
            dyn_policy = f"{algorithm}_dynamic_reoptimization"
            sta_policy = f"{algorithm}_static_no_reoptimization"
            left = sub[sub["policy"].eq(dyn_policy)]
            right = sub[sub["policy"].eq(sta_policy)]
            merged = left.merge(right, on=["experiment", "scenario_id", "optimizer_algorithm"], suffixes=("_left", "_right"))
            for metric in [m for m in AGRICULTURAL_METRICS if f"{m}_left" in merged.columns]:
                stat, p_value, n = safe_wilcoxon(merged[f"{metric}_left"], merged[f"{metric}_right"])
                rows.append(
                    {
                        "experiment": experiment,
                        "comparison": f"{dyn_policy}_vs_{sta_policy}",
                        "optimizer_algorithm": algorithm,
                        "metric": metric,
                        "n_pairs": n,
                        "wilcoxon_statistic": stat,
                        "p_value": p_value,
                        "mean_difference_left_minus_right": float((merged[f"{metric}_left"] - merged[f"{metric}_right"]).mean()) if n else math.nan,
                    }
                )
    return pd.DataFrame(rows)


def paired_algorithm_wilcoxon_tests(df: pd.DataFrame) -> pd.DataFrame:
    means = scenario_policy_means(df)
    rows: list[dict[str, Any]] = []
    for experiment in ["single_fault", "multi_fault"]:
        sub = means[
            means["experiment"].eq(experiment)
            & means["policy"].astype(str).str.endswith("_dynamic_reoptimization")
        ].copy()
        algorithms = sorted(a for a in sub["optimizer_algorithm"].dropna().unique() if a != "none")
        for i, left_alg in enumerate(algorithms):
            for right_alg in algorithms[i + 1 :]:
                left = sub[sub["optimizer_algorithm"].eq(left_alg)]
                right = sub[sub["optimizer_algorithm"].eq(right_alg)]
                merged = left.merge(right, on=["experiment", "scenario_id"], suffixes=("_left", "_right"))
                for metric in [m for m in AGRICULTURAL_METRICS if f"{m}_left" in merged.columns]:
                    stat, p_value, n = safe_wilcoxon(merged[f"{metric}_left"], merged[f"{metric}_right"])
                    rows.append(
                        {
                            "experiment": experiment,
                            "comparison": f"{left_alg}_dynamic_vs_{right_alg}_dynamic",
                            "left_algorithm": left_alg,
                            "right_algorithm": right_alg,
                            "metric": metric,
                            "n_pairs": n,
                            "wilcoxon_statistic": stat,
                            "p_value": p_value,
                            "mean_difference_left_minus_right": float((merged[f"{metric}_left"] - merged[f"{metric}_right"]).mean()) if n else math.nan,
                        }
                    )
    return pd.DataFrame(rows)


def create_distribution_figures(output_dir: Path, df: pd.DataFrame, single_gain: pd.DataFrame, multi_gain: pd.DataFrame) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_metrics = [
        "achievement_pct",
        "unmet_emitter_ratio_pct",
        "average_deficit_m3_per_emitter",
        "pressure_uniformity_index_pct",
        "feasible_group_ratio_pct",
        "affected_mean_completion_pct",
        "total_reoptimization_time_s",
    ]
    for experiment, exp_df in df.groupby("experiment"):
        for metric in plot_metrics:
            if metric not in exp_df.columns:
                continue
            groups = []
            labels = []
            for policy, group in exp_df.groupby("policy"):
                values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
                if len(values):
                    groups.append(values)
                    labels.append(str(policy))
            if not groups:
                continue
            plt.figure(figsize=(max(8, len(labels) * 0.85), 5))
            plt.boxplot(groups, tick_labels=labels, showmeans=True)
            plt.xticks(rotation=35, ha="right", fontsize=8)
            plt.ylabel(metric)
            plt.title(f"{experiment}: {metric}")
            plt.tight_layout()
            plt.savefig(fig_dir / f"boxplot_{experiment}_{metric}.png", dpi=220)
            plt.close()

    for experiment, gain in [("single_fault", single_gain), ("multi_fault", multi_gain)]:
        if gain.empty:
            continue
        for metric in [
            "achievement_gain_pct_points",
            "unmet_ratio_reduction_pct_points",
            "average_deficit_reduction_m3",
            "affected_completion_gain_pct_points",
            "dynamic_reoptimization_time_s",
        ]:
            groups = []
            labels = []
            for algorithm, group in gain.groupby("optimizer_algorithm"):
                values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
                if len(values):
                    groups.append(values)
                    labels.append(str(algorithm))
            if not groups:
                continue
            plt.figure(figsize=(8, 5))
            plt.boxplot(groups, tick_labels=labels, showmeans=True)
            plt.ylabel(metric)
            plt.title(f"{experiment}: dynamic gain {metric}")
            plt.tight_layout()
            plt.savefig(fig_dir / f"boxplot_{experiment}_dynamic_gain_{metric}.png", dpi=220)
            plt.close()


def paired_dynamic_value(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    sub = df[df["experiment"].eq(experiment)].copy()
    static = sub[sub["policy"].astype(str).str.endswith("_static_no_reoptimization")]
    dynamic = sub[sub["policy"].astype(str).str.endswith("_dynamic_reoptimization")]
    keys = ["scenario_id", "optimizer_algorithm"]
    merged = dynamic.merge(static, on=keys, suffixes=("_dynamic", "_static"))
    rows = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "experiment": experiment,
                "scenario_id": int(row["scenario_id"]),
                "optimizer_algorithm": row["optimizer_algorithm"],
                "achievement_gain_pct_points": row["achievement_pct_dynamic"] - row["achievement_pct_static"],
                "unmet_ratio_reduction_pct_points": row["unmet_emitter_ratio_pct_static"] - row["unmet_emitter_ratio_pct_dynamic"],
                "average_deficit_reduction_m3": row["average_deficit_m3_per_emitter_static"] - row["average_deficit_m3_per_emitter_dynamic"],
                "affected_completion_gain_pct_points": row["affected_mean_completion_pct_dynamic"] - row["affected_mean_completion_pct_static"],
                "dynamic_reoptimization_time_s": row["total_reoptimization_time_s_dynamic"],
            }
        )
    return pd.DataFrame(rows)


def management_comparison_tables(
    summary: pd.DataFrame,
    single_gain: pd.DataFrame,
    multi_gain: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def row_for(experiment: str, policy: str) -> pd.Series:
        rows = summary[summary["experiment"].eq(experiment) & summary["policy"].eq(policy)]
        if rows.empty:
            raise ValueError(f"Missing summary row for {experiment}/{policy}")
        return rows.iloc[0]

    nsga = row_for("normal", "nsga2_verified")
    original = row_for("normal", "original_near_far")
    random_policy = row_for("normal", "random_rotation")
    normal_rows = []
    for baseline_name, baseline in [("original_near_far", original), ("random_rotation", random_policy)]:
        normal_rows.append(
            {
                "comparison": f"nsga2_verified_vs_{baseline_name}",
                "achievement_gain_pct_points": nsga["achievement_pct_mean"] - baseline["achievement_pct_mean"],
                "unmet_ratio_reduction_pct_points": baseline["unmet_emitter_ratio_pct_mean"] - nsga["unmet_emitter_ratio_pct_mean"],
                "average_deficit_reduction_m3_per_emitter": baseline["average_deficit_m3_per_emitter_mean"] - nsga["average_deficit_m3_per_emitter_mean"],
                "max_deficit_reduction_m3": baseline["max_deficit_m3_mean"] - nsga["max_deficit_m3_mean"],
                "pressure_uniformity_gain_pct_points": nsga["pressure_uniformity_index_pct_mean"] - baseline["pressure_uniformity_index_pct_mean"],
                "feasible_group_ratio_gain_pct_points": nsga["feasible_group_ratio_pct_mean"] - baseline["feasible_group_ratio_pct_mean"],
                "mean_nsga2_optimization_time_s": nsga["total_reoptimization_time_s_mean"],
            }
        )
    normal_table = pd.DataFrame(normal_rows)

    def gain_summary(gain: pd.DataFrame, label: str) -> pd.DataFrame:
        if gain.empty:
            return pd.DataFrame()
        rows = []
        for algorithm, alg_group in gain.groupby("optimizer_algorithm", dropna=False):
            for metric in [
                "achievement_gain_pct_points",
                "unmet_ratio_reduction_pct_points",
                "average_deficit_reduction_m3",
                "affected_completion_gain_pct_points",
                "dynamic_reoptimization_time_s",
            ]:
                values = pd.to_numeric(alg_group[metric], errors="coerce")
                rows.append(
                    {
                        "experiment": label,
                        "optimizer_algorithm": algorithm,
                        "metric": metric,
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0,
                        "p05": float(values.quantile(0.05)),
                        "p50": float(values.quantile(0.50)),
                        "p95": float(values.quantile(0.95)),
                        "min": float(values.min()),
                        "max": float(values.max()),
                    }
                )
        return pd.DataFrame(rows)

    dynamic_table = pd.concat(
        [gain_summary(single_gain, "single_fault"), gain_summary(multi_gain, "multi_fault")],
        ignore_index=True,
    )

    management_rows = []
    management_rows.extend(
        {
            "experiment": "normal",
            "finding": row["comparison"],
            "main_management_metric": "feasible_group_ratio_gain_pct_points",
            "value": row["feasible_group_ratio_gain_pct_points"],
            "supporting_metric": "achievement_gain_pct_points",
            "supporting_value": row["achievement_gain_pct_points"],
        }
        for _, row in normal_table.iterrows()
    )
    for label, gain in [("single_fault", single_gain), ("multi_fault", multi_gain)]:
        if not gain.empty:
            for algorithm, alg_group in gain.groupby("optimizer_algorithm", dropna=False):
                management_rows.append(
                    {
                        "experiment": label,
                        "optimizer_algorithm": algorithm,
                        "finding": "dynamic_reoptimization_vs_static",
                        "main_management_metric": "achievement_gain_pct_points_mean",
                        "value": float(alg_group["achievement_gain_pct_points"].mean()),
                        "supporting_metric": "unmet_ratio_reduction_pct_points_mean",
                        "supporting_value": float(alg_group["unmet_ratio_reduction_pct_points"].mean()),
                    }
                )
                management_rows.append(
                    {
                        "experiment": label,
                        "optimizer_algorithm": algorithm,
                        "finding": "dynamic_reoptimization_recovery_capacity",
                        "main_management_metric": "affected_completion_gain_pct_points_mean",
                        "value": float(alg_group["affected_completion_gain_pct_points"].mean()),
                        "supporting_metric": "mean_reoptimization_time_s",
                        "supporting_value": float(alg_group["dynamic_reoptimization_time_s"].mean()),
                    }
                )
    management_table = pd.DataFrame(management_rows)
    return normal_table, dynamic_table, management_table


def write_report(
    output_dir: Path,
    all_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    single_gain: pd.DataFrame,
    multi_gain: pd.DataFrame,
    ci_summary: pd.DataFrame,
    wilcoxon_df: pd.DataFrame,
) -> None:
    def policy_table(experiment: str) -> str:
        cols = [
            "experiment",
            "policy",
            "n",
            "achievement_pct_mean",
            "unmet_emitter_ratio_pct_mean",
            "average_deficit_m3_per_emitter_mean",
            "max_deficit_m3_mean",
            "pressure_uniformity_index_pct_mean",
            "feasible_group_ratio_pct_mean",
            "affected_mean_completion_pct_mean",
            "total_reoptimization_time_s_mean",
        ]
        available = [c for c in cols if c in summary.columns]
        return summary[summary["experiment"].eq(experiment)][available].to_markdown(index=False)

    def gain_table(gain: pd.DataFrame) -> str:
        if gain.empty:
            return "No paired records."
        desc = gain[
            [
                "achievement_gain_pct_points",
                "unmet_ratio_reduction_pct_points",
                "average_deficit_reduction_m3",
                "affected_completion_gain_pct_points",
                "dynamic_reoptimization_time_s",
            ]
        ].agg(["mean", "std", "min", "median", "max"])
        return desc.to_markdown()

    lines = [
        "# Main Experiments After Hydraulic Verification",
        "",
        "## Model",
        "",
        "- Hydraulic calculation uses the verified roughness-based Darcy-Weisbach model.",
        "- Darcy friction factor is calculated from roughness and Reynolds number using Colebrook-White iteration.",
        "- Irrigation delivery uses fixed emitter demand with pressure reduction only for agricultural achievement accounting.",
        "- Metrics are reported in agricultural management language rather than only Pareto quality indicators.",
        "",
        "## Normal Operation",
        "",
        policy_table("normal"),
        "",
        "## Single Fault and Single Recovery",
        "",
        policy_table("single_fault"),
        "",
        "Dynamic reoptimization value:",
        "",
        gain_table(single_gain),
        "",
        "## Multiple Faults and Recoveries",
        "",
        policy_table("multi_fault"),
        "",
        "Dynamic reoptimization value:",
        "",
        gain_table(multi_gain),
        "",
        "## Key Output Files",
        "",
        "- `all_policy_metrics_after_verification.csv`",
        "- `policy_summary_after_verification.csv`",
        "- `normal_policy_metrics.csv`",
        "- `single_fault_policy_metrics.csv`",
        "- `multi_fault_policy_metrics.csv`",
        "- `single_fault_dynamic_value.csv`",
        "- `multi_fault_dynamic_value.csv`",
        "- `normal_management_comparison.csv`",
        "- `dynamic_reoptimization_management_summary.csv`",
        "- `agricultural_management_summary.csv`",
        "- `agricultural_metric_summary_with_ci.csv`",
        "- `paired_wilcoxon_tests.csv`",
        "- `paired_dynamic_algorithm_wilcoxon_tests.csv`",
        "- `figures/*.png`",
        "- `nsga2_convergence_after_verification.csv`",
    ]
    (output_dir / "main_experiment_report_after_verification.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    network_dir = root / "data" / "network"
    p = argparse.ArgumentParser(description="Main irrigation experiments after hydraulic verification.")
    p.add_argument("--nodes", type=Path, default=network_dir / "Nodes.xlsx")
    p.add_argument("--pipes", type=Path, default=network_dir / "Pipes.xlsx")
    p.add_argument("--output-dir", type=Path, default=root / "outputs" / "main_experiments")
    p.add_argument("--workers", type=str, default="auto")
    p.add_argument("--base-seed", type=int, default=20260709)
    p.add_argument("--normal-scenarios", type=int, default=100)
    p.add_argument("--single-fault-scenarios", type=int, default=100)
    p.add_argument("--multi-fault-scenarios", type=int, default=500)
    p.add_argument("--random-repeats", type=int, default=50)
    p.add_argument("--nsga-repeats", type=int, default=5)
    p.add_argument("--dynamic-algorithms", type=str, default="nsga2,spea2,rvea,moead")
    p.add_argument("--population", type=int, default=48)
    p.add_argument("--generations", type=int, default=50)
    p.add_argument("--roughness-upvc-mm", type=float, default=0.018)
    p.add_argument("--roughness-pe-mm", type=float, default=0.020)
    p.add_argument("--disable-cache", action="store_true", help="Disable group-level and full-plan evaluation caches.")
    p.add_argument("--group-cache-max-items", type=int, default=50000)
    p.add_argument("--plan-cache-max-items", type=int, default=20000)
    p.add_argument("--pipe-loss-cache-max-items", type=int, default=20000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    workers = worker_count(args.workers)
    params = Params()
    roughness = {"UPVC": args.roughness_upvc_mm, "PE": args.roughness_pe_mm}
    dynamic_algorithms = [a.strip().lower() for a in args.dynamic_algorithms.split(",") if a.strip()]
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "nodes": str(args.nodes),
        "pipes": str(args.pipes),
        "output_dir": str(args.output_dir),
        "params": params.__dict__,
        "roughness_mm": roughness,
        "workers": workers,
        "cpu_count": os.cpu_count(),
        "normal_scenarios": args.normal_scenarios,
        "single_fault_scenarios": args.single_fault_scenarios,
        "multi_fault_scenarios": args.multi_fault_scenarios,
        "random_repeats": args.random_repeats,
        "nsga_repeats": args.nsga_repeats,
        "dynamic_algorithms": dynamic_algorithms,
        "population": args.population,
        "generations": args.generations,
        "cache_enabled": not args.disable_cache,
        "group_cache_max_items": args.group_cache_max_items,
        "plan_cache_max_items": args.plan_cache_max_items,
        "pipe_loss_cache_max_items": args.pipe_loss_cache_max_items,
    }
    (args.output_dir / "experiment_metadata_after_verification.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    init_args = (
        str(args.nodes),
        str(args.pipes),
        params.__dict__,
        roughness,
        not args.disable_cache,
        args.group_cache_max_items,
        args.plan_cache_max_items,
        args.pipe_loss_cache_max_items,
    )
    all_rows: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    all_traces: list[dict[str, Any]] = []
    start = time.perf_counter()

    normal_tasks = [
        {
            "scenario_id": i,
            "seed": args.base_seed + i * 100,
            "random_repeats": args.random_repeats,
            "nsga_repeats": args.nsga_repeats,
            "population": args.population,
            "generations": args.generations,
        }
        for i in range(1, args.normal_scenarios + 1)
    ]
    fault_tasks = [
        {
            "experiment": "single_fault",
            "scenario_id": i,
            "seed": args.base_seed + 100000 + i * 100,
            "population": args.population,
            "generations": args.generations,
            "dynamic_algorithms": dynamic_algorithms,
        }
        for i in range(1, args.single_fault_scenarios + 1)
    ] + [
        {
            "experiment": "multi_fault",
            "scenario_id": i,
            "seed": args.base_seed + 200000 + i * 100,
            "population": args.population,
            "generations": args.generations,
            "dynamic_algorithms": dynamic_algorithms,
        }
        for i in range(1, args.multi_fault_scenarios + 1)
    ]

    print(f"Workers: {workers} of {os.cpu_count() or 1} logical CPUs", flush=True)
    print(f"Normal tasks: {len(normal_tasks)}, fault tasks: {len(fault_tasks)}", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=init_args) as executor:
        futures = [executor.submit(run_normal_task, task) for task in normal_tasks]
        futures.extend(executor.submit(run_fault_task, task) for task in fault_tasks)
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            if len(result) == 2:
                rows, traces = result
                records = []
            else:
                rows, records, traces = result
            all_rows.extend(rows)
            all_records.extend(records)
            all_traces.extend(traces)
            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == len(futures):
                elapsed = time.perf_counter() - start
                print(f"Completed {completed}/{len(futures)} tasks; elapsed={elapsed/60:.1f} min", flush=True)

    metrics_df = pd.DataFrame(all_rows)
    records_df = pd.DataFrame(all_records)
    traces_df = pd.DataFrame(all_traces)
    summary_df = aggregate_by_policy(metrics_df)
    single_gain = paired_dynamic_value(metrics_df, "single_fault")
    multi_gain = paired_dynamic_value(metrics_df, "multi_fault")
    normal_management, dynamic_management, agricultural_management = management_comparison_tables(
        summary_df, single_gain, multi_gain
    )
    ci_summary_df = agricultural_summary_with_ci(metrics_df, seed=args.base_seed + 909)
    wilcoxon_df = paired_wilcoxon_tests(metrics_df)
    algorithm_wilcoxon_df = paired_algorithm_wilcoxon_tests(metrics_df)
    create_distribution_figures(args.output_dir, metrics_df, single_gain, multi_gain)

    metrics_df.to_csv(args.output_dir / "all_policy_metrics_after_verification.csv", index=False, encoding="utf-8-sig")
    metrics_df[metrics_df["experiment"].eq("normal")].to_csv(args.output_dir / "normal_policy_metrics.csv", index=False, encoding="utf-8-sig")
    metrics_df[metrics_df["experiment"].eq("single_fault")].to_csv(args.output_dir / "single_fault_policy_metrics.csv", index=False, encoding="utf-8-sig")
    metrics_df[metrics_df["experiment"].eq("multi_fault")].to_csv(args.output_dir / "multi_fault_policy_metrics.csv", index=False, encoding="utf-8-sig")
    records_df.to_csv(args.output_dir / "fault_execution_records_after_verification.csv", index=False, encoding="utf-8-sig")
    traces_df.to_csv(args.output_dir / "nsga2_convergence_after_verification.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(args.output_dir / "policy_summary_after_verification.csv", index=False, encoding="utf-8-sig")
    single_gain.to_csv(args.output_dir / "single_fault_dynamic_value.csv", index=False, encoding="utf-8-sig")
    multi_gain.to_csv(args.output_dir / "multi_fault_dynamic_value.csv", index=False, encoding="utf-8-sig")
    normal_management.to_csv(args.output_dir / "normal_management_comparison.csv", index=False, encoding="utf-8-sig")
    dynamic_management.to_csv(args.output_dir / "dynamic_reoptimization_management_summary.csv", index=False, encoding="utf-8-sig")
    agricultural_management.to_csv(args.output_dir / "agricultural_management_summary.csv", index=False, encoding="utf-8-sig")
    ci_summary_df.to_csv(args.output_dir / "agricultural_metric_summary_with_ci.csv", index=False, encoding="utf-8-sig")
    wilcoxon_df.to_csv(args.output_dir / "paired_wilcoxon_tests.csv", index=False, encoding="utf-8-sig")
    algorithm_wilcoxon_df.to_csv(args.output_dir / "paired_dynamic_algorithm_wilcoxon_tests.csv", index=False, encoding="utf-8-sig")
    write_report(args.output_dir, metrics_df, summary_df, single_gain, multi_gain, ci_summary_df, wilcoxon_df)
    elapsed = time.perf_counter() - start
    metadata["total_wall_time_s"] = elapsed
    (args.output_dir / "experiment_metadata_after_verification.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Finished in {elapsed/60:.2f} min. Output: {args.output_dir}", flush=True)
    print(summary_df[["experiment", "policy", "n", "achievement_pct_mean", "unmet_emitter_ratio_pct_mean", "average_deficit_m3_per_emitter_mean", "feasible_group_ratio_pct_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
