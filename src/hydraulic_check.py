from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wntr

from network_topology import add_network_args, derive_tree, lateral_emitter_groups, network_paths, pipe_roughness_mm, read_network_tables


G = 9.80665
EPS = 1e-12
WATER_KINEMATIC_VISCOSITY_M2S = 1.0e-6


@dataclass(frozen=True)
class PipeNetworkData:
    nodes: pd.DataFrame
    pipes: pd.DataFrame
    node_z: dict[str, float]
    node_x: dict[str, float]
    node_y: dict[str, float]
    pipe_ids: list[str]
    from_nodes: list[str]
    to_nodes: list[str]
    lengths_m: np.ndarray
    diameters_m: np.ndarray
    areas_m2: np.ndarray
    materials: list[str]
    roughness_mm: np.ndarray
    parent_pipe: dict[str, int]
    irrigation_nodes: list[str]
    emitters: list[str]
    emitter_to_node: list[str]
    emitter_index: dict[str, int]
    node_to_path: dict[str, tuple[int, ...]]
    source_node: str
    laterals: tuple[tuple[str, ...], ...]
    node_lateral: dict[str, int]


@dataclass(frozen=True)
class GroupCase:
    case_id: str
    category: str
    source_head_m: float
    emitters: tuple[str, ...]
    notes: str
    origin: str


@dataclass(frozen=True)
class MyHydraulics:
    node_demands_m3s: dict[str, float]
    pipe_flows_m3s: np.ndarray
    pipe_velocities_ms: np.ndarray
    pipe_reynolds: np.ndarray
    pipe_darcy_factors: np.ndarray
    pipe_friction_loss_m: np.ndarray
    pipe_engineering_loss_1p1_m: np.ndarray
    node_pressure_friction_m: dict[str, float]
    node_pressure_engineering_1p1_m: dict[str, float]
    node_path_friction_loss_m: dict[str, float]
    node_path_engineering_loss_1p1_m: dict[str, float]


def node_key(node_id: str) -> tuple[int, str]:
    return (int(node_id[1:]), node_id) if node_id.startswith("J") and node_id[1:].isdigit() else (10**9, node_id)


def emitter_node(network: "PipeNetworkData", emitter: str) -> str:
    return network.emitter_to_node[network.emitter_index[emitter]]


def parse_emitter_list(value: Any) -> tuple[str, ...]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return tuple()
    return tuple(part.strip() for part in str(value).split(";") if part.strip())


def load_network(
    nodes_path: Path,
    pipes_path: Path | None,
    roughness_mm_by_material: dict[str, float],
    emitters_path: Path | None = None,
) -> PipeNetworkData:
    tables = read_network_tables(nodes_path, pipes_path, emitters_path)
    tree = derive_tree(tables)
    nodes, pipes = tables.nodes, tables.pipes
    diameters = pipes["Diameter_m"].astype(float).to_numpy()
    return PipeNetworkData(
        nodes=nodes,
        pipes=pipes,
        node_z=nodes.set_index("node_id")["Z"].astype(float).to_dict(),
        node_x=nodes.set_index("node_id")["X"].astype(float).to_dict(),
        node_y=nodes.set_index("node_id")["Y"].astype(float).to_dict(),
        pipe_ids=pipes["Pipe ID"].tolist(),
        from_nodes=pipes["FromNode"].tolist(),
        to_nodes=pipes["ToNode"].tolist(),
        lengths_m=pipes["Length_m"].astype(float).to_numpy(),
        diameters_m=diameters,
        areas_m2=math.pi * np.square(diameters) / 4.0,
        materials=pipes["Material"].tolist(),
        roughness_mm=np.array(pipe_roughness_mm(tables, roughness_mm_by_material), dtype=float),
        parent_pipe=tree.parent_pipe,
        irrigation_nodes=tree.irrigation_nodes,
        emitters=tree.emitters,
        emitter_to_node=tree.emitter_to_node,
        emitter_index=tree.emitter_index,
        node_to_path=tree.node_to_path,
        source_node=tree.source_node,
        laterals=tree.laterals,
        node_lateral=tree.node_lateral,
    )


def build_wntr_model(
    network: PipeNetworkData,
    node_demands_m3s: dict[str, float] | None,
    source_head_m: float,
) -> wntr.network.WaterNetworkModel:
    node_demands_m3s = node_demands_m3s or {}
    wn = wntr.network.WaterNetworkModel()
    wn.options.hydraulic.headloss = "D-W"
    wn.options.hydraulic.demand_model = "DDA"
    # EPANET's VISCOSITY is relative to 1.1e-5 ft2/s (1.022e-6 m2/s); match the model's 1.0e-6 m2/s.
    wn.options.hydraulic.viscosity = WATER_KINEMATIC_VISCOSITY_M2S / (1.1e-5 * 0.3048**2)
    wn.options.time.duration = 0
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.report_timestep = 3600
    try:
        wn.options.hydraulic.inpfile_units = "LPS"
    except Exception:
        pass

    source = network.source_node
    wn.add_reservoir(source, base_head=float(source_head_m))
    try:
        wn.get_node(source).coordinates = (float(network.node_x[source]), float(network.node_y[source]))
    except Exception:
        pass

    for node_id in sorted((set(network.nodes["node_id"]) - {source}), key=node_key):
        wn.add_junction(
            node_id,
            base_demand=float(node_demands_m3s.get(node_id, 0.0)),
            elevation=float(network.node_z[node_id]),
        )
        try:
            wn.get_node(node_id).coordinates = (float(network.node_x[node_id]), float(network.node_y[node_id]))
        except Exception:
            pass

    for i, pipe_id in enumerate(network.pipe_ids):
        wn.add_pipe(
            pipe_id,
            network.from_nodes[i],
            network.to_nodes[i],
            length=float(network.lengths_m[i]),
            diameter=float(network.diameters_m[i]),
            roughness=float(network.roughness_mm[i] / 1000.0),
            minor_loss=0.0,
            initial_status="OPEN",
        )
    return wn


def darcy_friction_factor(reynolds: float, roughness_m: float, diameter_m: float) -> float:
    """EPANET 2.2 all-regime Darcy factor, identical to the workbench's ``friction.darcy_friction_factor``.

    64/Re up to Re = 2000, Swamee-Jain from Re = 4000, Dunlop's cubic anchored at Re = 4000 in between.
    """
    if reynolds <= 0.0:
        return 0.0
    if reynolds <= 2000.0:
        return 64.0 / reynolds
    rr = max(roughness_m / diameter_m, 0.0)
    if reynolds >= 4000.0:
        return 0.25 / math.log10(rr / 3.7 + 5.74 / reynolds**0.9) ** 2
    y2 = rr / 3.7 + 0.0032889547634539906  # 5.74 / 4000**0.9
    y3 = -2.0 * math.log10(y2)
    fa = y3**-2
    fb = fa * (2.0 - 0.005142149657990938 / (y2 * y3))
    ratio = reynolds / 2000.0
    x1 = 7.0 * fa - fb
    x2 = 0.128 - 17.0 * fa + 2.5 * fb
    x3 = -0.128 + 13.0 * fa - 2.0 * fb
    x4 = ratio * (0.032 - 3.0 * fa + 0.5 * fb)
    return x1 + ratio * (x2 + ratio * (x3 + x4))

def darcy_factors_from_roughness(
    flows_m3s: np.ndarray,
    velocities_ms: np.ndarray,
    diameters_m: np.ndarray,
    roughness_mm: np.ndarray,
    kinematic_viscosity_m2s: float,
) -> tuple[np.ndarray, np.ndarray]:
    reynolds = np.zeros_like(flows_m3s, dtype=float)
    factors = np.zeros_like(flows_m3s, dtype=float)
    active = flows_m3s > EPS
    reynolds[active] = np.abs(velocities_ms[active]) * diameters_m[active] / max(kinematic_viscosity_m2s, EPS)
    roughness_m = roughness_mm / 1000.0
    for idx in np.flatnonzero(active):
        factors[idx] = darcy_friction_factor(
            float(reynolds[idx]),
            float(roughness_m[idx]),
            float(diameters_m[idx]),
        )
    return reynolds, factors


def compute_my_hydraulics(
    network: PipeNetworkData,
    emitters: tuple[str, ...],
    source_head_m: float,
    emitter_demand_m3s: float,
    local_loss_ratio: float,
    kinematic_viscosity_m2s: float = WATER_KINEMATIC_VISCOSITY_M2S,
) -> MyHydraulics:
    counts = Counter(emitter_node(network, e) for e in emitters)
    node_demands = {node: count * emitter_demand_m3s for node, count in counts.items()}
    pipe_flows = np.zeros(len(network.pipe_ids), dtype=float)
    for node_id, demand in node_demands.items():
        for pipe_idx in network.node_to_path[node_id]:
            pipe_flows[pipe_idx] += demand

    active = pipe_flows > EPS
    velocities = np.zeros_like(pipe_flows)
    velocities[active] = pipe_flows[active] / network.areas_m2[active]
    reynolds, darcy_factors = darcy_factors_from_roughness(
        pipe_flows,
        velocities,
        network.diameters_m,
        network.roughness_mm,
        kinematic_viscosity_m2s,
    )
    friction_loss = np.zeros_like(pipe_flows)
    friction_loss[active] = (
        darcy_factors[active]
        * (network.lengths_m[active] / network.diameters_m[active])
        * np.square(velocities[active])
        / (2.0 * G)
    )
    engineering_loss = friction_loss * (1.0 + local_loss_ratio)

    pressure_friction: dict[str, float] = {}
    pressure_engineering: dict[str, float] = {}
    path_friction: dict[str, float] = {}
    path_engineering: dict[str, float] = {}
    for node_id in sorted(node_demands, key=node_key):
        path = list(network.node_to_path[node_id])
        hf = float(friction_loss[path].sum())
        he = float(engineering_loss[path].sum())
        path_friction[node_id] = hf
        path_engineering[node_id] = he
        pressure_friction[node_id] = float(source_head_m - hf - network.node_z[node_id])
        pressure_engineering[node_id] = float(source_head_m - he - network.node_z[node_id])

    return MyHydraulics(
        node_demands_m3s=node_demands,
        pipe_flows_m3s=pipe_flows,
        pipe_velocities_ms=velocities,
        pipe_reynolds=reynolds,
        pipe_darcy_factors=darcy_factors,
        pipe_friction_loss_m=friction_loss,
        pipe_engineering_loss_1p1_m=engineering_loss,
        node_pressure_friction_m=pressure_friction,
        node_pressure_engineering_1p1_m=pressure_engineering,
        node_path_friction_loss_m=path_friction,
        node_path_engineering_loss_1p1_m=path_engineering,
    )


def original_groups(network: PipeNetworkData, group_size: int = 4) -> list[tuple[str, ...]]:
    return [
        tuple(network.emitters[i] for i in group)
        for group in lateral_emitter_groups(network.laterals, network.emitter_to_node, group_size)
    ]


def read_normal_groups(run_data_dir: Path) -> list[tuple[str, ...]]:
    path = run_data_dir / "normal_nsga3_groups.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return [parse_emitter_list(v) for v in df["emitters"].tolist()]


def choose_lowest_pressure_group(
    groups: list[tuple[str, ...]],
    network: PipeNetworkData,
    source_head_m: float,
    emitter_demand_m3s: float,
    local_loss_ratio: float,
    kinematic_viscosity_m2s: float = WATER_KINEMATIC_VISCOSITY_M2S,
) -> tuple[tuple[str, ...], float]:
    best_group: tuple[str, ...] | None = None
    best_min_pressure = math.inf
    for emitters in groups:
        calc = compute_my_hydraulics(
            network,
            emitters,
            source_head_m,
            emitter_demand_m3s,
            local_loss_ratio,
            kinematic_viscosity_m2s,
        )
        if not calc.node_pressure_friction_m:
            continue
        min_pressure = min(calc.node_pressure_friction_m.values())
        if min_pressure < best_min_pressure:
            best_group = emitters
            best_min_pressure = min_pressure
    if best_group is None:
        raise ValueError("No valid groups were available for pressure screening.")
    return best_group, float(best_min_pressure)


def choose_cross_branch_group(network: PipeNetworkData, groups: list[tuple[str, ...]]) -> tuple[str, ...] | None:
    best: tuple[str, ...] | None = None
    best_score: tuple[int, int, int] = (-1, -1, -1)
    for group in groups:
        nodes = [emitter_node(network, e) for e in group]
        branches = {network.node_lateral[node] for node in nodes}
        unique_nodes = set(nodes)
        score = (len(branches), len(unique_nodes), len(group))
        if score > best_score:
            best_score = score
            best = group
    return best


def choose_original_violation_from_history(run_data_dir: Path) -> tuple[tuple[str, ...], float, str] | None:
    path = run_data_dir / "executed_schedule_records.csv"
    if not path.exists():
        return None
    usecols = [
        "policy",
        "stage",
        "source_head_m",
        "emitters",
        "active_emitters",
        "min_working_head_m",
        "scenario_id",
        "start_h",
    ]
    df = pd.read_csv(path, usecols=lambda c: c in usecols)
    df = df[
        df["policy"].eq("original_fixed")
        & df["stage"].astype(str).str.contains("no_fault", na=False)
        & df["min_working_head_m"].astype(float).lt(11.4)
    ].copy()
    if df.empty:
        return None
    row = df.sort_values("min_working_head_m", ascending=True).iloc[0]
    emitters = parse_emitter_list(row.get("active_emitters")) or parse_emitter_list(row.get("emitters"))
    notes = f"from scenario {int(row['scenario_id'])}, start_h={float(row['start_h']):.3f}"
    return emitters, float(row["source_head_m"]), notes


def choose_fault_reoptimized_from_history(run_data_dir: Path) -> tuple[tuple[str, ...], float, str] | None:
    path = run_data_dir / "executed_schedule_records.csv"
    if not path.exists():
        return None
    usecols = [
        "policy",
        "config",
        "repeat",
        "stage",
        "source_head_m",
        "emitters",
        "active_emitters",
        "n_active",
        "min_working_head_m",
        "scenario_id",
        "start_h",
        "cv",
    ]
    df = pd.read_csv(path, usecols=lambda c: c in usecols)
    df = df[
        df["policy"].eq("nsga3_fault_recovery")
        & df["config"].eq("std_p72_g90")
        & df["stage"].astype(str).str.contains("active_fault", na=False)
        & df["n_active"].astype(float).eq(4)
    ].copy()
    if df.empty:
        return None
    df["sort_cv"] = df["cv"].astype(float)
    df["sort_pressure"] = df["min_working_head_m"].astype(float)
    row = df.sort_values(["sort_cv", "sort_pressure"], ascending=[True, True]).iloc[0]
    emitters = parse_emitter_list(row.get("active_emitters")) or parse_emitter_list(row.get("emitters"))
    notes = (
        f"from scenario {int(row['scenario_id'])}, repeat={int(row['repeat'])}, "
        f"start_h={float(row['start_h']):.3f}"
    )
    return emitters, float(row["source_head_m"]), notes


def read_predefined_typical_groups(
    network: PipeNetworkData,
    run_data_dir: Path,
) -> list[GroupCase]:
    path = run_data_dir / "typical_groups.csv"
    if not path.exists():
        return []

    table = pd.read_csv(path)
    required = {"case_id", "category", "source_head_m", "emitters"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    cases: list[GroupCase] = []
    for row in table.to_dict(orient="records"):
        emitters = parse_emitter_list(row["emitters"])
        unknown = sorted(set(emitters).difference(network.emitter_index))
        if unknown:
            raise ValueError(f"Unknown emitters in {path}: {unknown}")
        origin = "published_typical_groups" if pd.isna(row.get("origin")) else str(row["origin"])
        notes = "" if pd.isna(row.get("notes")) else str(row["notes"])
        cases.append(
            GroupCase(
                case_id=str(row["case_id"]),
                category=str(row["category"]),
                source_head_m=float(row["source_head_m"]),
                emitters=emitters,
                notes=notes,
                origin=origin,
            )
        )
    return cases


def select_typical_groups(
    network: PipeNetworkData,
    run_data_dir: Path,
    source_head_m: float,
    emitter_demand_m3s: float,
    local_loss_ratio: float,
    kinematic_viscosity_m2s: float = WATER_KINEMATIC_VISCOSITY_M2S,
) -> list[GroupCase]:
    predefined = read_predefined_typical_groups(network, run_data_dir)
    if predefined:
        return predefined

    cases: list[GroupCase] = []
    originals = original_groups(network)
    normal_groups = read_normal_groups(run_data_dir)

    cases.append(
        GroupCase(
            "C01_near_original",
            "near-end original rotation group",
            source_head_m,
            originals[0],
            f"Original first group near {network.source_node}.",
            "constructed_original",
        )
    )
    cases.append(
        GroupCase(
            "C02_far_original",
            "far-end original rotation group",
            source_head_m,
            originals[-1],
            "Original last group at the far end of branch 10.",
            "constructed_original",
        )
    )

    if normal_groups:
        cross = choose_cross_branch_group(network, normal_groups)
        if cross:
            cases.append(
                GroupCase(
                    "C03_cross_branch_optimized",
                    "cross-branch optimized group",
                    source_head_m,
                    cross,
                    "Selected from normal_nsga3_groups.csv by maximum branch coverage.",
                    "normal_nsga3_groups",
                )
            )
        low_group, low_pressure = choose_lowest_pressure_group(
            normal_groups,
            network,
            source_head_m,
            emitter_demand_m3s,
            local_loss_ratio,
            kinematic_viscosity_m2s,
        )
        cases.append(
            GroupCase(
                "C04_low_pressure_optimized",
                "lowest-pressure optimized group",
                source_head_m,
                low_group,
                f"Lowest roughness-based pressure among normal optimized groups: {low_pressure:.3f} m.",
                "normal_nsga3_groups",
            )
        )

    violation = choose_original_violation_from_history(run_data_dir)
    if violation:
        emitters, head, notes = violation
        cases.append(
            GroupCase(
                "C05_original_pressure_violation",
                "original group with minimum-pressure violation",
                head,
                emitters,
                notes,
                "executed_schedule_records",
            )
        )
    else:
        low_group, low_pressure = choose_lowest_pressure_group(
            originals,
            network,
            20.0,
            emitter_demand_m3s,
            local_loss_ratio,
            kinematic_viscosity_m2s,
        )
        cases.append(
            GroupCase(
                "C05_original_pressure_violation",
                "original group with minimum-pressure violation",
                20.0,
                low_group,
                f"Fallback: lowest original group at source_head=20 m; pressure={low_pressure:.3f} m.",
                "constructed_original",
            )
        )

    fault_case = choose_fault_reoptimized_from_history(run_data_dir)
    if fault_case:
        emitters, head, notes = fault_case
        cases.append(
            GroupCase(
                "C06_fault_reoptimized",
                "post-fault reoptimized group",
                head,
                emitters,
                notes,
                "executed_schedule_records",
            )
        )

    return cases


def error_metrics(
    case_id: str,
    metric_name: str,
    model_values: np.ndarray,
    epanet_values: np.ndarray,
) -> dict[str, Any]:
    model_values = np.asarray(model_values, dtype=float)
    epanet_values = np.asarray(epanet_values, dtype=float)
    diff = model_values - epanet_values
    abs_diff = np.abs(diff)
    denom_mask = np.abs(epanet_values) > EPS
    mape = float(np.mean(abs_diff[denom_mask] / np.abs(epanet_values[denom_mask])) * 100.0) if denom_mask.any() else math.nan
    return {
        "case_id": case_id,
        "metric": metric_name,
        "n": int(len(model_values)),
        "mae": float(np.mean(abs_diff)) if len(abs_diff) else math.nan,
        "rmse": float(np.sqrt(np.mean(np.square(diff)))) if len(diff) else math.nan,
        "mape_pct": mape,
        "max_abs_error": float(np.max(abs_diff)) if len(abs_diff) else math.nan,
        "mean_model": float(np.mean(model_values)) if len(model_values) else math.nan,
        "mean_epanet": float(np.mean(epanet_values)) if len(epanet_values) else math.nan,
    }


def run_epanet_group(
    network: PipeNetworkData,
    case: GroupCase,
    my_calc: MyHydraulics,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    wn = build_wntr_model(network, my_calc.node_demands_m3s, case.source_head_m)
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    node_pressure = results.node["pressure"].iloc[0]
    node_head = results.node["head"].iloc[0]
    link_flow = results.link["flowrate"].iloc[0]
    return node_pressure, node_head, link_flow


def export_inp(path: Path, network: PipeNetworkData, node_demands_m3s: dict[str, float] | None, source_head_m: float) -> None:
    wn = build_wntr_model(network, node_demands_m3s, source_head_m)
    wntr.network.write_inpfile(wn, str(path), units="LPS", version=2.2, force_coordinates=True)


def evaluate_case(
    network: PipeNetworkData,
    case: GroupCase,
    emitter_demand_m3s: float,
    local_loss_ratio: float,
    inp_dir: Path,
    kinematic_viscosity_m2s: float = WATER_KINEMATIC_VISCOSITY_M2S,
    export_case_inp: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    for emitter in case.emitters:
        if emitter not in network.emitter_index:
            raise ValueError(f"Unknown emitter in {case.case_id}: {emitter}")

    my_calc = compute_my_hydraulics(
        network,
        case.emitters,
        case.source_head_m,
        emitter_demand_m3s,
        local_loss_ratio,
        kinematic_viscosity_m2s,
    )
    if export_case_inp:
        export_inp(inp_dir / f"{case.case_id}.inp", network, my_calc.node_demands_m3s, case.source_head_m)
    epanet_pressure, epanet_head, epanet_flow = run_epanet_group(network, case, my_calc)

    active_nodes = sorted(my_calc.node_demands_m3s, key=node_key)
    active_pipe_idx = np.flatnonzero(my_calc.pipe_flows_m3s > EPS)

    node_rows: list[dict[str, Any]] = []
    for node_id in active_nodes:
        my_pressure = my_calc.node_pressure_friction_m[node_id]
        ep_pressure = float(epanet_pressure[node_id])
        node_rows.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "node_id": node_id,
                "demand_m3s": my_calc.node_demands_m3s[node_id],
                "source_head_m": case.source_head_m,
                "elevation_m": network.node_z[node_id],
                "my_pressure_friction_only_m": my_pressure,
                "my_pressure_with_1p1_engineering_m": my_calc.node_pressure_engineering_1p1_m[node_id],
                "epanet_pressure_m": ep_pressure,
                "pressure_error_model_minus_epanet_m": my_pressure - ep_pressure,
                "my_path_friction_loss_m": my_calc.node_path_friction_loss_m[node_id],
                "my_path_with_1p1_engineering_loss_m": my_calc.node_path_engineering_loss_1p1_m[node_id],
                "epanet_path_headloss_m": case.source_head_m - float(epanet_head[node_id]),
            }
        )

    pipe_rows: list[dict[str, Any]] = []
    for i in active_pipe_idx:
        pipe_id = network.pipe_ids[int(i)]
        ep_flow = float(epanet_flow[pipe_id])
        ep_velocity = abs(ep_flow) / float(network.areas_m2[i])
        ep_headloss = float(epanet_head[network.from_nodes[i]] - epanet_head[network.to_nodes[i]])
        pipe_rows.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "pipe_id": pipe_id,
                "from_node": network.from_nodes[i],
                "to_node": network.to_nodes[i],
                "material": network.materials[i],
                "length_m": network.lengths_m[i],
                "diameter_m": network.diameters_m[i],
                "epanet_roughness_mm": network.roughness_mm[i],
                "model_reynolds": my_calc.pipe_reynolds[i],
                "model_darcy_f": my_calc.pipe_darcy_factors[i],
                "my_flow_m3s": my_calc.pipe_flows_m3s[i],
                "epanet_flow_m3s": ep_flow,
                "flow_error_model_minus_epanet_m3s": my_calc.pipe_flows_m3s[i] - ep_flow,
                "my_velocity_ms": my_calc.pipe_velocities_ms[i],
                "epanet_velocity_ms": ep_velocity,
                "velocity_error_model_minus_epanet_ms": my_calc.pipe_velocities_ms[i] - ep_velocity,
                "my_friction_headloss_m": my_calc.pipe_friction_loss_m[i],
                "my_headloss_with_1p1_engineering_m": my_calc.pipe_engineering_loss_1p1_m[i],
                "epanet_pipe_headloss_m": ep_headloss,
                "headloss_error_model_minus_epanet_m": my_calc.pipe_friction_loss_m[i] - ep_headloss,
            }
        )

    node_df = pd.DataFrame(node_rows)
    pipe_df = pd.DataFrame(pipe_rows)
    summary = {
        "case_id": case.case_id,
        "category": case.category,
        "origin": case.origin,
        "notes": case.notes,
        "source_head_m": case.source_head_m,
        "active_emitters": ";".join(case.emitters),
        "active_nodes": ";".join(active_nodes),
        "n_emitters": len(case.emitters),
        "n_active_nodes": len(active_nodes),
        "total_demand_m3s": len(case.emitters) * emitter_demand_m3s,
        "min_my_pressure_friction_only_m": node_df["my_pressure_friction_only_m"].min(),
        "min_my_pressure_with_1p1_engineering_m": node_df["my_pressure_with_1p1_engineering_m"].min(),
        "min_epanet_pressure_m": node_df["epanet_pressure_m"].min(),
        "sum_my_active_pipe_friction_headloss_m": pipe_df["my_friction_headloss_m"].sum(),
        "sum_my_active_pipe_headloss_with_1p1_engineering_m": pipe_df["my_headloss_with_1p1_engineering_m"].sum(),
        "sum_epanet_active_pipe_headloss_m": pipe_df["epanet_pipe_headloss_m"].sum(),
        "max_my_path_friction_headloss_m": node_df["my_path_friction_loss_m"].max(),
        "max_my_path_headloss_with_1p1_engineering_m": node_df["my_path_with_1p1_engineering_loss_m"].max(),
        "max_epanet_path_headloss_m": node_df["epanet_path_headloss_m"].max(),
        "max_my_velocity_ms": pipe_df["my_velocity_ms"].max(),
        "max_epanet_velocity_ms": pipe_df["epanet_velocity_ms"].max(),
    }

    metrics_rows = [
        error_metrics(
            case.case_id,
            "active_node_pressure_m",
            node_df["my_pressure_friction_only_m"].to_numpy(),
            node_df["epanet_pressure_m"].to_numpy(),
        ),
        error_metrics(
            case.case_id,
            "pipe_flow_m3s",
            pipe_df["my_flow_m3s"].to_numpy(),
            pipe_df["epanet_flow_m3s"].to_numpy(),
        ),
        error_metrics(
            case.case_id,
            "pipe_velocity_ms",
            pipe_df["my_velocity_ms"].to_numpy(),
            pipe_df["epanet_velocity_ms"].to_numpy(),
        ),
        error_metrics(
            case.case_id,
            "pipe_friction_headloss_m",
            pipe_df["my_friction_headloss_m"].to_numpy(),
            pipe_df["epanet_pipe_headloss_m"].to_numpy(),
        ),
        error_metrics(
            case.case_id,
            "sum_active_pipe_headloss_m",
            np.array([summary["sum_my_active_pipe_friction_headloss_m"]]),
            np.array([summary["sum_epanet_active_pipe_headloss_m"]]),
        ),
        error_metrics(
            case.case_id,
            "max_path_headloss_m",
            np.array([summary["max_my_path_friction_headloss_m"]]),
            np.array([summary["max_epanet_path_headloss_m"]]),
        ),
    ]
    return node_rows, pipe_rows, [summary], metrics_rows


def select_random_validation_cases(
    network: PipeNetworkData,
    count: int,
    seed: int,
    source_head_min_m: float,
    source_head_max_m: float,
    emitters_per_case: int,
) -> list[GroupCase]:
    rng = np.random.default_rng(seed)
    emitters = np.array(network.emitters, dtype=object)
    cases: list[GroupCase] = []
    for idx in range(1, count + 1):
        chosen = tuple(str(x) for x in rng.choice(emitters, size=emitters_per_case, replace=False).tolist())
        head = float(rng.uniform(source_head_min_m, source_head_max_m))
        cases.append(
            GroupCase(
                case_id=f"R{idx:03d}",
                category="random validation group",
                source_head_m=head,
                emitters=chosen,
                notes=f"random_seed={seed}, emitters_per_case={emitters_per_case}",
                origin="random_validation",
            )
        )
    return cases


def error_observation_rows(
    node_df: pd.DataFrame,
    pipe_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add_row(case_id: str, metric: str, item_id: str, model: float, epanet: float) -> None:
        err = float(model - epanet)
        abs_err = abs(err)
        pct_err = abs_err / abs(epanet) * 100.0 if abs(epanet) > EPS else math.nan
        rows.append(
            {
                "case_id": case_id,
                "metric": metric,
                "item_id": item_id,
                "model_value": float(model),
                "epanet_value": float(epanet),
                "signed_error": err,
                "abs_error": abs_err,
                "pct_error": pct_err,
            }
        )

    for row in node_df.itertuples(index=False):
        add_row(
            str(row.case_id),
            "active_node_pressure_m",
            str(row.node_id),
            float(row.my_pressure_friction_only_m),
            float(row.epanet_pressure_m),
        )
    for row in pipe_df.itertuples(index=False):
        add_row(str(row.case_id), "pipe_flow_m3s", str(row.pipe_id), float(row.my_flow_m3s), float(row.epanet_flow_m3s))
        add_row(
            str(row.case_id),
            "pipe_velocity_ms",
            str(row.pipe_id),
            float(row.my_velocity_ms),
            float(row.epanet_velocity_ms),
        )
        add_row(
            str(row.case_id),
            "pipe_friction_headloss_m",
            str(row.pipe_id),
            float(row.my_friction_headloss_m),
            float(row.epanet_pipe_headloss_m),
        )
    for row in summary_df.itertuples(index=False):
        add_row(
            str(row.case_id),
            "sum_active_pipe_headloss_m",
            "active_pipe_sum",
            float(row.sum_my_active_pipe_friction_headloss_m),
            float(row.sum_epanet_active_pipe_headloss_m),
        )
        add_row(
            str(row.case_id),
            "max_path_headloss_m",
            "max_active_node_path",
            float(row.max_my_path_friction_headloss_m),
            float(row.max_epanet_path_headloss_m),
        )
    return rows


def distribution_summary(values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {
            "mean": math.nan,
            "std": math.nan,
            "min": math.nan,
            "p05": math.nan,
            "p25": math.nan,
            "p50": math.nan,
            "p75": math.nan,
            "p95": math.nan,
            "p99": math.nan,
            "max": math.nan,
        }
    return {
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)) if len(clean) > 1 else 0.0,
        "min": float(clean.min()),
        "p05": float(clean.quantile(0.05)),
        "p25": float(clean.quantile(0.25)),
        "p50": float(clean.quantile(0.50)),
        "p75": float(clean.quantile(0.75)),
        "p95": float(clean.quantile(0.95)),
        "p99": float(clean.quantile(0.99)),
        "max": float(clean.max()),
    }


def overall_metrics_from_observations(obs_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric, group in obs_df.groupby("metric", sort=True):
        signed = group["signed_error"].astype(float).to_numpy()
        abs_error = np.abs(signed)
        pct = pd.to_numeric(group["pct_error"], errors="coerce").dropna()
        rows.append(
            {
                "metric": metric,
                "n": int(len(group)),
                "mae": float(abs_error.mean()) if len(abs_error) else math.nan,
                "rmse": float(np.sqrt(np.mean(np.square(signed)))) if len(signed) else math.nan,
                "mape_pct": float(pct.mean()) if not pct.empty else math.nan,
                "max_abs_error": float(abs_error.max()) if len(abs_error) else math.nan,
                "bias_mean_signed_error": float(signed.mean()) if len(signed) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def observation_error_distribution(obs_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric, group in obs_df.groupby("metric", sort=True):
        abs_stats = distribution_summary(group["abs_error"])
        pct_stats = distribution_summary(group["pct_error"])
        row: dict[str, Any] = {"metric": metric, "n": int(len(group))}
        row.update({f"abs_error_{k}": v for k, v in abs_stats.items()})
        row.update({f"pct_error_{k}": v for k, v in pct_stats.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def case_metric_distribution(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    value_columns = ["mae", "rmse", "mape_pct", "max_abs_error"]
    for metric, group in metrics_df.groupby("metric", sort=True):
        row: dict[str, Any] = {"metric": metric, "n_cases": int(group["case_id"].nunique())}
        for col in value_columns:
            stats = distribution_summary(group[col])
            row.update({f"{col}_{k}": v for k, v in stats.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def write_random_validation_report(
    output_dir: Path,
    random_cases: list[GroupCase],
    overall_metrics: pd.DataFrame,
    obs_distribution: pd.DataFrame,
    case_distribution: pd.DataFrame,
) -> None:
    key_metrics = [
        "active_node_pressure_m",
        "pipe_friction_headloss_m",
        "sum_active_pipe_headloss_m",
        "max_path_headloss_m",
        "pipe_flow_m3s",
        "pipe_velocity_ms",
    ]
    overall_view = overall_metrics[overall_metrics["metric"].isin(key_metrics)].copy()
    distribution_view = obs_distribution[obs_distribution["metric"].isin(key_metrics)].copy()
    case_view = case_distribution[case_distribution["metric"].isin(key_metrics)].copy()
    lines = [
        "# Random 100-Scenario Hydraulic Verification",
        "",
        "## Setup",
        "",
        f"- Random cases: {len(random_cases)}",
        "- Each case uses 4 distinct emitters sampled without replacement from the 120 emitters.",
        "- Source head is sampled uniformly from 20-30 m.",
        "- EPANET/WNTR uses demand-driven Darcy-Weisbach analysis with fixed emitter demand.",
        "- The project verification model uses roughness-based Darcy-Weisbach with the EPANET 2.2 friction law (Swamee-Jain, Dunlop transition).",
        "- Minor/local loss is excluded from the direct EPANET comparison; `1.1 * h_friction` remains an engineering correction output.",
        "",
        "## Overall Error Metrics",
        "",
        overall_view[["metric", "n", "mae", "rmse", "mape_pct", "max_abs_error", "bias_mean_signed_error"]].to_markdown(index=False),
        "",
        "## Observation-Level Absolute Error Distribution",
        "",
        distribution_view[
            [
                "metric",
                "n",
                "abs_error_mean",
                "abs_error_std",
                "abs_error_p50",
                "abs_error_p95",
                "abs_error_p99",
                "abs_error_max",
                "pct_error_mean",
                "pct_error_p95",
            ]
        ].to_markdown(index=False),
        "",
        "## Case-Level Metric Distribution",
        "",
        case_view[
            [
                "metric",
                "n_cases",
                "mae_mean",
                "mae_std",
                "mae_p50",
                "mae_p95",
                "max_abs_error_mean",
                "max_abs_error_p95",
                "max_abs_error_max",
            ]
        ].to_markdown(index=False),
        "",
        "## Output Files",
        "",
        "- `random100_cases.csv`",
        "- `random100_node_pressure_comparison.csv`",
        "- `random100_pipe_hydraulics_comparison.csv`",
        "- `random100_group_summary_metrics.csv`",
        "- `random100_error_metrics_by_case.csv`",
        "- `random100_error_observations.csv`",
        "- `random100_overall_error_metrics.csv`",
        "- `random100_observation_error_distribution.csv`",
        "- `random100_case_metric_distribution.csv`",
    ]
    (output_dir / "random100_hydraulic_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(
    output_dir: Path,
    env: dict[str, Any],
    cases: list[GroupCase],
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    pressure_metrics = metrics[metrics["metric"].eq("active_node_pressure_m")]
    headloss_metrics = metrics[metrics["metric"].eq("pipe_friction_headloss_m")]
    flow_metrics = metrics[metrics["metric"].eq("pipe_flow_m3s")]

    lines = [
        "# Hydraulic Verification Report",
        "",
        "## Scope",
        "",
        "- Excel network data were converted to WNTR/EPANET models with the tree root (source node) as a fixed-head reservoir and all other nodes as junctions.",
        "- EPANET/WNTR was run in demand-driven mode with fixed emitter demand; pressure-flow reduction was intentionally excluded.",
        "- Headloss option was set to Darcy-Weisbach. Minor losses were set to zero in EPANET and excluded from the direct comparison.",
        "- The project model was evaluated with the same Darcy-Weisbach roughness formulation: roughness, Reynolds number, and the EPANET 2.2 friction factor.",
        "- The engineering correction `h_total = 1.1 * h_friction` is reported separately and is not mixed into the EPANET comparison.",
        "",
        "## Environment",
        "",
        f"- Python: {env['python']}",
        f"- Platform: {env['platform']}",
        f"- numpy: {env['numpy']}",
        f"- pandas: {env['pandas']}",
        f"- wntr: {env['wntr']}",
        "",
        "## Important Interpretation Note",
        "",
        "EPANET's Darcy-Weisbach pipe parameter is roughness, not a fixed Darcy friction factor. "
        "The exported INP files use roughness values in millimetres (UPVC=0.018 mm, PE=0.020 mm). "
        "The project-side verification model now calculates Darcy `f` from roughness and Reynolds number with the EPANET 2.2 all-regime law, "
        "so the comparison is no longer based on the earlier fixed-f simplification.",
        "",
        "## Typical Groups",
        "",
    ]
    for case in cases:
        lines.append(
            f"- {case.case_id}: {case.category}; source_head={case.source_head_m:.3f} m; "
            f"emitters={';'.join(case.emitters)}; {case.notes}"
        )

    lines.extend(
        [
            "",
            "## Group Summary",
            "",
            summary[
                [
                    "case_id",
                    "n_emitters",
                    "total_demand_m3s",
                    "min_my_pressure_friction_only_m",
                    "min_epanet_pressure_m",
                    "sum_my_active_pipe_friction_headloss_m",
                    "sum_epanet_active_pipe_headloss_m",
                    "max_my_path_friction_headloss_m",
                    "max_epanet_path_headloss_m",
                ]
            ].to_markdown(index=False),
            "",
            "## Error Metrics",
            "",
            "Pressure error is computed as roughness-based project model minus EPANET/WNTR.",
            "",
            "### Active Node Pressure",
            "",
            pressure_metrics[["case_id", "mae", "rmse", "mape_pct", "max_abs_error"]].to_markdown(index=False),
            "",
            "### Pipe Flow",
            "",
            flow_metrics[["case_id", "mae", "rmse", "mape_pct", "max_abs_error"]].to_markdown(index=False),
            "",
            "### Pipe Darcy-Weisbach Friction Headloss",
            "",
            headloss_metrics[["case_id", "mae", "rmse", "mape_pct", "max_abs_error"]].to_markdown(index=False),
            "",
            "## Output Files",
            "",
            "- `base_network_darcy_weibach.inp`: base EPANET model without irrigation demands.",
            "- `C01...C06*.inp`: EPANET model for each typical group with fixed demands.",
            "- `node_pressure_comparison.csv`: active node pressure comparison.",
            "- `pipe_hydraulics_comparison.csv`: active pipe flow, velocity, and headloss comparison.",
            "- `error_metrics_by_group.csv`: MAE, RMSE, MAPE, and maximum absolute error.",
            "- `group_summary_metrics.csv`: group-level hydraulic summaries.",
        ]
    )
    (output_dir / "hydraulic_verification_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    network_dir = root / "data" / "network"
    parser = argparse.ArgumentParser(description="Hydraulic verification with WNTR/EPANET and roughness-based Darcy-Weisbach model.")
    add_network_args(parser, network_dir)
    parser.add_argument(
        "--run-data-dir",
        type=Path,
        default=root / "data" / "hydraulic_verification",
        help="Existing optimizer output directory used only for selecting representative groups.",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "hydraulic_verification")
    parser.add_argument("--source-head", type=float, default=25.0)
    parser.add_argument("--source-head-min", type=float, default=20.0)
    parser.add_argument("--source-head-max", type=float, default=30.0)
    parser.add_argument("--emitter-demand", type=float, default=0.013)
    parser.add_argument("--local-loss-ratio", type=float, default=0.1)
    parser.add_argument("--roughness-upvc-mm", type=float, default=0.018)
    parser.add_argument("--roughness-pe-mm", type=float, default=0.020)
    parser.add_argument("--kinematic-viscosity", type=float, default=WATER_KINEMATIC_VISCOSITY_M2S)
    parser.add_argument("--random-cases", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=20260709)
    parser.add_argument("--random-emitters-per-case", type=int, default=4)
    return parser


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="Changing the headloss formula from H-W to D-W will not change the units of the roughness coefficient.",
        category=UserWarning,
    )
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    roughness = {"UPVC": args.roughness_upvc_mm, "PE": args.roughness_pe_mm}
    nodes, pipes, emitters = network_paths(args)
    network = load_network(Path(nodes), Path(pipes) if pipes else None, roughness, Path(emitters) if emitters else None)

    export_inp(args.output_dir / "base_network_darcy_weibach.inp", network, {}, args.source_head)

    cases = select_typical_groups(
        network,
        args.run_data_dir,
        args.source_head,
        args.emitter_demand,
        args.local_loss_ratio,
        args.kinematic_viscosity,
    )
    pd.DataFrame(
        [
            {
                "case_id": c.case_id,
                "category": c.category,
                "source_head_m": c.source_head_m,
                "emitters": ";".join(c.emitters),
                "nodes": ";".join(emitter_node(network, e) for e in c.emitters),
                "origin": c.origin,
                "notes": c.notes,
            }
            for c in cases
        ]
    ).to_csv(args.output_dir / "typical_groups.csv", index=False, encoding="utf-8-sig")

    node_rows: list[dict[str, Any]] = []
    pipe_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for case in cases:
            n_rows, p_rows, s_rows, m_rows = evaluate_case(
                network,
                case,
                args.emitter_demand,
                args.local_loss_ratio,
                args.output_dir,
                args.kinematic_viscosity,
            )
            node_rows.extend(n_rows)
            pipe_rows.extend(p_rows)
            summary_rows.extend(s_rows)
            metrics_rows.extend(m_rows)

    node_df = pd.DataFrame(node_rows)
    pipe_df = pd.DataFrame(pipe_rows)
    summary_df = pd.DataFrame(summary_rows)
    metrics_df = pd.DataFrame(metrics_rows)

    node_df.to_csv(args.output_dir / "node_pressure_comparison.csv", index=False, encoding="utf-8-sig")
    pipe_df.to_csv(args.output_dir / "pipe_hydraulics_comparison.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(args.output_dir / "group_summary_metrics.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(args.output_dir / "error_metrics_by_group.csv", index=False, encoding="utf-8-sig")

    random_cases = select_random_validation_cases(
        network,
        args.random_cases,
        args.random_seed,
        args.source_head_min,
        args.source_head_max,
        args.random_emitters_per_case,
    )
    pd.DataFrame(
        [
            {
                "case_id": c.case_id,
                "category": c.category,
                "source_head_m": c.source_head_m,
                "emitters": ";".join(c.emitters),
                "nodes": ";".join(emitter_node(network, e) for e in c.emitters),
                "origin": c.origin,
                "notes": c.notes,
            }
            for c in random_cases
        ]
    ).to_csv(args.output_dir / "random100_cases.csv", index=False, encoding="utf-8-sig")

    random_node_rows: list[dict[str, Any]] = []
    random_pipe_rows: list[dict[str, Any]] = []
    random_summary_rows: list[dict[str, Any]] = []
    random_metrics_rows: list[dict[str, Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for case in random_cases:
            n_rows, p_rows, s_rows, m_rows = evaluate_case(
                network,
                case,
                args.emitter_demand,
                args.local_loss_ratio,
                args.output_dir,
                args.kinematic_viscosity,
                export_case_inp=False,
            )
            random_node_rows.extend(n_rows)
            random_pipe_rows.extend(p_rows)
            random_summary_rows.extend(s_rows)
            random_metrics_rows.extend(m_rows)

    random_node_df = pd.DataFrame(random_node_rows)
    random_pipe_df = pd.DataFrame(random_pipe_rows)
    random_summary_df = pd.DataFrame(random_summary_rows)
    random_metrics_df = pd.DataFrame(random_metrics_rows)
    random_obs_df = pd.DataFrame(error_observation_rows(random_node_df, random_pipe_df, random_summary_df))
    random_overall_df = overall_metrics_from_observations(random_obs_df)
    random_obs_dist_df = observation_error_distribution(random_obs_df)
    random_case_dist_df = case_metric_distribution(random_metrics_df)

    random_node_df.to_csv(args.output_dir / "random100_node_pressure_comparison.csv", index=False, encoding="utf-8-sig")
    random_pipe_df.to_csv(args.output_dir / "random100_pipe_hydraulics_comparison.csv", index=False, encoding="utf-8-sig")
    random_summary_df.to_csv(args.output_dir / "random100_group_summary_metrics.csv", index=False, encoding="utf-8-sig")
    random_metrics_df.to_csv(args.output_dir / "random100_error_metrics_by_case.csv", index=False, encoding="utf-8-sig")
    random_obs_df.to_csv(args.output_dir / "random100_error_observations.csv", index=False, encoding="utf-8-sig")
    random_overall_df.to_csv(args.output_dir / "random100_overall_error_metrics.csv", index=False, encoding="utf-8-sig")
    random_obs_dist_df.to_csv(args.output_dir / "random100_observation_error_distribution.csv", index=False, encoding="utf-8-sig")
    random_case_dist_df.to_csv(args.output_dir / "random100_case_metric_distribution.csv", index=False, encoding="utf-8-sig")

    env = {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "wntr": wntr.__version__,
        "nodes": nodes,
        "pipes": pipes,
        "emitters": emitters,
        "run_data_dir": str(args.run_data_dir),
        "output_dir": str(args.output_dir),
        "epanet_roughness_mm": roughness,
        "kinematic_viscosity_m2s": args.kinematic_viscosity,
        "emitter_demand_m3s": args.emitter_demand,
        "local_loss_ratio": args.local_loss_ratio,
        "random_cases": args.random_cases,
        "random_seed": args.random_seed,
        "random_emitters_per_case": args.random_emitters_per_case,
    }
    (args.output_dir / "wntr_environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.output_dir, env, cases, summary_df, metrics_df)
    write_random_validation_report(args.output_dir, random_cases, random_overall_df, random_obs_dist_df, random_case_dist_df)

    print(f"Hydraulic verification completed: {args.output_dir}")
    print(summary_df[["case_id", "min_my_pressure_friction_only_m", "min_epanet_pressure_m"]].to_string(index=False))
    print(metrics_df[metrics_df["metric"].eq("active_node_pressure_m")][["case_id", "mae", "rmse", "mape_pct", "max_abs_error"]].to_string(index=False))
    print("Random 100-case overall metrics:")
    print(random_overall_df[["metric", "n", "mae", "rmse", "mape_pct", "max_abs_error"]].to_string(index=False))


if __name__ == "__main__":
    main()
