"""Optimality gap of the evolutionary rotation optimisers against certified optima.

For networks reduced to the first ``k`` laterals, under one constant source head
(normal operation: every feasible plan delivers the full volume, so the
pressure-uniformity index is the objective that separates plans), the workbench
``hydraulic_moc.api.optimize_exact`` (API 1.2) gives the best partition into
groups of four with a proved upper bound.  NSGA-II, SPEA2, RVEA and MOEA/D
(``optimize_nsga2_plan``) and random rotations are then scored on the same
instance; the gap is the certified optimum minus the optimiser's uniformity.
Requires ``hydraulic_moc`` on ``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main_experiment_after_verification as base  # noqa: E402
from network_topology import (  # noqa: E402
    add_network_args, derive_tree, network_paths, read_network_tables, to_platform_json,
)

ALGORITHMS = ("nsga2", "spea2", "rvea", "moead")


def reduced_tables(tables, laterals: int):
    """Emitters of the first ``laterals`` laterals and the pipes that feed them."""
    tree = derive_tree(tables)
    keep_nodes = {n for lateral in tree.laterals[:laterals] for n in lateral}
    emitters = tables.emitters[tables.emitters["node_id"].isin(keep_nodes)].reset_index(drop=True)
    parent = {r.ToNode: r.FromNode for r in tables.pipes.itertuples(index=False)}
    need = {tree.source_node}
    for node in emitters["node_id"]:
        while node != tree.source_node:
            need.add(node)
            node = parent[node]
    nodes = tables.nodes[tables.nodes["node_id"].isin(need)].reset_index(drop=True)
    pipes = tables.pipes[tables.pipes["FromNode"].isin(need) & tables.pipes["ToNode"].isin(need)].reset_index(drop=True)
    return replace(tables, nodes=nodes, pipes=pipes, emitters=emitters)


def write_instance(tables, roughness, params: base.Params, head: float, path: Path) -> None:
    data = to_platform_json(
        tables, roughness, source_head_m=head, design_flow_m3s=params.design_flow_m3s,
        target_volume_m3=params.target_per_emitter_m3, min_pressure_m=params.design_head_m,
        min_velocity_ms=params.min_velocity_ms, max_velocity_ms=params.max_velocity_ms, group_size=params.group_size,
        kinematic_viscosity_m2s=params.kinematic_viscosity_m2s, local_loss_ratio=params.local_loss_ratio)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def exact(path: Path, params: base.Params, head: float, time_limit: float) -> dict:
    from hydraulic_moc import api

    ev = api.load(path, local_loss_ratio=params.local_loss_ratio, min_pressure_m=params.design_head_m,
                  outlet_design_pressure_m=params.design_head_m, velocity_scope="active_only",
                  min_velocity_m_s=params.min_velocity_ms, max_velocity_m_s=params.max_velocity_ms,
                  friction="epanet", max_source_flow_m3_h=None)
    r = api.optimize_exact(ev, source_head_m=head, group_size=params.group_size, step_h=params.target_duration_h,
                           time_limit_s=time_limit)
    return r.to_dict()


def heuristics(path: Path, params: base.Params, head: float, seeds: int, population: int, generations: int,
               random_plans: int, exact_groups: list[list[str]]) -> list[dict]:
    network = base.load_network(path, None, {})
    base.WORKER_NETWORK = network
    profile = [head] * (int(params.horizon_h) + 1)
    original = [e for g in base.original_plan(network, params) for e in g.emitters]
    rows = []

    def score(policy, seed, plan, elapsed):
        _, cv, m = base.plan_objectives(network, params, plan, profile)
        rows.append({"policy": policy, "seed": seed, "pressure_uniformity_pct": m["pressure_uniformity_index_pct"],
                     "cv": cv, "achievement_pct": m["achievement_pct"], "elapsed_s": elapsed})

    if exact_groups:  # the certified plan scored by the research evaluator (checks the two implementations agree)
        score("exact", 0, [base.Group(tuple(network.emitter_index[e] for e in grp), params.target_duration_h, "exact")
                           for grp in exact_groups], 0.0)
    score("original_near_far", 0, base.original_plan(network, params), 0.0)
    for s in range(1, random_plans + 1):
        score("random_rotation", s, base.random_plan(network, params, random.Random(s)), 0.0)
    for algorithm in ALGORITHMS:
        for s in range(1, seeds + 1):
            t0 = time.perf_counter()
            plan, _, _, _ = base.optimize_nsga2_plan(list(range(len(network.emitters))), params, profile, 1009 * s,
                                                     population, generations, initial_orders=[original],
                                                     algorithm=algorithm)
            score(algorithm, s, plan, time.perf_counter() - t0)
    return rows


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    add_network_args(p, root / "data" / "network")
    p.add_argument("--roughness-upvc-mm", type=float, default=0.018)
    p.add_argument("--roughness-pe-mm", type=float, default=0.020)
    p.add_argument("--laterals", type=str, default="2,3")
    p.add_argument("--source-head", type=float, default=25.0, help="constant source HGL (m)")
    p.add_argument("--exact-time-limit", type=float, default=3600.0)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--population", type=int, default=48)
    p.add_argument("--generations", type=int, default=50)
    p.add_argument("--random-plans", type=int, default=50)
    p.add_argument("--output-dir", type=Path, default=root / "outputs" / "optimality_gap")
    args = p.parse_args()
    params = base.Params()
    roughness = {"UPVC": args.roughness_upvc_mm, "PE": args.roughness_pe_mm}
    nodes, pipes, emitters = network_paths(args)
    tables = read_network_tables(Path(nodes), Path(pipes) if pipes else None, Path(emitters) if emitters else None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, certificates = [], []
    for k in (int(x) for x in args.laterals.split(",")):
        path = args.output_dir / f"network_lat{k}.json"
        reduced = reduced_tables(tables, k)
        write_instance(reduced, roughness, params, args.source_head, path)
        cert = exact(path, params, args.source_head, args.exact_time_limit)
        cert["laterals"] = k
        certificates.append(cert)
        print(json.dumps({k2: v for k2, v in cert.items() if k2 != "groups"}), flush=True)
        for row in heuristics(path, params, args.source_head, args.seeds, args.population, args.generations,
                              args.random_plans, cert["groups"]):
            row.update(laterals=k, emitters=len(reduced.emitters),
                       exact_pui_pct=cert["pressure_uniformity_pct"],
                       exact_upper_bound_pct=cert["uniformity_upper_bound_pct"])
            row["gap_pct_points"] = row["exact_pui_pct"] - row["pressure_uniformity_pct"]
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "gap_runs.csv", index=False)
    feasible = df[df["cv"] <= 1e-9]
    summary = (feasible.groupby(["laterals", "emitters", "policy"])["gap_pct_points"]
               .agg(["count", "mean", "std", "min", "max"]).reset_index())
    share = (df.assign(feasible_share_pct=(df["cv"] <= 1e-9) * 100.0)
             .groupby(["laterals", "emitters", "policy"])["feasible_share_pct"].mean().reset_index())
    summary = share.merge(summary, how="left")
    summary.to_csv(args.output_dir / "gap_summary.csv", index=False)
    (args.output_dir / "certificates.json").write_text(json.dumps(certificates, indent=1), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
