"""Evaluate rotation groups on a planning JSON with the workbench evaluator.

    PYTHONPATH=<workbench source> python examples/evaluate_groups.py outputs/workflow/network.json

Shows the two calls the research workflow relies on: one group under a node
failure and a vectorised batch. New studies should start from the workbench's
general headless API (docs/headless_api.md in the workbench source).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from hydraulic_moc.irrigation_data import load_planning_project
from hydraulic_moc.irrigation_eval import Emitter, GroupEvaluator, Limits, SteadyModel, TreeNetwork, group_mask

DESIGN_PRESSURE_M = 11.4  # emitter design head of the research model


def build(path: Path, friction: str) -> GroupEvaluator:
    project = load_planning_project(path)
    s = project.settings
    emitters = [Emitter(f.field_id, f.node_id, f.design_flow_m3_h / 3600.0, DESIGN_PRESSURE_M) for f in project.fields]
    limits = Limits(s.get("min_pressure_m", 11.4), s.get("min_velocity_m_s", 0.3), s.get("max_velocity_m_s", 2.5))
    return GroupEvaluator(TreeNetwork(project.topology, emitters), SteadyModel(friction=friction, local_loss_ratio=0.1), limits)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("network_json", type=Path)
    p.add_argument("--friction", choices=("epanet", "colebrook"), default="epanet")
    args = p.parse_args()
    ev = build(args.network_json, args.friction)
    net = ev.network
    print(f"{net.shape[0]} nodes, {net.shape[1]} pipes, {net.shape[2]} emitters; digest {ev.digest}")

    # One group, source head 25 m, the node of emitter 2 has failed (its subtree is cut off).
    failed = net.emitters[2].node_id
    r = ev.evaluate([0, 1, 2, 3], source_head_m=25.0, unavailable=net.downstream_emitters(failed))
    print(f"group [0, 1, 2, 3], node {failed} failed -> open {r.active_emitters}, feasible={r.feasible}, "
          f"min pressure {r.min_pressure_m:.3f} m, max velocity {r.max_velocity_m_s:.3f} m/s")

    # Vectorised batch: 10000 random groups of four, source head 20-30 m.
    rng = np.random.default_rng(20260709)
    groups = [rng.choice(net.shape[2], 4, replace=False) for _ in range(10000)]
    heads = rng.uniform(20.0, 30.0, len(groups))
    start = time.perf_counter()
    b = ev.evaluate_batch(group_mask(net, groups), heads)
    us = (time.perf_counter() - start) / len(groups) * 1e6
    print(f"batch of {len(groups)}: {b.feasible.mean():.1%} feasible, {us:.1f} us/group")


if __name__ == "__main__":
    main()
