"""Adapter to the hydraulic workbench's headless group evaluator.

``--hydraulic-backend platform`` routes every cache miss of
``compute_hydraulic_base`` to ``hydraulic_moc.irrigation_eval.GroupEvaluator``.
The group and plan caches of the experiment stay in front of it; the
per-pipe loss cache belongs to the research backend and records no traffic
here, so cache benchmarks should use the research backend. ``hydraulic_moc`` ships with the
workbench and is only imported when this backend is selected.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def build_platform_evaluator(network: Any, params: Any, friction: str = "epanet") -> Any:
    from hydraulic_moc.irrigation_data import IrrigationNode, IrrigationPipe, IrrigationTopology
    from hydraulic_moc.irrigation_eval import Emitter, GroupEvaluator, Limits, SteadyModel, TreeNetwork

    nodes = tuple(
        IrrigationNode(str(row.node_id), float(row.Z), float(row.X), float(row.Y))
        for row in network.nodes.itertuples(index=False)
    )
    pipes = tuple(
        IrrigationPipe(
            pipe_id=pid,
            start_node=start,
            end_node=end,
            length_m=float(length),
            diameter_m=float(diameter),
            headloss_formula="D-W",
            roughness_m=float(roughness) / 1000.0,
            kinematic_viscosity_m2_s=params.kinematic_viscosity_m2s,
        )
        for pid, start, end, length, diameter, roughness in zip(
            network.pipe_ids, network.from_nodes, network.to_nodes,
            network.lengths_m, network.diameters_m, network.roughness_mm,
        )
    )
    topology = IrrigationTopology(nodes, pipes, network.source_node, "research network")
    emitters = [
        Emitter(name, node, params.design_flow_m3s, params.design_head_m)
        for name, node in zip(network.emitters, network.emitter_to_node)
    ]
    tree = TreeNetwork(topology, emitters)
    model = SteadyModel(friction=friction, local_loss_ratio=params.local_loss_ratio)
    limits = Limits(params.design_head_m, params.min_velocity_ms, params.max_velocity_ms)
    # The experiment's own LRU caches sit in front of the evaluator.
    return GroupEvaluator(tree, model, limits, cache_items=0)


def platform_base(evaluator: Any, network: Any, params: Any, key: tuple[int, ...]) -> dict[str, Any]:
    """Same dict as ``compute_hydraulic_base`` for the research backend."""

    flow, velocity, factor, head_loss, path_loss = (a[0] for a in evaluator.hydraulics(_mask(evaluator, key)))
    tree = evaluator.network
    friction_loss = factor * (tree.length_m / tree.diameter_m) * velocity ** 2 / (2.0 * evaluator.model.gravity_m_s2)
    node_counts = Counter(network.emitter_to_node[e] for e in key)
    node_path_loss = {node: float(path_loss[tree.node_index[node]]) for node in node_counts}
    active = flow > 1e-12
    v = velocity[active]
    low = v < params.min_velocity_ms
    high = v > params.max_velocity_ms
    return {
        "node_counts": dict(node_counts),
        "node_path_loss": node_path_loss,
        "pipe_flows": flow,
        "velocities": velocity,
        "friction_loss": friction_loss,
        "head_loss": head_loss,
        "active_velocities": v,
        "max_velocity_ms": float(v.max()) if len(v) else 0.0,
        "min_velocity_ms": float(v.min()) if len(v) else 0.0,
        "velocity_violation_ms": float((params.min_velocity_ms - v[low]).sum() + (v[high] - params.max_velocity_ms).sum()),
        "velocity_violation_count": int(low.sum() + high.sum()),
    }


def _mask(evaluator: Any, key: tuple[int, ...]) -> np.ndarray:
    mask = np.zeros((1, evaluator.network.shape[2]), dtype=bool)
    mask[0, list(key)] = True
    return mask
