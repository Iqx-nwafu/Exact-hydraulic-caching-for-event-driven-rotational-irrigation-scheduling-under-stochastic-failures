"""Plan-level equivalence of the research objectives and the workbench ``api.irrigation_metrics``.

Runs the original rotation and fixed-seed random rotations through
``evaluate_plan`` / ``plan_objectives`` (research) and ``hydraulic_moc.api.simulate``
with the research convention (``head_sampling="step_start"``,
``pressure_weighting="segment"``) under the same hourly source-head profiles,
and reports the largest differences of the three objectives, the constraint
violation and the delivered volumes. Requires ``hydraulic_moc`` on ``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main_experiment_after_verification as base  # noqa: E402
from network_topology import add_network_args, network_paths, read_network_tables, to_platform_json  # noqa: E402

RESEARCH_SIM = {"head_sampling": "step_start", "pressure_weighting": "segment"}


def platform_evaluator(network_json: Path, params: base.Params):
    from hydraulic_moc import api

    return api, api.load(network_json, local_loss_ratio=params.local_loss_ratio, min_pressure_m=params.design_head_m,
                         outlet_design_pressure_m=params.design_head_m, velocity_scope="active_only",
                         min_velocity_m_s=params.min_velocity_ms, max_velocity_m_s=params.max_velocity_ms,
                         friction="epanet")


def compare(network: base.Network, params: base.Params, network_json: Path, scenarios: int, random_plans: int,
            seed: int) -> pd.DataFrame:
    api, ev = platform_evaluator(network_json, params)
    base.WORKER_CACHE_ENABLED = False
    rows = []
    for sid in range(scenarios):
        profile = base.source_profile(seed + sid, params)
        plans = [("original", base.original_plan(network, params))]
        plans += [(f"random{r}", base.random_plan(network, params, random.Random(seed + sid * 101 + r)))
                  for r in range(random_plans)]
        head = api.HeadProfile.regular(profile)
        for name, plan in plans:
            obj, cv, m = base.plan_objectives(network, params, plan, profile)
            _, volumes, _ = base.evaluate_plan(network, params, plan, profile)
            steps = [api.Step(tuple(network.emitters[e] for e in g.emitters), g.duration_h) for g in plan]
            sim = api.simulate(ev, steps, head=head, horizon_h=params.horizon_h, **RESEARCH_SIM)
            obj_p = api.plan_objectives(sim.metrics)
            order = [ev.network.outlet_index[e] for e in network.emitters]
            rows.append({
                "scenario_id": sid, "plan": name,
                "achievement_loss_research": obj[0], "achievement_loss_platform": obj_p[0],
                "mean_deficit_research": obj[1], "mean_deficit_platform": obj_p[1],
                "nonuniformity_research": obj[2], "nonuniformity_platform": obj_p[2],
                "cv_research": cv, "cv_platform": sim.metrics["constraint_violation"],
                "pressure_std_research": m["pressure_std_m"], "pressure_std_platform": sim.metrics["pressure_std_m"],
                "max_volume_diff_m3": float(np.max(np.abs(volumes - sim.volumes_m3[order]))),
            })
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame) -> dict:
    out = {"plans": len(df)}
    for key in ("achievement_loss", "mean_deficit", "nonuniformity", "cv", "pressure_std"):
        out[f"max_abs_diff_{key}"] = float((df[f"{key}_research"] - df[f"{key}_platform"]).abs().max())
    out["max_volume_diff_m3"] = float(df["max_volume_diff_m3"].max())
    out["infeasible_plans"] = int((df["cv_research"] > 0).sum())
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    add_network_args(p, root / "data" / "network")
    p.add_argument("--roughness-upvc-mm", type=float, default=0.018)
    p.add_argument("--roughness-pe-mm", type=float, default=0.020)
    p.add_argument("--scenarios", type=int, default=100)
    p.add_argument("--random-plans", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260709)
    p.add_argument("--tolerance", type=float, default=1e-9)
    p.add_argument("--output-dir", type=Path, default=root / "outputs" / "platform_metrics")
    args = p.parse_args()
    params = base.Params()
    roughness = {"UPVC": args.roughness_upvc_mm, "PE": args.roughness_pe_mm}
    nodes, pipes, emitters = network_paths(args)
    network = base.load_network(Path(nodes), Path(pipes) if pipes else None, roughness, Path(emitters) if emitters else None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # The platform side reads the same network through a planning JSON written from the loaded tables.
    tables = read_network_tables(Path(nodes), Path(pipes) if pipes else None, Path(emitters) if emitters else None)
    data = to_platform_json(
        tables, roughness, source_head_m=25.0, design_flow_m3s=params.design_flow_m3s,
        target_volume_m3=params.target_per_emitter_m3, min_pressure_m=params.design_head_m,
        min_velocity_ms=params.min_velocity_ms, max_velocity_ms=params.max_velocity_ms, group_size=params.group_size,
        kinematic_viscosity_m2s=params.kinematic_viscosity_m2s, local_loss_ratio=params.local_loss_ratio)
    network_json = args.output_dir / "network.json"
    network_json.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    df = compare(network, params, network_json, args.scenarios, args.random_plans, args.seed)
    df.to_csv(args.output_dir / "metrics_pairs.csv", index=False)
    info = summary(df)
    info["passed"] = all(v <= args.tolerance for k, v in info.items() if k.startswith("max_"))
    (args.output_dir / "summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    sys.exit(0 if info["passed"] else 1)


if __name__ == "__main__":
    main()
