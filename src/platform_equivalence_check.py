"""Group-level equivalence of the research and platform hydraulic backends.

Evaluates the published 100 verification cases plus fixed-seed random groups
(with fault-unavailable emitters) through ``hydraulic_group`` under both
backends and reports the maximum absolute differences. Requires the
workbench package ``hydraulic_moc`` on ``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main_experiment_after_verification as base  # noqa: E402
from network_topology import add_network_args, network_paths  # noqa: E402

FIELDS = ("pipe_flows", "velocities", "friction_loss", "head_loss")


def cases(network: base.Network, root: Path, n_random: int, seed: int) -> list[tuple[str, tuple[int, ...], set[int], float]]:
    out = []
    table = pd.read_csv(root / "data" / "hydraulic_verification" / "random100_cases.csv", encoding="utf-8-sig")
    for row in table.itertuples(index=False):
        names = str(row.emitters).split(";")
        if all(e in network.emitter_index for e in names):  # published cases only fit the default network
            out.append((str(row.case_id), tuple(network.emitter_index[e] for e in names), set(), float(row.source_head_m)))
    rng = random.Random(seed)
    nodes = [n for n in network.downstream_emitters if n != network.source_node]
    for i in range(n_random):
        size = rng.randint(1, 8)
        group = tuple(rng.sample(range(len(network.emitters)), size))
        unavailable: set[int] = set()
        for node in rng.sample(nodes, rng.randint(0, 2)):
            unavailable |= set(network.downstream_emitters[node])
        out.append((f"G{i:05d}", group, unavailable, rng.uniform(15.0, 30.0)))
    return out


def evaluate(backend: str, network, params, items) -> tuple[list[dict], float]:
    base.WORKER_HYDRAULIC_BACKEND = backend
    base.WORKER_CACHE_ENABLED = False
    start = time.perf_counter()
    results = [base.hydraulic_group(network, params, g, u, h) for _, g, u, h in items]
    return results, time.perf_counter() - start


def compare(network, params, items) -> tuple[pd.DataFrame, dict]:
    research, t_research = evaluate("research", network, params, items)
    platform, t_platform = evaluate("platform", network, params, items)
    base.WORKER_HYDRAULIC_BACKEND = "research"
    rows = []
    for (case_id, *_), a, b in zip(items, research, platform):
        row = {"case_id": case_id, "feasible_equal": a["feasible"] == b["feasible"],
               "active_equal": a["active_emitters"] == b["active_emitters"]}
        nodes = set(a["node_pressure"]) | set(b["node_pressure"])
        row["pressure_abs_err_m"] = max((abs(a["node_pressure"][n] - b["node_pressure"][n]) for n in nodes), default=0.0)
        row["flow_abs_err_m3s"] = float(np.max(np.abs(a["pipe_flows"] - b["pipe_flows"])))
        row["velocity_abs_err_ms"] = float(np.max(np.abs(a["velocities"] - b["velocities"])))
        fa = a["delivered_flow"]
        row["delivered_abs_err_m3s"] = max((abs(fa[e] - b["delivered_flow"][e]) for e in fa), default=0.0)
        rows.append(row)
    df = pd.DataFrame(rows)
    summary = {
        "groups": len(df),
        "feasible_mismatch": int((~df["feasible_equal"]).sum()),
        "active_set_mismatch": int((~df["active_equal"]).sum()),
        "max_pressure_abs_err_m": float(df["pressure_abs_err_m"].max()),
        "max_flow_abs_err_m3s": float(df["flow_abs_err_m3s"].max()),
        "max_velocity_abs_err_ms": float(df["velocity_abs_err_ms"].max()),
        "max_delivered_abs_err_m3s": float(df["delivered_abs_err_m3s"].max()),
        "research_seconds": t_research,
        "platform_seconds": t_platform,
        "local_loss_ratio": params.local_loss_ratio,
    }
    return df, summary


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    add_network_args(p, root / "data" / "network")
    p.add_argument("--random-groups", type=int, default=20000)
    p.add_argument("--seed", type=int, default=20260709)
    p.add_argument("--roughness-upvc-mm", type=float, default=0.018)
    p.add_argument("--roughness-pe-mm", type=float, default=0.020)
    p.add_argument("--output-dir", type=Path, default=root / "outputs" / "platform_equivalence")
    args = p.parse_args()
    nodes, pipes, emitters = network_paths(args)
    roughness = {"UPVC": args.roughness_upvc_mm, "PE": args.roughness_pe_mm}
    network = base.load_network(Path(nodes), Path(pipes) if pipes else None, roughness, Path(emitters) if emitters else None)
    params = base.Params()
    df, summary = compare(network, params, cases(network, root, args.random_groups, args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_dir / "group_equivalence.csv", index=False)
    (args.output_dir / "group_equivalence_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
