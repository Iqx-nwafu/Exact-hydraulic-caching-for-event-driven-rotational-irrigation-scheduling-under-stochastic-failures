from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wntr

from revision_core import HydraulicParams, load_network, solve_group_pda


NODES_PATH: Path | None = None
PIPES_PATH: Path | None = None


def init_worker(nodes_path: str, pipes_path: str) -> None:
    global NODES_PATH, PIPES_PATH
    NODES_PATH = Path(nodes_path)
    PIPES_PATH = Path(pipes_path)


def build_wntr_model(network: Any, params: HydraulicParams, emitters: tuple[int, ...], source_head_m: float) -> Any:
    node_counts: dict[str, int] = {}
    for emitter in emitters:
        node = network.emitter_to_node[emitter]
        node_counts[node] = node_counts.get(node, 0) + 1

    model = wntr.network.WaterNetworkModel()
    model.options.hydraulic.headloss = "D-W"
    model.options.hydraulic.demand_model = "PDA"
    model.options.hydraulic.minimum_pressure = 0.0
    model.options.hydraulic.required_pressure = params.design_head_m
    model.options.hydraulic.pressure_exponent = 1.0
    model.options.hydraulic.accuracy = 1.0e-8
    model.options.hydraulic.trials = 1000
    model.options.hydraulic.headerror = 1.0e-8
    model.options.hydraulic.flowchange = 1.0e-10
    model.options.time.duration = 0
    model.options.time.hydraulic_timestep = 3600
    model.options.time.report_timestep = 3600
    model.options.hydraulic.inpfile_units = "LPS"
    model.add_reservoir("J0", base_head=float(source_head_m))
    for node in network.node_ids:
        if node == "J0":
            continue
        model.add_junction(
            node,
            base_demand=float(node_counts.get(node, 0) * params.design_flow_m3s),
            elevation=float(network.node_z[network.node_index[node]]),
        )
    for pipe_idx, pipe_id in enumerate(network.pipe_ids):
        model.add_pipe(
            pipe_id,
            network.from_nodes[pipe_idx],
            network.to_nodes[pipe_idx],
            length=float(network.lengths_m[pipe_idx]),
            diameter=float(network.diameters_m[pipe_idx]),
            roughness=float(network.roughness_mm[pipe_idx] / 1000.0),
            minor_loss=0.0,
            initial_status="OPEN",
        )
    return model


def error_statistics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    error = estimate - reference
    absolute = np.abs(error)
    nonzero = np.abs(reference) > 1.0e-10
    return {
        "n": int(len(reference)),
        "mae": float(np.mean(absolute)) if len(absolute) else math.nan,
        "rmse": float(np.sqrt(np.mean(np.square(error)))) if len(error) else math.nan,
        "mape_pct": float(np.mean(absolute[nonzero] / np.abs(reference[nonzero])) * 100.0) if np.any(nonzero) else math.nan,
        "max_abs_error": float(np.max(absolute)) if len(absolute) else math.nan,
    }


def evaluate_case(case: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if NODES_PATH is None or PIPES_PATH is None:
        raise RuntimeError("worker was not initialized")
    network = load_network(NODES_PATH, PIPES_PATH)
    params = HydraulicParams()
    emitters = tuple(int(value) for value in case["emitters"])
    source_head = float(case["source_head_m"])
    custom = solve_group_pda(network, params, emitters, set(), source_head)
    model = build_wntr_model(network, params, emitters, source_head)
    with tempfile.TemporaryDirectory(prefix=f"pda_case_{case['case_id']}_") as directory:
        prefix = str(Path(directory) / f"case_{case['case_id']}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = wntr.sim.EpanetSimulator(model).run_sim(
                file_prefix=prefix,
                version=2.2,
                convergence_error=True,
            )
    ep_pressure = result.node["pressure"].iloc[0]
    ep_head = result.node["head"].iloc[0]
    ep_demand = result.node["demand"].iloc[0]
    ep_flow = result.link["flowrate"].iloc[0]

    rows: list[dict[str, Any]] = []
    active_nodes = sorted(set(network.emitter_to_node[e] for e in emitters))
    for node in active_nodes:
        local_indices = [i for i, e in enumerate(custom.active_emitters) if network.emitter_to_node[e] == node]
        custom_demand = float(np.sum(custom.delivered_flow_m3s[local_indices]))
        rows.append(
            {
                "case_id": case["case_id"],
                "case_class": case["case_class"],
                "source_head_m": source_head,
                "group_size": len(emitters),
                "metric": "active_node_pressure_m",
                "entity": node,
                "reference": float(ep_pressure[node]),
                "estimate": float(custom.node_pressure[node]),
            }
        )
        rows.append(
            {
                "case_id": case["case_id"],
                "case_class": case["case_class"],
                "source_head_m": source_head,
                "group_size": len(emitters),
                "metric": "active_node_demand_m3s",
                "entity": node,
                "reference": float(ep_demand[node]),
                "estimate": custom_demand,
            }
        )

    active_pipes = np.flatnonzero(np.abs(custom.pipe_flows_m3s) > 1.0e-12)
    for pipe_idx in active_pipes:
        pipe_id = network.pipe_ids[pipe_idx]
        reference_flow = float(ep_flow[pipe_id])
        reference_velocity = reference_flow / float(network.areas_m2[pipe_idx])
        reference_loss = float(
            ep_head[network.from_nodes[pipe_idx]] - ep_head[network.to_nodes[pipe_idx]]
        )
        for metric, reference, estimate in (
            ("active_pipe_flow_m3s", reference_flow, custom.pipe_flows_m3s[pipe_idx]),
            ("active_pipe_velocity_ms", reference_velocity, custom.velocities_ms[pipe_idx]),
            ("active_pipe_friction_loss_m", reference_loss, custom.friction_loss_m[pipe_idx]),
        ):
            rows.append(
                {
                    "case_id": case["case_id"],
                    "case_class": case["case_class"],
                    "source_head_m": source_head,
                    "group_size": len(emitters),
                    "metric": metric,
                    "entity": pipe_id,
                    "reference": float(reference),
                    "estimate": float(estimate),
                }
            )

    summary = {
        "case_id": case["case_id"],
        "case_class": case["case_class"],
        "source_head_m": source_head,
        "group_size": len(emitters),
        "emitters": ";".join(network.emitters[e] for e in emitters),
        "custom_converged": custom.converged,
        "custom_iterations": custom.iterations,
        "custom_flow_residual_m3s": custom.flow_residual_m3s,
        "custom_head_residual_m": custom.head_residual_m,
    }
    return rows, summary


def build_cases(count: int, seed: int, emitter_count: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    cases: list[dict[str, Any]] = []
    for case_id in range(1, count + 1):
        operational = case_id <= count // 2
        source_head = rng.uniform(20.0, 30.0) if operational else rng.uniform(8.0, 20.0)
        group_size = int(rng.integers(1, 9))
        emitters = tuple(sorted(rng.choice(emitter_count, size=group_size, replace=False).tolist()))
        cases.append(
            {
                "case_id": case_id,
                "case_class": "operational_20_30m" if operational else "low_pressure_stress_8_20m",
                "source_head_m": float(source_head),
                "emitters": emitters,
            }
        )
    return cases


def summarize_comparisons(comparisons: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    case_rows: list[dict[str, Any]] = []
    for (case_id, case_class, metric), group in comparisons.groupby(["case_id", "case_class", "metric"]):
        stats = error_statistics(group["reference"].to_numpy(), group["estimate"].to_numpy())
        case_rows.append({"case_id": case_id, "case_class": case_class, "metric": metric, **stats})
    case_metrics = pd.DataFrame(case_rows)
    distribution_rows: list[dict[str, Any]] = []
    for (case_class, metric), group in case_metrics.groupby(["case_class", "metric"]):
        for statistic in ("mae", "rmse", "mape_pct", "max_abs_error"):
            values = group[statistic].dropna().to_numpy(dtype=float)
            distribution_rows.append(
                {
                    "case_class": case_class,
                    "metric": metric,
                    "statistic": statistic,
                    "n_cases": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "p05": float(np.quantile(values, 0.05)),
                    "median": float(np.quantile(values, 0.50)),
                    "p95": float(np.quantile(values, 0.95)),
                    "maximum": float(np.max(values)),
                }
            )
    return case_metrics, pd.DataFrame(distribution_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the closed linear-PDA solver against EPANET 2.2 PDA")
    repository_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--nodes", type=Path, default=repository_root / "data" / "network" / "Nodes.xlsx")
    parser.add_argument("--pipes", type=Path, default=repository_root / "data" / "network" / "Pipes.xlsx")
    parser.add_argument("--output-dir", type=Path, default=repository_root / "outputs" / "hydraulic_validation")
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    network = load_network(args.nodes, args.pipes)
    cases = build_cases(args.cases, args.seed, len(network.emitters))
    started = time.perf_counter()
    comparison_rows: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    # EPANET's toolkit state is not reliable under concurrent WNTR calls on this Windows build.
    # The validation reference is therefore intentionally serial; all actual parallel experiments use 20 workers.
    init_worker(str(args.nodes), str(args.pipes))
    for completed, case in enumerate(cases, start=1):
        rows, summary = evaluate_case(case)
        comparison_rows.extend(rows)
        case_summaries.append(summary)
        if completed == 1 or completed % 20 == 0 or completed == len(cases):
            print(f"PDA validation {completed}/{len(cases)}", flush=True)
    comparisons = pd.DataFrame(comparison_rows)
    cases_df = pd.DataFrame(case_summaries).sort_values("case_id")
    case_metrics, distribution = summarize_comparisons(comparisons)
    comparisons["error"] = comparisons["estimate"] - comparisons["reference"]
    comparisons["absolute_error"] = comparisons["error"].abs()
    comparisons.to_csv(args.output_dir / "pda_epanet_entity_comparisons.csv", index=False, encoding="utf-8-sig")
    cases_df.to_csv(args.output_dir / "pda_validation_cases.csv", index=False, encoding="utf-8-sig")
    case_metrics.to_csv(args.output_dir / "pda_error_metrics_by_case.csv", index=False, encoding="utf-8-sig")
    distribution.to_csv(args.output_dir / "pda_error_metric_distributions.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.replace("\n", " "),
        "wntr": wntr.__version__,
        "epanet_version": "2.2",
        "workers": 1,
        "parallel": False,
        "serial_reason": "WNTR 1.5.0 EPANET toolkit result isolation is unreliable under concurrent Windows processes",
        "case_count": args.cases,
        "seed": args.seed,
        "pda_model": "q=q_required*clip(H/11.4,0,1), iterated to hydraulic closure",
        "wall_time_s": time.perf_counter() - started,
        "all_custom_cases_converged": bool(cases_df["custom_converged"].all()),
    }
    (args.output_dir / "pda_validation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(distribution.to_string(index=False), flush=True)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
