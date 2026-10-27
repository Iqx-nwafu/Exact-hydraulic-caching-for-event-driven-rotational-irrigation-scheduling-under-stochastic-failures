"""Reduced-scale end-to-end run of the research workflow on one network.

    planning JSON -> platform loader -> EPANET/WNTR verification
      -> research/platform backend equivalence -> optimisation experiments
      -> exact-cache benchmark -> surrogate comparison -> variant sensitivity

Each step calls the published script with ``--network-json`` and writes under
``--output-dir``; ``workflow_report.json`` records every command, its wall time
and the checks listed in ``CHECKS``. Run it on a new network or machine before
launching full-scale experiments. The surrogate step skips its fixed-size
wall-time benchmark, whose code paths the other steps already run. Platform
steps need the workbench package ``hydraulic_moc`` on ``PYTHONPATH``; without
it they are skipped and the report says ``"platform_available": false``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
SMALL = ["--population", "16", "--generations", "4"]
SCENARIOS = ["--normal-scenarios", "2", "--single-fault-scenarios", "2", "--multi-fault-scenarios", "4",
             "--random-repeats", "2", "--nsga-repeats", "1"]
KEYS = ["experiment", "scenario_id", "policy", "repeat", "optimizer_algorithm"]
CHECKS = {
    "platform_load": "the platform loader accepts the JSON and builds one emitter per field",
    "verification": "max |model - EPANET| active-node pressure below --max-epanet-error-m",
    "backend_equivalence": "research and platform evaluators agree on every group (feasibility, pressures)",
    "backend_experiments": "the optimisation experiment gives the same metrics with either backend",
    "cache_equivalence": "cached and uncached runs give identical agricultural metrics",
    "surrogate": "surrogate comparison is paired with this run's main experiment",
    "variant": "variant sensitivity summary is written",
}


def run(step: str, args: list[str], report: dict) -> None:
    cmd = [sys.executable, *args]
    start = time.perf_counter()
    print(f"[{step}] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)
    report["steps"].append({"step": step, "command": cmd, "wall_time_s": round(time.perf_counter() - start, 2)})


def check(report: dict, name: str, ok: bool, **values) -> None:
    report["checks"][name] = {"passed": bool(ok), "criterion": CHECKS[name], **values}
    print(f"[check] {name}: {'PASS' if ok else 'FAIL'} {values}", flush=True)


def metrics_match(a: Path, b: Path) -> float:
    """Max abs difference of the numeric metrics of two experiment runs, ignoring timings and cache counters."""
    left, right = (pd.read_csv(p / "all_policy_metrics_after_verification.csv").sort_values(KEYS).reset_index(drop=True)
                   for p in (a, b))
    cols = [c for c in left.select_dtypes("number").columns if not c.endswith("_time_s") and "cache" not in c]
    if len(left) != len(right):
        return float("inf")
    return float((left[cols] - right[cols]).abs().max().max())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--network-json", type=Path, default=None,
                   help="planning JSON exported by the workbench; default: convert data/network tables")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "workflow")
    p.add_argument("--workers", default="auto")
    p.add_argument("--max-epanet-error-m", type=float, default=0.05)
    p.add_argument("--equivalence-atol", type=float, default=1e-9)
    args = p.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"output_dir": str(out), "steps": [], "checks": {}}
    platform = importlib.util.find_spec("hydraulic_moc") is not None
    workers = ["--workers", str(args.workers)]

    network = args.network_json.resolve() if args.network_json else out / "network.json"
    if args.network_json is None:
        run("to_json", [str(SRC / "network_topology.py"), "to-json", "--out", str(network)], report)
    net = ["--network-json", str(network)]
    report["network_json"] = str(network)

    if platform:
        from hydraulic_moc.irrigation_data import load_planning_project

        project = load_planning_project(network)
        fields = json.loads(network.read_text(encoding="utf-8-sig"))["fields"]
        check(report, "platform_load", len(project.fields) == len(fields) > 0,
              nodes=len(project.topology.nodes), pipes=len(project.topology.pipes), emitters=len(project.fields))

    run("verification", [str(SRC / "hydraulic_check.py"), *net, "--random-cases", "20",
                         "--output-dir", str(out / "hydraulic_verification")], report)
    errors = pd.read_csv(out / "hydraulic_verification" / "random100_overall_error_metrics.csv").set_index("metric")
    err = float(errors.loc["active_node_pressure_m", "max_abs_error"])
    check(report, "verification", err < args.max_epanet_error_m, max_abs_pressure_error_m=err,
          mae_m=float(errors.loc["active_node_pressure_m", "mae"]))

    main_script = str(SRC / "main_experiment_after_verification.py")
    run("main_research", [main_script, *net, *workers, *SCENARIOS, *SMALL, "--output-dir", str(out / "main_research")], report)
    if platform:
        run("equivalence", [str(SRC / "platform_equivalence_check.py"), *net, "--random-groups", "2000",
                            "--output-dir", str(out / "platform_equivalence")], report)
        eq = json.loads((out / "platform_equivalence" / "group_equivalence_summary.json").read_text(encoding="utf-8"))
        check(report, "backend_equivalence",
              eq["feasible_mismatch"] == 0 and eq["max_pressure_abs_err_m"] < args.equivalence_atol,
              groups=eq["groups"], feasible_mismatch=eq["feasible_mismatch"], max_pressure_abs_err_m=eq["max_pressure_abs_err_m"])
        run("main_platform", [main_script, *net, *workers, *SCENARIOS, *SMALL, "--hydraulic-backend", "platform",
                              "--output-dir", str(out / "main_platform")], report)
        diff = metrics_match(out / "main_research", out / "main_platform")
        check(report, "backend_experiments", diff < args.equivalence_atol, max_abs_metric_diff=diff)

    run("cache_benchmark", [str(SRC / "nsga2_cache_benchmark.py"), *net, *workers, *SCENARIOS, *SMALL,
                            "--output-base", str(out / "cache_benchmark")], report)
    cache = pd.read_csv(out / "cache_benchmark" / "v2_cache_metric_equivalence_comparison.csv")
    cache = cache[~cache["metric"].str.endswith("_time_s")]
    check(report, "cache_equivalence", int(cache["nonzero_pairs_gt_1e-9"].sum()) == 0,
          metrics=len(cache), max_abs_difference=float(cache["max_abs_difference"].max()))

    run("surrogate", [str(SRC / "surrogate_assisted_nsga2.py"), *net, *workers, *SCENARIOS, *SMALL,
                      "--training-samples", "300", "--train-threads", "2", "--skip-efficiency-benchmark",
                      "--baseline-main-dir", str(out / "main_research"), "--output-dir", str(out / "surrogate")], report)
    paired = out / "surrogate" / "surrogate_vs_exact_nsga2_paired_comparison.csv"
    check(report, "surrogate", paired.exists() and len(pd.read_csv(paired)) > 0, paired_csv=str(paired))

    run("variant", [str(SRC / "variant_generalization_analysis.py"), *net, *workers, "--scenarios-per-level", "2",
                    "--random-repeats", "2", *SMALL, "--output-dir", str(out / "variant_network")], report)
    summary = out / "variant_network" / "variant_summary_with_ci.csv"
    check(report, "variant", summary.exists() and len(pd.read_csv(summary)) > 0, summary_csv=str(summary))

    report["platform_available"] = platform
    report["passed"] = all(c["passed"] for c in report["checks"].values())
    (out / "workflow_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"workflow {'PASSED' if report['passed'] else 'FAILED'}; report: {out / 'workflow_report.json'}")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
