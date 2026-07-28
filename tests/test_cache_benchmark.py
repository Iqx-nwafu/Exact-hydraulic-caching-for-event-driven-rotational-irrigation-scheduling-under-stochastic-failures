from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

import nsga2_cache_benchmark as benchmark


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output_base=tmp_path,
        python=Path("python"),
        main_script=Path("main.py"),
        workers="2",
        normal_scenarios=1,
        single_fault_scenarios=2,
        multi_fault_scenarios=3,
        random_repeats=4,
        nsga_repeats=5,
        population=6,
        generations=7,
        base_seed=8,
    )


def test_run_case_builds_expected_subprocess_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[list[str]] = []

    def fake_run(command, check):
        assert check
        captured.append(command)

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    out = benchmark.run_case(_args(tmp_path), "without", disable_cache=True)
    assert out == tmp_path / "without"
    assert captured[0][:2] == ["python", "main.py"]
    assert "--disable-cache" in captured[0]
    assert captured[0][captured[0].index("--dynamic-algorithms") + 1] == "nsga2"


def _write_run(path: Path, wall_time: float, achievement: float, cache_hits: int) -> None:
    path.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "experiment": "normal",
                "scenario_id": 1,
                "policy": "nsga2_verified",
                "repeat": 1,
                "seed": 10,
                "achievement_pct": achievement,
                "unmet_emitter_ratio_pct": 100.0 - achievement,
                "average_deficit_m3_per_emitter": 1.0,
                "max_deficit_m3": 2.0,
                "pressure_uniformity_index_pct": 90.0,
                "feasible_group_ratio_pct": 100.0,
                "total_reoptimization_time_s": 1.0,
                "initial_optimization_time_s": 2.0,
                "cache_group_hits": cache_hits,
            }
        ]
    ).to_csv(path / "all_policy_metrics_after_verification.csv", index=False)
    metadata = {
        "total_wall_time_s": wall_time,
        "normal_scenarios": 1,
        "single_fault_scenarios": 2,
        "multi_fault_scenarios": 3,
        "workers": 4,
        "population": 8,
        "generations": 9,
    }
    (path / "experiment_metadata_after_verification.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )


def test_read_and_compare_runs_write_equivalence_and_timing_outputs(
    tmp_path: Path,
) -> None:
    no_dir = tmp_path / "no"
    cache_dir = tmp_path / "cache"
    output = tmp_path / "comparison"
    _write_run(no_dir, 10.0, 90.0, 0)
    _write_run(cache_dir, 5.0, 90.0, 12)

    metrics, metadata = benchmark.read_run(no_dir)
    assert metrics.iloc[0]["achievement_pct"] == 90.0
    assert metadata["total_wall_time_s"] == 10.0

    benchmark.compare_runs(output, no_dir, cache_dir)
    equivalence = pd.read_csv(output / "v2_cache_metric_equivalence_comparison.csv")
    wall = pd.read_csv(output / "v2_cache_wall_time_comparison.csv")
    timing = pd.read_csv(output / "v2_cache_time_and_hit_rate_summary.csv")
    assert equivalence["max_abs_difference"].fillna(0.0).max() == 0.0
    assert wall.iloc[0]["speedup_factor_no_cache_div_with_cache"] == 2.0
    assert wall.iloc[0]["time_reduction_pct"] == 50.0
    assert set(timing["run"]) == {"no_cache", "with_cache"}
    assert (output / "v2_cache_acceleration_benchmark_report.md").exists()
