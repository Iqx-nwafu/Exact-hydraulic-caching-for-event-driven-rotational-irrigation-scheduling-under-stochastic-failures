from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import main_experiment_after_verification as m
import variant_generalization_analysis as v


def test_variant_spec_matrix_has_expected_baselines() -> None:
    specs = v.variant_specs()
    assert len(specs) == 9
    assert {spec.category for spec in specs} == {
        "elevation_disturbance",
        "fault_probability",
        "pressure_range",
    }
    assert sum(spec.level == spec.baseline_level for spec in specs) == 3


def test_source_profiles_are_deterministic_and_bounded() -> None:
    params = m.Params()
    left = v.source_profile(12, params, 18.0, 22.0, hours=5)
    right = v.source_profile(12, params, 18.0, 22.0, hours=5)
    assert left == right
    assert all(18.0 <= value <= 22.0 for value in left)


def test_elevation_disturbance_preserves_source_and_is_reproducible(
    experiment_network: m.Network,
) -> None:
    assert v.network_with_elevation_disturbance(experiment_network, 0.0, 1) is experiment_network
    first = v.network_with_elevation_disturbance(experiment_network, 1.0, 42)
    second = v.network_with_elevation_disturbance(experiment_network, 1.0, 42)
    assert first is not experiment_network
    assert first.node_z["J0"] == experiment_network.node_z["J0"]
    assert first.node_z == second.node_z
    assert any(
        first.node_z[node] != experiment_network.node_z[node]
        for node in experiment_network.irrigation_nodes
    )
    assert first.nodes.set_index("node_id")["Z"].to_dict() == first.node_z


def test_fault_event_generation_respects_probabilities_and_order(
    monkeypatch: pytest.MonkeyPatch,
    experiment_network: m.Network,
) -> None:
    monkeypatch.setattr(v, "BASE_MODULE", m)
    assert v.fault_events(1, experiment_network, m.Params(), 0.0) == []
    events = v.fault_events(
        1,
        experiment_network,
        m.Params(),
        fault_probability=1.0,
        recovery_probability=0.0,
    )
    assert len(events) == 3
    assert all(event.kind == "fault" for event in events)
    assert len({event.node for event in events}) == 3
    assert events == sorted(events, key=lambda event: (event.time_h, 0 if event.kind == "fault" else 1))


def test_variant_identity_and_worker_count(monkeypatch: pytest.MonkeyPatch) -> None:
    task = {
        "category": "pressure_range",
        "level": "20_30m",
        "baseline_level": "20_30m",
        "elevation_sd_m": 0.0,
        "fault_probability": 0.5,
        "pressure_min_m": 20.0,
        "pressure_max_m": 30.0,
        "scenario_id": 3,
    }
    row = v.add_variant_identity({"value": 1}, task, "normal", "policy", 2, 9)
    assert row["variant_category"] == "pressure_range"
    assert row["scenario_id"] == 3
    assert row["seed"] == 9
    monkeypatch.setattr(v.os, "cpu_count", lambda: 10)
    assert v.worker_count("auto") == 8
    assert v.worker_count("0") == 1


def _variant_rows() -> pd.DataFrame:
    rows = []
    levels = [("sd_0.0m", 90.0), ("sd_1.0m", 80.0)]
    for level, achievement in levels:
        for scenario_id in (1, 2):
            entries = [
                ("normal", "nsga2_generalized", achievement),
                ("normal", "original_near_far", achievement - 10.0),
                ("normal", "random_rotation", achievement - 15.0),
                ("dynamic_fault", "nsga2_dynamic_reoptimization", achievement - 5.0),
                ("dynamic_fault", "nsga2_static_no_reoptimization", achievement - 20.0),
            ]
            for experiment, policy, value in entries:
                rows.append(
                    {
                        "variant_category": "elevation_disturbance",
                        "variant_level": level,
                        "baseline_level": "sd_0.0m",
                        "scenario_id": scenario_id,
                        "experiment": experiment,
                        "policy": policy,
                        "achievement_pct": value + scenario_id,
                        "unmet_emitter_ratio_pct": 100.0 - value,
                        "average_deficit_m3_per_emitter": (100.0 - value) / 10.0,
                        "max_deficit_m3": 100.0 - value,
                        "pressure_uniformity_index_pct": value,
                        "feasible_group_ratio_pct": value,
                        "affected_mean_completion_pct": value,
                        "affected_target_met_pct": value,
                        "total_reoptimization_time_s": 1.0,
                        "mean_reoptimization_time_s": 0.5,
                    }
                )
    return pd.DataFrame(rows)


def test_bootstrap_and_wilcoxon_edge_cases() -> None:
    rng = np.random.default_rng(1)
    empty = v.bootstrap_mean_ci(pd.Series(dtype=float), rng)
    assert math.isnan(empty[0]) and math.isnan(empty[1])
    assert v.bootstrap_mean_ci(pd.Series([2.0]), rng) == (2.0, 2.0)
    low, high = v.bootstrap_mean_ci(pd.Series([1.0, 2.0, 3.0]), rng, n_boot=200)
    assert low <= 2.0 <= high
    assert v.safe_wilcoxon(pd.Series([], dtype=float), pd.Series([], dtype=float))[2] == 0
    assert v.safe_wilcoxon(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0])) == (0.0, 1.0, 2)


def test_aggregation_pairing_degradation_and_sensitivity() -> None:
    data = _variant_rows()
    summary = v.aggregate_metrics(data, seed=1)
    assert len(summary) == data.groupby(
        ["variant_category", "variant_level", "experiment", "policy"]
    ).ngroups
    assert "achievement_pct_ci95_low" in summary

    paired = v.paired_wilcoxon(data)
    assert not paired.empty
    assert paired["n_pairs"].min() == 2

    gain = v.dynamic_gain_table(data)
    assert set(gain["achievement_gain_pct_points"]) == {15.0}
    assert set(gain["unmet_ratio_reduction_pct_points"]) == {15.0}

    degradation = v.degradation_table(summary)
    disturbed = degradation[degradation["variant_level"].eq("sd_1.0m")]
    assert set(disturbed["achievement_delta_pct_points"]) == {-10.0}

    sensitivity = v.sensitivity_table(summary)
    achievement = sensitivity[sensitivity["metric"].eq("achievement_pct_mean")]
    assert not achievement.empty
    assert achievement["level_range"].max() == pytest.approx(10.0)


def test_variant_task_orchestrates_all_policies(
    monkeypatch: pytest.MonkeyPatch,
    experiment_network: m.Network,
) -> None:
    metrics_template = {
        "achievement_pct": 90.0,
        "unmet_emitter_ratio_pct": 10.0,
        "average_deficit_m3_per_emitter": 1.0,
        "max_deficit_m3": 2.0,
        "pressure_uniformity_index_pct": 95.0,
        "feasible_group_ratio_pct": 100.0,
        "affected_mean_completion_pct": 90.0,
        "affected_target_met_pct": 80.0,
        "total_reoptimization_time_s": 0.0,
        "mean_reoptimization_time_s": 0.0,
    }

    def copy_cache(target, source, prefix=""):
        for key, value in source.items():
            if key.startswith("cache_"):
                target[f"{prefix}{key}"] = value

    fake_base = SimpleNamespace(
        WORKER_NETWORK=experiment_network,
        Event=m.Event,
        original_plan=lambda _network, _params: [m.Group((0, 1, 2, 3), stage="original")],
        random_plan=lambda _network, _params, _rng: [m.Group((3, 2, 1, 0), stage="random")],
        evaluate_plan=lambda *_args, **_kwargs: (dict(metrics_template), np.zeros(120), []),
        optimize_nsga2_plan=lambda *_args, **_kwargs: (
            [m.Group((0, 1, 2, 3), stage="optimized")],
            {"cache_group_hits": 2},
            0.5,
            [{"generation": 0}],
        ),
        copy_cache_metrics=copy_cache,
        simulate_plan_with_events=lambda *_args, **_kwargs: (
            dict(metrics_template),
            [],
            [{"generation": 0}],
        ),
    )
    monkeypatch.setattr(v, "BASE_MODULE", fake_base)
    monkeypatch.setattr(v, "BASE_NETWORK", experiment_network)
    monkeypatch.setattr(v, "BASE_PARAMS", m.Params())
    task = {
        "seed": 1,
        "population": 4,
        "generations": 2,
        "random_repeats": 1,
        "elevation_sd_m": 0.0,
        "pressure_min_m": 20.0,
        "pressure_max_m": 30.0,
        "fault_probability": 0.0,
        "category": "elevation_disturbance",
        "level": "sd_0.0m",
        "baseline_level": "sd_0.0m",
        "scenario_id": 1,
    }
    rows, traces = v.run_variant_task(task)
    assert len(rows) == 5
    assert {row["policy"] for row in rows} == {
        "original_near_far",
        "random_rotation",
        "nsga2_generalized",
        "nsga2_static_no_reoptimization",
        "nsga2_dynamic_reoptimization",
    }
    assert len(traces) == 3
    assert all(row["variant_level"] == "sd_0.0m" for row in traces)


def test_create_figures_writes_variant_and_gain_charts(tmp_path: Path) -> None:
    data = _variant_rows()
    gain = v.dynamic_gain_table(data)
    v.create_figures(tmp_path, data, gain)
    figures = list((tmp_path / "figures").glob("*.png"))
    assert figures
    assert all(path.stat().st_size > 0 for path in figures)


def test_write_report_handles_populated_and_empty_sections(tmp_path: Path) -> None:
    data = _variant_rows()
    summary = v.aggregate_metrics(data, seed=1)
    degradation = v.degradation_table(summary)
    sensitivity = v.sensitivity_table(summary)
    gain = v.dynamic_gain_table(data)
    v.write_report(tmp_path, summary, degradation, sensitivity, gain, wall_time_s=120.0)
    report = (tmp_path / "variant_generalization_report.md").read_text(encoding="utf-8")
    assert "Total wall time: 2.00 min" in report
    assert "Dynamic Reoptimization Gain" in report
