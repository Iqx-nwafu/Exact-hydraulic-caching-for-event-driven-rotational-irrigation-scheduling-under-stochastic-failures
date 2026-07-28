from __future__ import annotations

import math
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import main_experiment_after_verification as m


def test_dataclass_properties_and_identifier_helpers() -> None:
    params = m.Params()
    assert params.target_per_emitter_m3 == pytest.approx(187.2)
    assert m.Individual(np.array([0.0]), cv=0.0).feasible
    assert not m.Individual(np.array([0.0]), cv=1e-3).feasible
    assert m.node_key("J9") < m.node_key("J10")
    assert m.branch_position("J106") == (10, 6)


def test_published_network_loads_with_expected_paths(experiment_network: m.Network) -> None:
    assert len(experiment_network.emitters) == 120
    assert len(experiment_network.pipe_ids) == 70
    assert experiment_network.node_to_path["J11"] == (0, 10)
    assert experiment_network.emitter_index["J106R"] == 119


def _minimal_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.DataFrame(
        {
            "Node": ["J0", "J1", "J11"],
            "X": [0.0, 1.0, 2.0],
            "Y": [0.0, 0.0, 0.0],
            "Z": [0.0, 0.0, 1.0],
        }
    )
    pipes = pd.DataFrame(
        {
            "Pipe ID": ["P0", "P1"],
            "FromNode": ["J0", "J1"],
            "ToNode": ["J1", "J11"],
            "Length_m": [10.0, 20.0],
            "Diameter_m": [0.2, 0.1],
            "Material": ["UPVC", "PE"],
        }
    )
    return nodes, pipes


@pytest.mark.parametrize(("duplicate", "message"), [("node", "Duplicated node IDs"), ("pipe", "Duplicated pipe IDs")])
def test_main_network_loader_rejects_duplicate_ids(
    monkeypatch: pytest.MonkeyPatch,
    duplicate: str,
    message: str,
) -> None:
    nodes, pipes = _minimal_tables()
    if duplicate == "node":
        nodes = pd.concat([nodes, nodes.iloc[[0]]], ignore_index=True)
    else:
        pipes = pd.concat([pipes, pipes.iloc[[0]]], ignore_index=True)
    tables = iter([nodes, pipes])
    monkeypatch.setattr(m.pd, "read_excel", lambda _path: next(tables))
    with pytest.raises(ValueError, match=message):
        m.load_network(Path("nodes.xlsx"), Path("pipes.xlsx"), {"UPVC": 0.018, "PE": 0.020})


def test_friction_and_source_profile_boundaries() -> None:
    assert m.colebrook_white_friction_factor(0.0, 2e-5, 0.1) == 0.0
    assert m.colebrook_white_friction_factor(1_000.0, 2e-5, 0.1) == pytest.approx(0.064)
    assert 0.01 < m.colebrook_white_friction_factor(100_000.0, 2e-5, 0.1) < 0.1
    profile = [20.0, 21.0, 22.0]
    assert m.source_head_at([], 10.0, 24.0) == 24.0
    assert m.source_head_at(profile, -1.0) == 20.0
    assert m.source_head_at(profile, 1.9) == 21.0
    assert m.source_head_at(profile, 99.0) == 22.0


def test_lru_cache_evicts_oldest_and_can_be_disabled() -> None:
    cache: OrderedDict[str, int] = OrderedDict()
    m.put_lru(cache, "a", 1, 2)
    m.put_lru(cache, "b", 2, 2)
    m.put_lru(cache, "c", 3, 2)
    assert list(cache) == ["b", "c"]
    m.put_lru(cache, "ignored", 4, 0)
    assert "ignored" not in cache


def test_hydraulic_caches_are_exact_and_report_hits(experiment_network: m.Network) -> None:
    params = m.Params(min_velocity_ms=0.0, max_velocity_ms=100.0)
    key = (0, 1, 2, 3)
    first = m.hydraulic_base(experiment_network, params, key)
    second = m.hydraulic_base(experiment_network, params, key)
    assert first is second
    assert m.WORKER_CACHE_STATS["group_cache_misses"] == 1
    assert m.WORKER_CACHE_STATS["group_cache_hits"] == 1
    assert m.WORKER_CACHE_STATS["pipe_loss_cache_misses"] > 0

    snapshot = m.cache_stats_snapshot()
    m.WORKER_CACHE_STATS["group_cache_hits"] += 2
    delta = m.cache_stats_delta(snapshot)
    assert delta["cache_group_hits"] == 2
    assert delta["cache_group_hit_rate_pct"] == 100.0


def test_hydraulic_group_handles_feasible_and_fully_unavailable_groups(
    experiment_network: m.Network,
) -> None:
    params = m.Params(design_head_m=1.0, min_velocity_ms=0.0, max_velocity_ms=100.0)
    emitters = (0, 1, 2, 3)
    result = m.hydraulic_group(experiment_network, params, emitters, set(), 30.0)
    assert result["active_emitters"] == list(emitters)
    assert result["feasible"]
    assert all(flow == pytest.approx(params.design_flow_m3s) for flow in result["delivered_flow"].values())

    unavailable = m.hydraulic_group(experiment_network, params, emitters, set(emitters), 30.0)
    assert unavailable["active_emitters"] == []
    assert not unavailable["feasible"]
    assert math.isnan(unavailable["min_pressure_m"])


def test_fault_isolation_and_plan_construction(experiment_network: m.Network) -> None:
    isolated = m.isolated_emitters_after_fault(experiment_network, "J13")
    assert len(isolated) == 8
    assert experiment_network.emitter_index["J13L"] in isolated
    assert experiment_network.emitter_index["J12L"] not in isolated

    params = m.Params(group_size=4)
    groups = m.groups_from_order(list(range(10)), params, "test")
    assert [len(group.emitters) for group in groups] == [4, 4, 2]
    assert all(group.stage == "test" for group in groups)
    keys = np.array([0.3, 0.1, 0.2])
    ordered = m.keys_to_plan(keys, [10, 11, 12], params, "keys")
    assert ordered[0].emitters == (11, 12, 10)
    assert m.plan_cache_key([m.Group((1, 2), 0.5)]) == (((1, 2), 1_800_000),)

    shuffled = m.random_plan(experiment_network, params, random.Random(7))
    flattened = [emitter for group in shuffled for emitter in group.emitters]
    assert sorted(flattened) == list(range(len(experiment_network.emitters)))


def test_evaluate_plan_truncates_at_horizon_and_collects_records(
    experiment_network: m.Network,
) -> None:
    params = m.Params(
        design_head_m=1.0,
        min_velocity_ms=0.0,
        max_velocity_ms=100.0,
        horizon_h=1.0,
    )
    plan = [m.Group((0, 1, 2, 3), duration_h=4.0, stage="test")]
    metrics, volumes, records = m.evaluate_plan(
        experiment_network,
        params,
        plan,
        [30.0],
        collect_records=True,
    )
    assert metrics["end_time_h"] == pytest.approx(1.0)
    assert metrics["executed_group_count"] == 1
    assert records[0]["duration_h"] == pytest.approx(1.0)
    np.testing.assert_allclose(volumes[:4], params.design_flow_m3s * 3600.0)
    np.testing.assert_allclose(volumes[4:], 0.0)


def test_volume_summary_caps_overdelivery(experiment_network: m.Network) -> None:
    params = m.Params()
    target = params.target_per_emitter_m3
    volumes = np.full(len(experiment_network.emitters), target)
    volumes[0] = target * 2.0
    volumes[1] = 0.0
    metrics = m.summarize_volumes(experiment_network, params, volumes, [10.0, 12.0], 1, 2)
    assert metrics["achievement_pct"] == pytest.approx(100.0 - 100.0 / len(volumes))
    assert metrics["unmet_emitters"] == 1
    assert metrics["max_deficit_m3"] == pytest.approx(target)
    assert metrics["feasible_group_ratio_pct"] == 50.0
    no_pressure = m.summarize_volumes(experiment_network, params, volumes, [], 0, 0)
    assert math.isnan(no_pressure["pressure_uniformity_index_pct"])


def _individual(
    objectives: tuple[float, float, float],
    cv: float = 0.0,
    achievement: float = 90.0,
) -> m.Individual:
    return m.Individual(
        np.array([0.0]),
        objectives=objectives,
        cv=cv,
        metrics={
            "achievement_pct": achievement,
            "average_deficit_m3_per_emitter": objectives[1],
            "pressure_uniformity_index_pct": 100.0 - objectives[2],
        },
    )


def test_dominance_sorting_crowding_and_selectors() -> None:
    best = _individual((1.0, 1.0, 1.0), achievement=99.0)
    tradeoff = _individual((0.5, 2.0, 2.0), achievement=98.0)
    dominated = _individual((3.0, 3.0, 3.0), achievement=90.0)
    infeasible = _individual((0.0, 0.0, 0.0), cv=0.5)
    assert m.dominates(best, dominated)
    assert m.dominates(best, infeasible)
    assert not m.dominates(infeasible, best)
    fronts = m.fast_non_dominated_sort([best, tradeoff, dominated, infeasible])
    assert set(fronts[0]) == {0, 1}
    m.assign_crowding([best, tradeoff], [0, 1])
    assert math.isinf(best.crowding) and math.isinf(tradeoff.crowding)

    population = [best, tradeoff, dominated, infeasible]
    for algorithm in ("nsga2", "spea2", "rvea", "moead"):
        selected = m.select_population(population.copy(), 2, algorithm)
        assert len(selected) == 2
        assert all(ind in population for ind in selected)
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        m.select_population(population, 2, "unknown")

    refs = m.reference_vectors_3d(2)
    assert refs.shape == (6, 3)
    np.testing.assert_allclose(np.linalg.norm(refs, axis=1), 1.0)


def test_population_evaluation_reuses_equivalent_plan(
    monkeypatch: pytest.MonkeyPatch,
    experiment_network: m.Network,
) -> None:
    monkeypatch.setattr(m, "WORKER_NETWORK", experiment_network)
    calls = 0

    def fake_objectives(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return (1.0, 2.0, 3.0), 0.0, {
            "achievement_pct": 99.0,
            "average_deficit_m3_per_emitter": 2.0,
            "pressure_uniformity_index_pct": 97.0,
        }

    monkeypatch.setattr(m, "plan_objectives", fake_objectives)
    pop = [m.Individual(np.array([0.1, 0.2])), m.Individual(np.array([0.1, 0.2]))]
    cache: OrderedDict = OrderedDict()
    m.evaluate_population(pop, [0, 1], m.Params(), [25.0], set(), 0.0, None, 120.0, cache)
    assert calls == 1
    assert m.WORKER_CACHE_STATS["plan_cache_misses"] == 1
    assert m.WORKER_CACHE_STATS["plan_cache_hits"] == 1
    assert pop[0].objectives == pop[1].objectives


def test_optimizer_smoke_with_deterministic_objective_stub(
    monkeypatch: pytest.MonkeyPatch,
    experiment_network: m.Network,
) -> None:
    monkeypatch.setattr(m, "WORKER_NETWORK", experiment_network)

    def fake_objectives(_network, _params, plan, *_args, **_kwargs):
        first = float(plan[0].emitters[0])
        return (first, first / 10.0, first / 100.0), 0.0, {
            "achievement_pct": 100.0 - first,
            "average_deficit_m3_per_emitter": first / 10.0,
            "pressure_uniformity_index_pct": 100.0 - first / 100.0,
        }

    monkeypatch.setattr(m, "plan_objectives", fake_objectives)
    plan, metrics, elapsed, trace = m.optimize_nsga2_plan(
        [0, 1, 2, 3],
        m.Params(),
        [25.0],
        seed=3,
        population=4,
        generations=1,
    )
    assert len(plan) == 1
    assert metrics["achievement_pct"] >= 97.0
    assert elapsed >= 0.0
    assert len(trace) == 1
    empty_plan, empty_metrics, empty_elapsed, empty_trace = m.optimize_nsga2_plan(
        [], m.Params(), [25.0], 1, 4, 1
    )
    assert (empty_plan, empty_metrics, empty_elapsed, empty_trace) == ([], {}, 0.0, [])


def test_event_processing_and_fault_union(experiment_network: m.Network) -> None:
    faults: set[str] = set()
    events = [
        m.Event(1.0, "fault", "J11"),
        m.Event(2.0, "fault", "J21"),
        m.Event(2.0, "recovery", "J11"),
    ]
    idx, changed = m.process_events_until(events, 0, 2.0, faults)
    assert idx == 3 and changed
    assert faults == {"J21"}
    unavailable = m.unavailable_from_faults(experiment_network, {"J11", "J21"})
    assert len(unavailable) == 24


def _perfect_hydraulic_group(_network, params, emitters, unavailable, _source_head):
    active = [emitter for emitter in emitters if emitter not in unavailable]
    return {
        "active_emitters": active,
        "pressure_values": [20.0] * len(active),
        "delivered_flow": {emitter: params.design_flow_m3s for emitter in active},
        "feasible": bool(active),
        "min_pressure_m": 20.0 if active else math.nan,
        "mean_pressure_m": 20.0 if active else math.nan,
        "pressure_violation_count": 0,
        "velocity_violation_count": 0,
        "max_velocity_ms": 1.0,
    }


def test_static_event_split_does_not_repeat_elapsed_group_time(
    monkeypatch: pytest.MonkeyPatch,
    experiment_network: m.Network,
) -> None:
    monkeypatch.setattr(m, "hydraulic_group", _perfect_hydraulic_group)
    params = m.Params(horizon_h=4.0, target_duration_h=4.0)
    initial_plan = [m.Group((0, 1, 2, 3), duration_h=4.0)]
    metrics, records, _ = m.simulate_plan_with_events(
        experiment_network,
        params,
        initial_plan,
        [25.0] * 5,
        [m.Event(1.0, "fault", "J11")],
        dynamic=False,
        optimizer_seed=1,
        population=4,
        generations=1,
        optimizer_algorithm="nsga2",
    )
    assert sum(row["duration_h"] for row in records) == pytest.approx(4.0)
    assert metrics["end_time_h"] == pytest.approx(4.0)
    assert initial_plan[0].duration_h == 4.0


def test_dynamic_event_triggers_reoptimization(
    monkeypatch: pytest.MonkeyPatch,
    experiment_network: m.Network,
) -> None:
    monkeypatch.setattr(m, "hydraulic_group", _perfect_hydraulic_group)
    calls = []

    def fake_optimize(*args, **kwargs):
        calls.append((args, kwargs))
        return [], {"cache_group_hits": 2}, 0.25, []

    monkeypatch.setattr(m, "optimize_nsga2_plan", fake_optimize)
    metrics, records, _ = m.simulate_plan_with_events(
        experiment_network,
        m.Params(horizon_h=4.0),
        [m.Group((0, 1, 2, 3), duration_h=4.0)],
        [25.0] * 5,
        [m.Event(1.0, "fault", "J11")],
        dynamic=True,
        optimizer_seed=1,
        population=4,
        generations=1,
        optimizer_algorithm="spea2",
    )
    assert len(calls) == 1
    assert len(records) == 1
    assert metrics["reoptimization_count"] == 1
    assert metrics["total_reoptimization_time_s"] == pytest.approx(0.25)
    assert metrics["cache_group_hits"] == 2


def test_scenarios_are_reproducible_and_ordered(experiment_network: m.Network) -> None:
    params = m.Params()
    profile = m.source_profile(42, params, 5)
    assert profile == m.source_profile(42, params, 5)
    assert all(params.source_head_min_m <= value <= params.source_head_max_m for value in profile)
    single = m.single_fault_scenario(42, experiment_network, params)
    assert single[0].kind == "fault"
    assert single[1].kind == "recovery"
    assert single[0].time_h < single[1].time_h <= params.horizon_h
    multi = m.multi_fault_scenario(42, experiment_network, params)
    assert multi == sorted(multi, key=lambda event: (event.time_h, 0 if event.kind == "fault" else 1))


def test_statistics_helpers_cover_empty_equal_and_paired_values() -> None:
    rng = np.random.default_rng(1)
    assert all(math.isnan(value) for value in m.bootstrap_mean_ci(pd.Series(dtype=float), rng))
    assert m.bootstrap_mean_ci(pd.Series([3.0]), rng) == (3.0, 3.0)
    low, high = m.bootstrap_mean_ci(pd.Series([1.0, 2.0, 3.0]), rng, n_boot=200)
    assert low <= 2.0 <= high
    assert m.safe_wilcoxon(pd.Series([], dtype=float), pd.Series([], dtype=float))[2] == 0
    assert m.safe_wilcoxon(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0])) == (0.0, 1.0, 2)
    stat, p_value, n = m.safe_wilcoxon(pd.Series([2.0, 3.0, 4.0]), pd.Series([1.0, 1.0, 1.0]))
    assert n == 3 and stat >= 0.0 and 0.0 <= p_value <= 1.0


def _policy_rows() -> pd.DataFrame:
    rows = []
    policies = {
        "normal": [
            ("nsga2_verified", "nsga2", 95.0),
            ("original_near_far", "none", 85.0),
            ("random_rotation", "none", 80.0),
        ],
        "single_fault": [
            ("nsga2_dynamic_reoptimization", "nsga2", 90.0),
            ("nsga2_static_no_reoptimization", "nsga2", 75.0),
            ("spea2_dynamic_reoptimization", "spea2", 88.0),
            ("spea2_static_no_reoptimization", "spea2", 74.0),
        ],
        "multi_fault": [
            ("nsga2_dynamic_reoptimization", "nsga2", 82.0),
            ("nsga2_static_no_reoptimization", "nsga2", 60.0),
        ],
    }
    for experiment, entries in policies.items():
        for scenario_id in (1, 2):
            for policy, algorithm, achievement in entries:
                rows.append(
                    {
                        "experiment": experiment,
                        "scenario_id": scenario_id,
                        "policy": policy,
                        "repeat": 1,
                        "optimizer_algorithm": algorithm,
                        "achievement_pct": achievement + scenario_id,
                        "unmet_emitter_ratio_pct": 100.0 - achievement,
                        "average_deficit_m3_per_emitter": (100.0 - achievement) / 10.0,
                        "max_deficit_m3": 100.0 - achievement,
                        "pressure_uniformity_index_pct": achievement,
                        "feasible_group_ratio_pct": achievement,
                        "affected_mean_completion_pct": achievement,
                        "affected_target_met_pct": achievement,
                        "total_reoptimization_time_s": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_aggregation_pairing_and_dynamic_gain_tables() -> None:
    df = _policy_rows()
    aggregate = m.aggregate_by_policy(df)
    assert len(aggregate) == df.groupby(["experiment", "policy"]).ngroups
    assert "achievement_pct_mean" in aggregate
    ci = m.agricultural_summary_with_ci(df, seed=1)
    assert {"metric", "bootstrap_ci95_low", "bootstrap_ci95_high"} <= set(ci.columns)
    paired = m.paired_wilcoxon_tests(df)
    assert not paired.empty
    assert paired["n_pairs"].min() == 2
    algorithms = m.paired_algorithm_wilcoxon_tests(df)
    assert set(algorithms["left_algorithm"]) == {"nsga2"}
    assert set(algorithms["right_algorithm"]) == {"spea2"}
    gain = m.paired_dynamic_value(df, "single_fault")
    assert set(gain["achievement_gain_pct_points"]) == {14.0, 15.0}


def test_worker_count_and_identity_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m.os, "cpu_count", lambda: 10)
    assert m.worker_count("auto") == 8
    assert m.worker_count("0") == 1
    row = m.add_identity({"value": 1}, "normal", 2, "policy", 3, 4)
    assert row["scenario_id"] == 2 and row["seed"] == 4
    target: dict[str, object] = {}
    m.copy_cache_metrics(target, {"cache_hits": 2, "other": 3}, prefix="initial_")
    assert target == {"initial_cache_hits": 2}
