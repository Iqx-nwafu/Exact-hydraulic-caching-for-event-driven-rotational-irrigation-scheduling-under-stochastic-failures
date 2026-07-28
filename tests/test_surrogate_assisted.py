from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import main_experiment_after_verification as m
import surrogate_assisted_nsga2 as s


class DummyPredictor:
    def __init__(self, values: np.ndarray):
        self.values = np.asarray(values, dtype=float)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.values.ndim == 1:
            return self.values
        return self.values[: len(x)]


class DummyRegressor:
    def fit(self, _x: np.ndarray, y: np.ndarray):
        self.mean_ = np.asarray(y, dtype=float).mean(axis=0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.tile(self.mean_, (len(x), 1))


def test_surrogate_bundle_normalizes_one_dimensional_prediction() -> None:
    bundle = s.SurrogateBundle(
        "dummy",
        DummyPredictor(np.array([1.0, 2.0, 3.0, 4.0])),
        [],
        s.TARGET_COLUMNS,
        0.5,
        0.0,
        1,
        1,
    )
    prediction = bundle.predict(np.zeros((1, 2)))
    assert prediction.shape == (1, 4)


def test_source_stats_and_node_branch_cover_boundaries() -> None:
    assert s.node_branch("J11") == 1
    assert s.node_branch("J106") == 10
    assert s.source_stats([], 0.0, 10.0) == (25.0, 25.0, 25.0, 25.0, 0.0)
    stats = s.source_stats([20.0, 22.0, 24.0], -1.0, 2.0)
    assert stats[0] == 20.0
    assert stats[1] == 21.0
    assert stats[2:4] == (20.0, 22.0)


def test_plan_features_match_declared_schema(experiment_network: m.Network) -> None:
    params = m.Params()
    plan = m.groups_from_order([0, 1, 2, 3, 4], params, "test")
    volumes = np.zeros(len(experiment_network.emitters))
    volumes[0] = params.target_per_emitter_m3
    features = s.plan_features(
        experiment_network,
        params,
        plan,
        [20.0, 21.0, 22.0],
        unavailable={1},
        start_time_h=1.0,
        start_volumes=volumes,
        horizon_h=3.0,
    )
    columns = s.feature_columns(len(experiment_network.emitters))
    assert len(features) == len(columns) == 4 * len(experiment_network.emitters) + 30
    positions = dict(zip(columns, features))
    assert positions["rank_1"] == -1.0
    assert positions["unavailable_1"] == 1.0
    assert positions["volume_fraction_0"] == 1.0
    assert positions["planned_emitter_ratio"] == pytest.approx(4 / 120)


def test_predicted_metrics_are_clipped_and_ranking_is_stable() -> None:
    metrics = s.predicted_metrics_from_targets(np.array([-10.0, -1.0, 120.0, 2.0]))
    assert metrics == {
        "achievement_pct": 100.0,
        "average_deficit_m3_per_emitter": 0.0,
        "pressure_uniformity_index_pct": 0.0,
        "feasible_group_ratio_pct": 0.0,
    }
    first = m.Individual(
        np.array([0.0]),
        cv=0.0,
        metrics={
            "achievement_pct": 90.0,
            "average_deficit_m3_per_emitter": 2.0,
            "pressure_uniformity_index_pct": 90.0,
        },
    )
    second = m.Individual(
        np.array([1.0]),
        cv=0.1,
        metrics={
            "achievement_pct": 100.0,
            "average_deficit_m3_per_emitter": 0.0,
            "pressure_uniformity_index_pct": 100.0,
        },
    )
    assert s.predicted_rank_indices([second, first]) == [1, 0]


def test_exact_plan_cache_returns_copies_of_cached_metrics() -> None:
    calls = 0

    class FakeBase:
        @staticmethod
        def plan_cache_key(plan):
            return tuple(plan)

        @staticmethod
        def plan_objectives(*_args):
            nonlocal calls
            calls += 1
            return (1.0, 2.0, 3.0), 0.0, {"achievement_pct": 99.0}

    cache: dict = {}
    stats: dict[str, int] = {}
    args = (
        FakeBase,
        object(),
        object(),
        [1, 2],
        [25.0],
        set(),
        0.0,
        None,
        120.0,
        cache,
        stats,
    )
    first = s.exact_plan_objectives_cached(*args)
    first[2]["changed"] = True
    second = s.exact_plan_objectives_cached(*args)
    assert calls == 1
    assert stats == {
        "exact_plan_cache_misses": 1,
        "exact_evaluations": 1,
        "exact_plan_cache_hits": 1,
    }
    assert "changed" not in second[2]


def test_surrogate_population_evaluation_refines_best_prediction(
    monkeypatch: pytest.MonkeyPatch,
    experiment_network: m.Network,
) -> None:
    base = SimpleNamespace(
        WORKER_NETWORK=experiment_network,
        keys_to_plan=m.keys_to_plan,
    )
    pop = [m.Individual(np.array([0.1, 0.2])), m.Individual(np.array([0.2, 0.1]))]
    bundle = s.SurrogateBundle(
        "dummy",
        DummyPredictor(
            np.array(
                [
                    [1.0, 1.0, 1.0, 0.0],
                    [2.0, 2.0, 2.0, 0.1],
                ]
            )
        ),
        s.feature_columns(len(experiment_network.emitters)),
        s.TARGET_COLUMNS,
        exact_fraction=0.0,
        exploration_fraction=0.0,
        min_exact=1,
        refine_generations=1,
    )

    selected: list[tuple[int, ...]] = []

    def fake_exact(_base, _network, _params, plan, *_args):
        selected.append(tuple(plan[0].emitters))
        return (0.5, 0.5, 0.5), 0.0, {
            "achievement_pct": 99.5,
            "average_deficit_m3_per_emitter": 0.5,
            "pressure_uniformity_index_pct": 99.5,
        }

    monkeypatch.setattr(s, "exact_plan_objectives_cached", fake_exact)
    stats: dict[str, int] = {}
    s.evaluate_population_surrogate(
        base,
        pop,
        [0, 1],
        m.Params(),
        [25.0],
        set(),
        0.0,
        None,
        120.0,
        bundle,
        np.random.default_rng(1),
        {},
        stats,
    )
    assert selected == [(0, 1)]
    assert stats["surrogate_predictions"] == 2
    assert stats["surrogate_exact_selected"] == 1
    assert pop[0].metrics["achievement_pct"] == 99.5
    assert pop[1].metrics["achievement_pct"] == 98.0


def test_unsupported_model_name_fails_fast() -> None:
    with pytest.raises(ValueError, match="Unsupported surrogate model"):
        s.model_factory("not-a-model", 1, 1)


def test_generate_training_dataset_writes_features_and_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    experiment_network: m.Network,
) -> None:
    fake_base = SimpleNamespace(
        Params=m.Params,
        load_network=lambda *_args: experiment_network,
        reset_worker_caches=lambda: None,
        plan_objectives=lambda *_args, **_kwargs: (
            (1.0, 2.0, 3.0),
            0.0,
            {
                "achievement_pct": 99.0,
                "average_deficit_m3_per_emitter": 2.0,
                "pressure_uniformity_index_pct": 97.0,
            },
        ),
    )
    context = {
        "context_type": "normal",
        "source_profile": [25.0] * 121,
        "start_time_h": 0.0,
        "start_volumes": np.zeros(len(experiment_network.emitters)),
        "unavailable": set(),
        "candidates": [0, 1, 2, 3],
    }
    monkeypatch.setattr(s, "BASE", fake_base)
    monkeypatch.setattr(s, "random_context", lambda *_args: dict(context))
    monkeypatch.setattr(
        s,
        "sample_plan",
        lambda _rng, candidates, params: m.groups_from_order(candidates, params, "training"),
    )

    def fake_to_parquet(self, path, index):
        assert not index
        Path(path).write_text("parquet stub", encoding="utf-8")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)
    x_df, y_df = s.generate_training_dataset(tmp_path, 2, 1, {"UPVC": 0.018, "PE": 0.020})
    assert x_df.shape == (2, 510)
    assert y_df["context_type"].tolist() == ["normal", "normal"]
    assert (tmp_path / "surrogate_training_features.parquet").exists()
    assert (tmp_path / "surrogate_training_labels.csv").exists()


def test_train_and_compare_surrogates_selects_and_serializes_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(s, "model_factory", lambda *_args: DummyRegressor())
    x_df = pd.DataFrame({"x1": np.arange(8, dtype=float), "x2": np.arange(8, dtype=float) / 2})
    y_df = pd.DataFrame(
        {
            target: np.arange(8, dtype=float) + idx
            for idx, target in enumerate(s.TARGET_COLUMNS)
        }
    )
    comparison, model_path, model_name = s.train_and_compare_surrogates(
        tmp_path,
        x_df,
        y_df,
        ["dummy"],
        seed=1,
        train_threads=1,
    )
    assert model_name == "dummy"
    assert model_path.exists()
    assert set(comparison["target"]) == set(s.TARGET_COLUMNS) | {"feasibility_misclassification"}
    metadata = json.loads((tmp_path / "best_surrogate_model.json").read_text(encoding="utf-8"))
    assert metadata["best_model"] == "dummy"


def test_exact_population_path_and_surrogate_optimizer_refinement(
    monkeypatch: pytest.MonkeyPatch,
    experiment_network: m.Network,
) -> None:
    base = SimpleNamespace(WORKER_NETWORK=experiment_network, keys_to_plan=m.keys_to_plan)
    exact_calls = 0

    def fake_exact(*_args):
        nonlocal exact_calls
        exact_calls += 1
        return (1.0, 1.0, 1.0), 0.0, {
            "achievement_pct": 99.0,
            "average_deficit_m3_per_emitter": 1.0,
            "pressure_uniformity_index_pct": 99.0,
        }

    monkeypatch.setattr(s, "exact_plan_objectives_cached", fake_exact)
    pop = [m.Individual(np.array([0.1, 0.2]))]
    s.evaluate_population_surrogate(
        base,
        pop,
        [0, 1],
        m.Params(),
        [25.0],
        set(),
        0.0,
        None,
        120.0,
        None,
        np.random.default_rng(1),
        {},
        {},
    )
    assert exact_calls == 1
    assert pop[0].objectives == (1.0, 1.0, 1.0)

    monkeypatch.setattr(m, "WORKER_NETWORK", experiment_network)

    def fake_evaluate(
        _base,
        evaluated,
        _candidates,
        _params,
        _profile,
        _unavailable,
        _start,
        _volumes,
        _horizon,
        _surrogate,
        _rng,
        _cache,
        stats,
        force_exact=False,
    ):
        for ind in evaluated:
            score = float(ind.x.sum())
            ind.objectives = (score, score, score)
            ind.cv = 0.0
            ind.metrics = {
                "achievement_pct": 100.0 - score,
                "average_deficit_m3_per_emitter": score,
                "pressure_uniformity_index_pct": 100.0 - score,
            }
        if force_exact:
            stats["exact_evaluations"] += len(evaluated)
        else:
            stats["surrogate_predictions"] += len(evaluated)

    monkeypatch.setattr(s, "evaluate_population_surrogate", fake_evaluate)
    monkeypatch.setattr(
        m,
        "optimize_nsga2_plan",
        lambda *_args, **_kwargs: (
            [m.Group((0, 1), stage="refined")],
            {
                "achievement_pct": 100.0,
                "average_deficit_m3_per_emitter": 0.0,
                "pressure_uniformity_index_pct": 100.0,
            },
            0.01,
            [{"generation": 0}],
        ),
    )
    bundle = s.SurrogateBundle(
        "dummy",
        DummyPredictor(np.zeros((4, 4))),
        [],
        s.TARGET_COLUMNS,
        0.5,
        0.0,
        1,
        1,
    )
    plan, metrics, elapsed, trace = s.optimize_surrogate_nsga2_plan(
        m,
        [0, 1, 2, 3],
        m.Params(),
        [25.0],
        seed=1,
        population=4,
        generations=1,
        surrogate=bundle,
        initial_orders=[[3, 2, 1, 0]],
    )
    assert plan[0].stage == "refined"
    assert metrics["surrogate_model"] == "dummy"
    assert metrics["surrogate_refinement_generations"] == 1
    assert trace[-1]["phase"] == "exact_refinement"
    assert elapsed >= 0.0
    assert s.optimize_surrogate_nsga2_plan(
        m, [], m.Params(), [25.0], 1, 4, 1, bundle
    ) == ([], {}, 0.0, [])


def test_paired_surrogate_comparison_uses_matching_scenarios(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    published = tmp_path / "published"
    published.mkdir()
    baseline = pd.DataFrame(
        [
            {
                "experiment": "normal",
                "scenario_id": 1,
                "repeat": 1,
                "policy": "nsga2_verified",
                "optimizer_algorithm": "nsga2",
                "achievement_pct": 90.0,
            },
            {
                "experiment": "single_fault",
                "scenario_id": 1,
                "repeat": 1,
                "policy": "nsga2_dynamic_reoptimization",
                "optimizer_algorithm": "nsga2",
                "achievement_pct": 80.0,
            },
        ]
    )
    baseline.to_csv(published / "all_policy_metrics_after_verification.csv", index=False)
    monkeypatch.setattr(s, "PUBLISHED_MAIN_DATA", published)
    surrogate = pd.DataFrame(
        [
            {
                "experiment": "normal",
                "scenario_id": 1,
                "repeat": 1,
                "policy": "surrogate_nsga2",
                "achievement_pct": 91.0,
            },
            {
                "experiment": "single_fault",
                "scenario_id": 1,
                "repeat": 1,
                "policy": "surrogate_nsga2_dynamic_reoptimization",
                "achievement_pct": 79.0,
            },
        ]
    )
    result = s.paired_surrogate_comparison(tmp_path, surrogate)
    assert len(result) == 2
    assert set(result["mean_difference_surrogate_minus_exact"]) == {-1.0, 1.0}
    assert (tmp_path / "surrogate_vs_exact_nsga2_paired_comparison.csv").exists()


def test_run_base_script_reads_reported_wall_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[list[str]] = []

    def fake_run(command, check):
        assert check
        captured.append(command)
        (tmp_path / "experiment_metadata_after_verification.json").write_text(
            json.dumps({"total_wall_time_s": 12.5}),
            encoding="utf-8",
        )

    monkeypatch.setattr(s.subprocess, "run", fake_run)
    elapsed = s.run_base_script(tmp_path, "2", True, 1, 2, 3, 4)
    assert elapsed == 12.5
    assert "--disable-cache" in captured[0]


def test_worker_count_uses_eighty_percent_of_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(s.os, "cpu_count", lambda: 5)
    assert s.worker_count("auto") == 4
    assert s.worker_count("-2") == 1
