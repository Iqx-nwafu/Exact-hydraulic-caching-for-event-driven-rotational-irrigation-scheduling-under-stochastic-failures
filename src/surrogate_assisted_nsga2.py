from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing
import os
import pickle
import platform
import random
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

from network_topology import add_network_args, file_sha256, network_cli_args, network_files, network_paths


warnings.filterwarnings("ignore", message="X does not have valid feature names")

ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = Path(__file__).with_name("main_experiment_after_verification.py")
NETWORK_DIR = ROOT / "data" / "network"
NETWORK_PATHS: tuple[str, str, str | None] = (str(NETWORK_DIR / "Nodes.xlsx"), str(NETWORK_DIR / "Pipes.xlsx"), None)
PUBLISHED_MAIN_DATA = ROOT / "data" / "main_experiments"
DATA_ROOT = ROOT / "outputs" / "surrogate"
TARGET_COLUMNS = ["obj_achievement_loss", "obj_average_deficit", "obj_pressure_nonuniformity", "constraint_violation"]
AGRICULTURAL_COLUMNS = [
    "achievement_pct",
    "unmet_emitter_ratio_pct",
    "average_deficit_m3_per_emitter",
    "max_deficit_m3",
    "pressure_uniformity_index_pct",
    "feasible_group_ratio_pct",
]


def load_base_module():
    spec = importlib.util.spec_from_file_location("verified_base_experiment", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()


@dataclass
class SurrogateBundle:
    name: str
    model: Any
    feature_columns: list[str]
    target_columns: list[str]
    exact_fraction: float
    exploration_fraction: float
    min_exact: int
    refine_generations: int

    def predict(self, x: np.ndarray) -> np.ndarray:
        y = np.asarray(self.model.predict(x), dtype=float)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        return y


WORKER_SURROGATE: SurrogateBundle | None = None
WORKER_BASE = None


def source_stats(source_profile: list[float], start_time_h: float, horizon_h: float) -> tuple[float, float, float, float, float]:
    start_idx = int(min(max(math.floor(start_time_h), 0), len(source_profile) - 1))
    end_idx = int(min(max(math.ceil(horizon_h), start_idx + 1), len(source_profile)))
    values = np.asarray(source_profile[start_idx:end_idx], dtype=float)
    if values.size == 0:
        values = np.asarray([25.0], dtype=float)
    return float(values[0]), float(values.mean()), float(values.min()), float(values.max()), float(values.std())


def plan_features(
    network,
    params,
    plan: list[Any],
    source_profile: list[float],
    unavailable: set[int] | None = None,
    start_time_h: float = 0.0,
    start_volumes: np.ndarray | None = None,
    horizon_h: float | None = None,
) -> np.ndarray:
    unavailable = set() if unavailable is None else set(unavailable)
    horizon = params.horizon_h if horizon_h is None else horizon_h
    n_emitters = len(network.emitters)
    ranks = np.full(n_emitters, -1.0, dtype=float)
    group_ids = np.full(n_emitters, -1.0, dtype=float)
    unavailable_mask = np.zeros(n_emitters, dtype=float)
    if unavailable:
        unavailable_mask[list(unavailable)] = 1.0
    volumes = np.zeros(n_emitters, dtype=float) if start_volumes is None else np.asarray(start_volumes, dtype=float)
    volume_fraction = np.clip(volumes / max(params.target_per_emitter_m3, 1e-12), 0.0, 2.0)

    flat: list[int] = []
    for group_idx, group in enumerate(plan):
        for emitter in group.emitters:
            if emitter not in unavailable and ranks[emitter] < 0:
                ranks[emitter] = len(flat)
                group_ids[emitter] = group_idx
                flat.append(emitter)
    if flat:
        denom = max(len(flat) - 1, 1)
        ranks[ranks >= 0] /= denom
        group_denom = max(len(plan) - 1, 1)
        group_ids[group_ids >= 0] /= group_denom

    branch_counts = np.zeros(len(network.laterals), dtype=float)
    for emitter in flat:
        branch_counts[network.node_lateral[network.emitter_to_node[emitter]]] += 1.0
    branch_counts /= lateral_emitter_counts(network)

    active_nodes = [network.emitter_to_node[e] for e in flat]
    z_values = np.asarray([network.node_z[n] for n in active_nodes], dtype=float) if active_nodes else np.asarray([0.0])
    path_lengths = (
        np.asarray([len(network.node_to_path[n]) for n in active_nodes], dtype=float) if active_nodes else np.asarray([0.0])
    )
    source_first, source_mean, source_min, source_max, source_std = source_stats(source_profile, start_time_h, horizon)
    n_groups = len(plan)
    last_group_size = len(plan[-1].emitters) if plan else 0
    scalar = np.asarray(
        [
            start_time_h / max(params.horizon_h, 1e-12),
            (horizon - start_time_h) / max(params.horizon_h, 1e-12),
            len(flat) / max(n_emitters, 1),
            len(unavailable) / max(n_emitters, 1),
            n_groups / 30.0,
            last_group_size / max(params.group_size, 1),
            source_first / 30.0,
            source_mean / 30.0,
            source_min / 30.0,
            source_max / 30.0,
            source_std / 10.0,
            float(z_values.mean()) / 30.0,
            float(z_values.std()) / 10.0,
            float(z_values.min()) / 30.0,
            float(z_values.max()) / 30.0,
            float(path_lengths.mean()) / 20.0,
            float(path_lengths.max()) / 20.0,
            float(volume_fraction.mean()),
            float(volume_fraction.max()),
            float(np.mean(volume_fraction >= 1.0)),
        ],
        dtype=float,
    )
    return np.concatenate([ranks, group_ids, unavailable_mask, volume_fraction, branch_counts, scalar])


def lateral_emitter_counts(network) -> np.ndarray:
    return np.bincount([network.node_lateral[n] for n in network.emitter_to_node], minlength=len(network.laterals)).astype(float)


def feature_columns(network) -> list[str]:
    n_emitters = len(network.emitters)
    cols = [f"rank_{i}" for i in range(n_emitters)]
    cols += [f"group_{i}" for i in range(n_emitters)]
    cols += [f"unavailable_{i}" for i in range(n_emitters)]
    cols += [f"volume_fraction_{i}" for i in range(n_emitters)]
    cols += [f"branch_count_{i}" for i in range(1, len(network.laterals) + 1)]
    cols += [
        "start_time_norm",
        "remaining_time_norm",
        "planned_emitter_ratio",
        "unavailable_ratio",
        "group_count_norm",
        "last_group_size_norm",
        "source_first_norm",
        "source_mean_norm",
        "source_min_norm",
        "source_max_norm",
        "source_std_norm",
        "z_mean_norm",
        "z_std_norm",
        "z_min_norm",
        "z_max_norm",
        "path_len_mean_norm",
        "path_len_max_norm",
        "volume_mean_fraction",
        "volume_max_fraction",
        "volume_met_ratio",
    ]
    return cols


def random_context(rng: random.Random, network, params, sample_id: int) -> dict[str, Any]:
    profile = BASE.source_profile(20260709 + sample_id * 17, params)
    start_time_h = 0.0
    start_volumes = np.zeros(len(network.emitters), dtype=float)
    unavailable: set[int] = set()
    context_type = rng.choices(["normal", "fault", "reopt"], weights=[0.45, 0.25, 0.30], k=1)[0]
    if context_type in {"fault", "reopt"}:
        n_faults = rng.randint(1, 3)
        for node in rng.sample(network.irrigation_nodes, k=n_faults):
            unavailable.update(BASE.isolated_emitters_after_fault(network, node))
    if context_type == "reopt":
        start_time_h = rng.uniform(8.0, 96.0)
        progress = rng.betavariate(1.3, 2.6)
        random_completion = np.asarray([rng.random() for _ in network.emitters], dtype=float)
        start_volumes = params.target_per_emitter_m3 * np.minimum(random_completion * progress * 1.4, 1.0)
        if unavailable:
            for emitter in unavailable:
                if rng.random() < 0.45:
                    start_volumes[emitter] = min(start_volumes[emitter], params.target_per_emitter_m3 * rng.random())
    candidates = [
        i
        for i in range(len(network.emitters))
        if i not in unavailable and start_volumes[i] < params.target_per_emitter_m3 - 1e-9
    ]
    if not candidates:
        candidates = [i for i in range(len(network.emitters)) if i not in unavailable]
    return {
        "context_type": context_type,
        "source_profile": profile,
        "start_time_h": start_time_h,
        "start_volumes": start_volumes,
        "unavailable": unavailable,
        "candidates": candidates,
    }


def sample_plan(rng: random.Random, candidates: list[int], params, stage: str = "training") -> list[Any]:
    order = list(candidates)
    rng.shuffle(order)
    return BASE.groups_from_order(order, params, stage)


def generate_training_dataset(
    output_dir: Path,
    samples: int,
    seed: int,
    roughness: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    params = BASE.Params()
    nodes, pipes, emitters = NETWORK_PATHS
    network = BASE.load_network(Path(nodes), Path(pipes) if pipes else None, roughness, Path(emitters) if emitters else None)
    BASE.WORKER_CACHE_ENABLED = True
    BASE.WORKER_GROUP_CACHE_MAX_ITEMS = 100000
    BASE.WORKER_PLAN_CACHE_MAX_ITEMS = 50000
    BASE.WORKER_PIPE_LOSS_CACHE_MAX_ITEMS = 50000
    BASE.reset_worker_caches()
    rng = random.Random(seed)
    x_rows: list[np.ndarray] = []
    y_rows: list[dict[str, float]] = []
    fcols = feature_columns(network)
    t0 = time.perf_counter()
    for sample_id in range(1, samples + 1):
        ctx = random_context(rng, network, params, sample_id)
        plan = sample_plan(rng, ctx["candidates"], params)
        objectives, cv, metrics = BASE.plan_objectives(
            network,
            params,
            plan,
            ctx["source_profile"],
            unavailable=ctx["unavailable"],
            start_time_h=ctx["start_time_h"],
            start_volumes=ctx["start_volumes"],
            horizon_h=params.horizon_h,
        )
        x_rows.append(
            plan_features(
                network,
                params,
                plan,
                ctx["source_profile"],
                unavailable=ctx["unavailable"],
                start_time_h=ctx["start_time_h"],
                start_volumes=ctx["start_volumes"],
                horizon_h=params.horizon_h,
            )
        )
        y_rows.append(
            {
                "obj_achievement_loss": objectives[0],
                "obj_average_deficit": objectives[1],
                "obj_pressure_nonuniformity": objectives[2],
                "constraint_violation": cv,
                **{col: float(metrics.get(col, math.nan)) for col in AGRICULTURAL_COLUMNS},
                "context_type": ctx["context_type"],
                "planned_emitters": len(ctx["candidates"]),
                "unavailable_emitters": len(ctx["unavailable"]),
                "start_time_h": ctx["start_time_h"],
            }
        )
        if sample_id % 1000 == 0 or sample_id == samples:
            elapsed = time.perf_counter() - t0
            print(f"Generated surrogate samples {sample_id}/{samples}; elapsed={elapsed:.1f}s", flush=True)
    x_df = pd.DataFrame(np.vstack(x_rows), columns=fcols)
    y_df = pd.DataFrame(y_rows)
    x_df.to_parquet(output_dir / "surrogate_training_features.parquet", index=False)
    y_df.to_csv(output_dir / "surrogate_training_labels.csv", index=False, encoding="utf-8-sig")
    return x_df, y_df


def model_factory(name: str, random_state: int, n_jobs: int) -> Any:
    lname = name.lower()
    if lname == "catboost":
        from catboost import CatBoostRegressor

        base_model = CatBoostRegressor(
            iterations=220,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            random_seed=random_state,
            verbose=False,
            thread_count=max(1, n_jobs),
            allow_writing_files=False,
        )
        return MultiOutputRegressor(base_model)
    if lname == "lightgbm":
        from lightgbm import LGBMRegressor

        base_model = LGBMRegressor(
            n_estimators=260,
            learning_rate=0.045,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=max(1, n_jobs),
            verbosity=-1,
        )
        return MultiOutputRegressor(base_model)
    if lname == "xgboost":
        from xgboost import XGBRegressor

        base_model = XGBRegressor(
            n_estimators=260,
            max_depth=6,
            learning_rate=0.045,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=max(1, n_jobs),
            objective="reg:squarederror",
            tree_method="hist",
            verbosity=0,
        )
        return MultiOutputRegressor(base_model)
    raise ValueError(f"Unsupported surrogate model: {name}")


def train_and_compare_surrogates(
    output_dir: Path,
    x_df: pd.DataFrame,
    y_df: pd.DataFrame,
    model_names: list[str],
    seed: int,
    train_threads: int,
) -> tuple[pd.DataFrame, Path, str]:
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    x = np.ascontiguousarray(x_df.to_numpy(dtype=np.float32))
    y = y_df[TARGET_COLUMNS].to_numpy(dtype=np.float32)
    train_idx, test_idx = train_test_split(np.arange(len(x)), test_size=0.25, random_state=seed, shuffle=True)
    rows: list[dict[str, Any]] = []
    best_name = ""
    best_score = math.inf
    best_path = model_dir / "none.pkl"
    for name in model_names:
        t0 = time.perf_counter()
        model = model_factory(name, seed, train_threads)
        model.fit(x[train_idx], y[train_idx])
        train_s = time.perf_counter() - t0
        pred = np.asarray(model.predict(x[test_idx]), dtype=float)
        pred[:, 3] = np.maximum(pred[:, 3], 0.0)
        true = y[test_idx]
        objective_rmse_sum = 0.0
        for col_idx, target in enumerate(TARGET_COLUMNS):
            mae = mean_absolute_error(true[:, col_idx], pred[:, col_idx])
            rmse = math.sqrt(mean_squared_error(true[:, col_idx], pred[:, col_idx]))
            objective_rmse_sum += rmse
            valid = np.abs(true[:, col_idx]) > 1e-6
            mape = (
                float(np.mean(np.abs((pred[valid, col_idx] - true[valid, col_idx]) / true[valid, col_idx])) * 100.0)
                if np.any(valid)
                else math.nan
            )
            smape = float(
                np.mean(
                    2.0
                    * np.abs(pred[:, col_idx] - true[:, col_idx])
                    / np.maximum(np.abs(pred[:, col_idx]) + np.abs(true[:, col_idx]), 1e-9)
                )
                * 100.0
            )
            rows.append(
                {
                    "model": name,
                    "target": target,
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "mape_pct": mape,
                    "smape_pct": smape,
                    "train_time_s": train_s,
                }
            )
        true_feasible = true[:, 3] <= 1e-9
        pred_feasible = pred[:, 3] <= 0.05
        false_feasible = np.mean(pred_feasible & ~true_feasible) * 100.0
        false_infeasible = np.mean((~pred_feasible) & true_feasible) * 100.0
        rows.append(
            {
                "model": name,
                "target": "feasibility_misclassification",
                "mae": float(np.mean(pred_feasible != true_feasible) * 100.0),
                "rmse": math.nan,
                "mape_pct": math.nan,
                "false_feasible_pct": float(false_feasible),
                "false_infeasible_pct": float(false_infeasible),
                "train_time_s": train_s,
            }
        )
        composite = objective_rmse_sum + 4.0 * false_feasible + 0.25 * train_s / 60.0
        model_path = model_dir / f"{name.lower()}_surrogate.pkl"
        bundle = SurrogateBundle(
            name=name.lower(),
            model=model,
            feature_columns=list(x_df.columns),
            target_columns=TARGET_COLUMNS,
            exact_fraction=0.45,
            exploration_fraction=0.08,
            min_exact=10,
            refine_generations=8,
        )
        with model_path.open("wb") as f:
            pickle.dump(bundle, f)
        if composite < best_score:
            best_score = composite
            best_name = name.lower()
            best_path = model_path
        print(f"Trained {name}; composite={composite:.4f}; train_time={train_s:.1f}s", flush=True)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "surrogate_model_prediction_comparison.csv", index=False, encoding="utf-8-sig")
    (output_dir / "best_surrogate_model.json").write_text(
        json.dumps({"best_model": best_name, "model_path": str(best_path), "selection_score": best_score}, indent=2),
        encoding="utf-8",
    )
    return comparison, best_path, best_name


def exact_plan_objectives_cached(
    base,
    network,
    params,
    plan: list[Any],
    source_profile: list[float],
    unavailable: set[int],
    start_time_h: float,
    start_volumes: np.ndarray | None,
    horizon_h: float,
    plan_cache: dict,
    stats: dict[str, int],
) -> tuple[tuple[float, float, float], float, dict[str, Any]]:
    key = base.plan_cache_key(plan)
    cached = plan_cache.get(key)
    if cached is not None:
        stats["exact_plan_cache_hits"] = stats.get("exact_plan_cache_hits", 0) + 1
        objectives, cv, metrics = cached
        return objectives, cv, dict(metrics)
    stats["exact_plan_cache_misses"] = stats.get("exact_plan_cache_misses", 0) + 1
    stats["exact_evaluations"] = stats.get("exact_evaluations", 0) + 1
    objectives, cv, metrics = base.plan_objectives(
        network,
        params,
        plan,
        source_profile,
        unavailable,
        start_time_h,
        start_volumes,
        horizon_h,
    )
    if len(plan_cache) > 30000:
        plan_cache.clear()
    plan_cache[key] = (objectives, cv, dict(metrics))
    return objectives, cv, metrics


def predicted_metrics_from_targets(y: np.ndarray) -> dict[str, Any]:
    return {
        "achievement_pct": float(np.clip(100.0 - y[0], 0.0, 100.0)),
        "average_deficit_m3_per_emitter": float(max(y[1], 0.0)),
        "pressure_uniformity_index_pct": float(np.clip(100.0 - y[2], 0.0, 100.0)),
        "feasible_group_ratio_pct": float(np.clip(100.0 * (1.0 - y[3]), 0.0, 100.0)),
    }


def predicted_rank_indices(pop: list[Any]) -> list[int]:
    def key(item: tuple[int, Any]) -> tuple[float, float, float, float]:
        idx, ind = item
        metrics = ind.metrics or {}
        return (
            ind.cv,
            -float(metrics.get("achievement_pct", -math.inf)),
            float(metrics.get("average_deficit_m3_per_emitter", math.inf)),
            -float(metrics.get("pressure_uniformity_index_pct", -math.inf)),
        )

    return [idx for idx, _ in sorted(enumerate(pop), key=key)]


def evaluate_population_surrogate(
    base,
    pop: list[Any],
    candidates: list[int],
    params,
    source_profile: list[float],
    unavailable: set[int],
    start_time_h: float,
    start_volumes: np.ndarray | None,
    horizon_h: float,
    surrogate: SurrogateBundle | None,
    rng: np.random.Generator,
    plan_cache: dict,
    stats: dict[str, int],
    force_exact: bool = False,
) -> None:
    network = base.WORKER_NETWORK
    assert network is not None
    plans = [base.keys_to_plan(ind.x, candidates, params, "surrogate_nsga2") for ind in pop]
    if surrogate is None or force_exact:
        for ind, plan in zip(pop, plans):
            objectives, cv, metrics = exact_plan_objectives_cached(
                base, network, params, plan, source_profile, unavailable, start_time_h, start_volumes, horizon_h, plan_cache, stats
            )
            ind.objectives = objectives
            ind.cv = cv
            ind.metrics = metrics
        return

    x = np.vstack(
        [
            plan_features(network, params, plan, source_profile, unavailable, start_time_h, start_volumes, horizon_h)
            for plan in plans
        ]
    ).astype(np.float32)
    pred = surrogate.predict(x)
    stats["surrogate_predictions"] = stats.get("surrogate_predictions", 0) + len(pop)
    for ind, y in zip(pop, pred):
        y = np.asarray(y, dtype=float)
        y[3] = max(float(y[3]), 0.0)
        ind.objectives = (float(max(y[0], 0.0)), float(max(y[1], 0.0)), float(max(y[2], 0.0)))
        ind.cv = float(y[3])
        ind.metrics = predicted_metrics_from_targets(y)

    ranked = predicted_rank_indices(pop)
    exact_count = max(surrogate.min_exact, int(math.ceil(len(pop) * surrogate.exact_fraction)))
    explore_count = int(math.ceil(len(pop) * surrogate.exploration_fraction))
    selected = set(ranked[: min(exact_count, len(pop))])
    if explore_count > 0 and len(pop) > len(selected):
        remaining = [i for i in range(len(pop)) if i not in selected]
        extra = rng.choice(remaining, size=min(explore_count, len(remaining)), replace=False)
        selected.update(int(i) for i in extra)
    for idx in selected:
        objectives, cv, metrics = exact_plan_objectives_cached(
            base,
            network,
            params,
            plans[idx],
            source_profile,
            unavailable,
            start_time_h,
            start_volumes,
            horizon_h,
            plan_cache,
            stats,
        )
        pop[idx].objectives = objectives
        pop[idx].cv = cv
        pop[idx].metrics = metrics
    stats["surrogate_exact_selected"] = stats.get("surrogate_exact_selected", 0) + len(selected)


def optimize_surrogate_nsga2_plan(
    base,
    candidates: list[int],
    params,
    source_profile: list[float],
    seed: int,
    population: int,
    generations: int,
    surrogate: SurrogateBundle | None,
    unavailable: set[int] | None = None,
    start_time_h: float = 0.0,
    start_volumes: np.ndarray | None = None,
    horizon_h: float | None = None,
    initial_orders: list[list[int]] | None = None,
) -> tuple[list[Any], dict[str, Any], float, list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    unavailable = set() if unavailable is None else set(unavailable)
    horizon = params.horizon_h if horizon_h is None else horizon_h
    dim = len(candidates)
    if dim == 0:
        return [], {}, 0.0, []
    t0 = time.perf_counter()
    plan_cache: dict = {}
    stats: dict[str, int] = {
        "exact_evaluations": 0,
        "surrogate_predictions": 0,
        "surrogate_exact_selected": 0,
        "exact_plan_cache_hits": 0,
        "exact_plan_cache_misses": 0,
    }
    pop: list[Any] = []
    initial_orders = initial_orders or []
    for order in initial_orders:
        rank = {em: i for i, em in enumerate(order) if em in candidates}
        keys = np.array([(rank.get(em, dim) + rng.random() * 0.01) / max(dim, 1) for em in candidates], dtype=float)
        pop.append(base.Individual(keys))
    while len(pop) < population:
        pop.append(base.Individual(rng.random(dim)))
    evaluate_population_surrogate(
        base,
        pop,
        candidates,
        params,
        source_profile,
        unavailable,
        start_time_h,
        start_volumes,
        horizon,
        surrogate,
        rng,
        plan_cache,
        stats,
        force_exact=False,
    )
    trace: list[dict[str, Any]] = []
    for gen in range(generations):
        fronts = base.fast_non_dominated_sort(pop)
        for front in fronts:
            base.assign_crowding(pop, front)
        offspring: list[Any] = []
        while len(offspring) < population:
            a, b = rng.choice(pop, size=2, replace=False)
            parent1 = a if (a.rank, -a.crowding) < (b.rank, -b.crowding) else b
            c, d = rng.choice(pop, size=2, replace=False)
            parent2 = c if (c.rank, -c.crowding) < (d.rank, -d.crowding) else d
            alpha = rng.random(dim)
            child_x = alpha * parent1.x + (1.0 - alpha) * parent2.x
            mutation_mask = rng.random(dim) < 0.08
            child_x[mutation_mask] += rng.normal(0.0, 0.12, size=int(mutation_mask.sum()))
            child_x = np.clip(child_x, 0.0, 1.0)
            offspring.append(base.Individual(child_x))
        evaluate_population_surrogate(
            base,
            offspring,
            candidates,
            params,
            source_profile,
            unavailable,
            start_time_h,
            start_volumes,
            horizon,
            surrogate,
            rng,
            plan_cache,
            stats,
            force_exact=False,
        )
        pop = base.select_population(pop + offspring, population, "nsga2")
        best = base.choose_solution(pop)
        trace.append(
            {
                "generation": gen,
                "algorithm": "surrogate_nsga2",
                "best_achievement_pct": best.metrics.get("achievement_pct", math.nan) if best.metrics else math.nan,
                "best_avg_deficit_m3": best.metrics.get("average_deficit_m3_per_emitter", math.nan) if best.metrics else math.nan,
                "best_pressure_uniformity_index_pct": best.metrics.get("pressure_uniformity_index_pct", math.nan) if best.metrics else math.nan,
                "best_cv": best.cv,
                "exact_evaluations": stats["exact_evaluations"],
                "surrogate_predictions": stats["surrogate_predictions"],
            }
        )
    evaluate_population_surrogate(
        base,
        pop,
        candidates,
        params,
        source_profile,
        unavailable,
        start_time_h,
        start_volumes,
        horizon,
        surrogate,
        rng,
        plan_cache,
        stats,
        force_exact=True,
    )
    best = base.choose_solution(pop)
    plan = base.keys_to_plan(best.x, candidates, params, "surrogate_nsga2")
    metrics = dict(best.metrics or {})
    refinement_elapsed = 0.0
    refinement_generations = int(surrogate.refine_generations) if surrogate is not None else 0
    refinement_reference_evaluations = 0
    if surrogate is not None and refinement_generations > 0:
        surrogate_order = [candidates[i] for i in np.argsort(best.x, kind="mergesort").tolist()]
        refined_plan, refined_metrics, refinement_elapsed, refinement_trace = base.optimize_nsga2_plan(
            candidates,
            params,
            source_profile,
            seed + 900001,
            population,
            refinement_generations,
            unavailable=unavailable,
            start_time_h=start_time_h,
            start_volumes=start_volumes,
            horizon_h=horizon,
            initial_orders=[surrogate_order],
            algorithm="nsga2",
        )
        plan = refined_plan
        metrics = dict(refined_metrics or metrics)
        refinement_reference_evaluations = population * (refinement_generations + 1)
        for row in refinement_trace:
            row.update({"phase": "exact_refinement"})
            trace.append(row)
    elapsed = time.perf_counter() - t0
    total_possible = population * (generations + 1)
    total_exact_equivalent = stats["exact_evaluations"] + refinement_reference_evaluations
    metrics.update(
        {
            "optimizer_algorithm": "surrogate_nsga2",
            "surrogate_model": surrogate.name if surrogate else "none",
            "surrogate_exact_evaluations": int(stats["exact_evaluations"]),
            "surrogate_predictions": int(stats["surrogate_predictions"]),
            "surrogate_exact_selected": int(stats["surrogate_exact_selected"]),
            "surrogate_exact_plan_cache_hits": int(stats["exact_plan_cache_hits"]),
            "surrogate_exact_plan_cache_misses": int(stats["exact_plan_cache_misses"]),
            "surrogate_reference_full_evaluations": int(total_possible),
            "surrogate_refinement_generations": refinement_generations,
            "surrogate_refinement_time_s": refinement_elapsed,
            "surrogate_refinement_reference_evaluations": refinement_reference_evaluations,
            "surrogate_total_exact_equivalent_evaluations": int(total_exact_equivalent),
            "surrogate_exact_reduction_pct": (1.0 - total_exact_equivalent / max(total_possible, 1)) * 100.0,
        }
    )
    return plan, metrics, elapsed, trace


def worker_init(
    nodes: str,
    pipes: str,
    params_dict: dict[str, Any],
    roughness: dict[str, float],
    surrogate_path: str | None,
    cache_enabled: bool,
    emitters: str | None = None,
) -> None:
    global WORKER_SURROGATE, WORKER_BASE
    WORKER_BASE = load_base_module()
    WORKER_BASE.init_worker(nodes, pipes, params_dict, roughness, cache_enabled, 100000, 50000, 50000, emitters)
    if surrogate_path:
        with Path(surrogate_path).open("rb") as f:
            WORKER_SURROGATE = pickle.load(f)
    else:
        WORKER_SURROGATE = None


def run_normal_surrogate_task(task: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = WORKER_BASE
    surrogate = WORKER_SURROGATE
    assert base is not None and base.WORKER_NETWORK is not None and base.WORKER_PARAMS is not None
    network, params = base.WORKER_NETWORK, base.WORKER_PARAMS
    sid = int(task["scenario_id"])
    seed = int(task["seed"])
    profile = base.source_profile(seed, params)
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    original = base.original_plan(network, params)
    metrics, _, _ = base.evaluate_plan(network, params, original, profile)
    metrics["optimizer_algorithm"] = "none"
    metrics["surrogate_model"] = "none"
    rows.append(base.add_identity(metrics, "normal", sid, "original_near_far", 0, seed))
    for repeat in range(1, int(task["random_repeats"]) + 1):
        plan = base.random_plan(network, params, random.Random(seed + repeat * 101))
        metrics, _, _ = base.evaluate_plan(network, params, plan, profile)
        metrics["optimizer_algorithm"] = "none"
        metrics["surrogate_model"] = "none"
        rows.append(base.add_identity(metrics, "normal", sid, "random_rotation", repeat, seed + repeat * 101))
    original_order = [e for g in original for e in g.emitters]
    for repeat in range(1, int(task["nsga_repeats"]) + 1):
        plan, metrics0, elapsed, trace = optimize_surrogate_nsga2_plan(
            base,
            list(range(len(network.emitters))),
            params,
            profile,
            seed + repeat * 1009,
            int(task["population"]),
            int(task["generations"]),
            surrogate,
            initial_orders=[original_order],
        )
        metrics, _, _ = base.evaluate_plan(network, params, plan, profile)
        metrics.update({k: v for k, v in metrics0.items() if k.startswith("surrogate_") or k in {"surrogate_model"}})
        metrics["total_reoptimization_time_s"] = elapsed
        metrics["mean_reoptimization_time_s"] = elapsed
        metrics["reoptimization_count"] = 1
        metrics["optimizer_algorithm"] = "surrogate_nsga2"
        rows.append(base.add_identity(metrics, "normal", sid, "surrogate_nsga2", repeat, seed + repeat * 1009))
        for row in trace:
            row.update({"experiment": "normal", "scenario_id": sid, "policy": "surrogate_nsga2", "repeat": repeat})
            traces.append(row)
    return rows, traces


def simulate_surrogate_plan_with_events(
    base,
    initial_plan: list[Any],
    source_profile: list[float],
    events: list[Any],
    dynamic: bool,
    optimizer_seed: int,
    population: int,
    generations: int,
    surrogate: SurrogateBundle | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    network, params = base.WORKER_NETWORK, base.WORKER_PARAMS
    assert network is not None and params is not None
    events = sorted(events, key=lambda e: (e.time_h, 0 if e.kind == "fault" else 1))
    volumes = np.zeros(len(network.emitters), dtype=float)
    active_faults: set[str] = set()
    event_idx = 0
    t = 0.0
    plan = initial_plan
    plan_idx = 0
    records: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    pressure_values: list[float] = []
    feasible_groups = 0
    executed_groups = 0
    reopt_count = 0
    reopt_time_s = 0.0
    surrogate_counts = {
        "surrogate_exact_evaluations": 0,
        "surrogate_predictions": 0,
        "surrogate_exact_selected": 0,
        "surrogate_reference_full_evaluations": 0,
        "surrogate_refinement_reference_evaluations": 0,
        "surrogate_total_exact_equivalent_evaluations": 0,
    }
    affected_emitters: set[int] = set()
    for event in events:
        if event.kind == "fault":
            affected_emitters.update(base.isolated_emitters_after_fault(network, event.node))

    while t < params.horizon_h - base.EPS:
        event_idx, event_changed = base.process_events_until(events, event_idx, t, active_faults)
        if dynamic and event_changed:
            plan_idx = len(plan)
        unavailable = base.unavailable_from_faults(network, active_faults)
        if dynamic and plan_idx >= len(plan):
            candidates = [
                i
                for i in range(len(network.emitters))
                if i not in unavailable and volumes[i] < params.target_per_emitter_m3 - 1e-9
            ]
            plan, opt_metrics, elapsed, trace = optimize_surrogate_nsga2_plan(
                base,
                candidates,
                params,
                source_profile,
                optimizer_seed + 1000 * (reopt_count + 1),
                population,
                max(10, generations // 2),
                surrogate,
                unavailable=unavailable,
                start_time_h=t,
                start_volumes=volumes,
                horizon_h=params.horizon_h,
            )
            for key in surrogate_counts:
                surrogate_counts[key] += int(opt_metrics.get(key, 0))
            for row in trace:
                row.update({"reopt_round": reopt_count + 1, "reopt_time_h": t, "optimizer_algorithm": "surrogate_nsga2"})
            traces.extend(trace)
            reopt_time_s += elapsed
            reopt_count += 1
            plan_idx = 0
            if not plan:
                break
        if plan_idx >= len(plan):
            break
        group = plan[plan_idx]
        next_event = events[event_idx] if event_idx < len(events) else None
        group_end = min(t + group.duration_h, params.horizon_h)
        if next_event and next_event.time_h < group_end - base.EPS and next_event.kind == "fault":
            duration = next_event.time_h - t
            stop_after = True
        else:
            duration = group_end - t
            stop_after = False
        if duration <= base.EPS:
            event_idx, event_changed = base.process_events_until(events, event_idx, t + base.EPS, active_faults)
            if dynamic and event_changed:
                plan_idx = len(plan)
            continue
        source_head = base.source_head_at(source_profile, t)
        result = base.hydraulic_group(network, params, group.emitters, unavailable, source_head)
        if result["active_emitters"]:
            executed_groups += 1
            feasible_groups += int(result["feasible"])
            pressure_values.extend(result["pressure_values"])
            for emitter, flow in result["delivered_flow"].items():
                volumes[emitter] += flow * duration * 3600.0
        records.append(
            {
                "start_h": t,
                "end_h": t + duration,
                "duration_h": duration,
                "stage": "dynamic" if dynamic else "static",
                "optimizer_algorithm": "surrogate_nsga2",
                "source_head_m": source_head,
                "active_faults": ";".join(sorted(active_faults)),
                "emitters": ";".join(network.emitters[e] for e in group.emitters),
                "active_emitters": ";".join(network.emitters[e] for e in result["active_emitters"]),
                "min_pressure_m": result["min_pressure_m"],
                "mean_pressure_m": result["mean_pressure_m"],
                "feasible": result["feasible"],
            }
        )
        t += duration
        if stop_after:
            event_idx, _ = base.process_events_until(events, event_idx, t, active_faults)
            if dynamic:
                plan_idx = len(plan)
            continue
        plan_idx += 1
        if next_event and next_event.time_h < group_end - base.EPS and next_event.kind == "recovery":
            event_idx, _ = base.process_events_until(events, event_idx, group_end, active_faults)
            if dynamic:
                plan_idx = len(plan)

    metrics = base.summarize_volumes(network, params, volumes, pressure_values, feasible_groups, executed_groups)
    affected = sorted(affected_emitters)
    if affected:
        target = params.target_per_emitter_m3
        affected_deficit = np.maximum(target - volumes[affected], 0.0)
        affected_completion = np.minimum(volumes[affected], target) / target
        metrics.update(
            {
                "affected_emitters": len(affected),
                "affected_target_met_pct": float((affected_deficit <= 1e-9).mean() * 100.0),
                "affected_mean_completion_pct": float(affected_completion.mean() * 100.0),
                "affected_average_deficit_m3": float(affected_deficit.mean()),
                "affected_max_deficit_m3": float(affected_deficit.max()),
            }
        )
    else:
        metrics.update(
            {
                "affected_emitters": 0,
                "affected_target_met_pct": math.nan,
                "affected_mean_completion_pct": math.nan,
                "affected_average_deficit_m3": math.nan,
                "affected_max_deficit_m3": math.nan,
            }
        )
    metrics.update(
        {
            "executed_group_count": executed_groups,
            "feasible_group_ratio_pct": feasible_groups / max(executed_groups, 1) * 100.0,
            "reoptimization_count": reopt_count,
            "total_reoptimization_time_s": reopt_time_s,
            "mean_reoptimization_time_s": reopt_time_s / max(reopt_count, 1),
            "end_time_h": t,
            "optimizer_algorithm": "surrogate_nsga2",
            "surrogate_model": surrogate.name if surrogate else "none",
            **surrogate_counts,
            "surrogate_exact_reduction_pct": (
                1.0
                - surrogate_counts["surrogate_total_exact_equivalent_evaluations"]
                / max(surrogate_counts["surrogate_reference_full_evaluations"], 1)
            )
            * 100.0,
        }
    )
    return metrics, records, traces


def run_fault_surrogate_task(task: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    base = WORKER_BASE
    surrogate = WORKER_SURROGATE
    assert base is not None and base.WORKER_NETWORK is not None and base.WORKER_PARAMS is not None
    network, params = base.WORKER_NETWORK, base.WORKER_PARAMS
    sid = int(task["scenario_id"])
    seed = int(task["seed"])
    experiment = str(task["experiment"])
    profile = base.source_profile(seed, params)
    events = base.single_fault_scenario(seed + 333, network, params) if experiment == "single_fault" else base.multi_fault_scenario(seed + 777, network, params)
    original = base.original_plan(network, params)
    original_order = [e for g in original for e in g.emitters]
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    base_plan, base_metrics, base_elapsed, trace = optimize_surrogate_nsga2_plan(
        base,
        list(range(len(network.emitters))),
        params,
        profile,
        seed + 1009,
        int(task["population"]),
        int(task["generations"]),
        surrogate,
        initial_orders=[original_order],
    )
    for row in trace:
        row.update(
            {
                "experiment": experiment,
                "scenario_id": sid,
                "policy": "surrogate_nsga2_initial",
                "repeat": 1,
                "optimizer_algorithm": "surrogate_nsga2",
            }
        )
        traces.append(row)
    for dynamic, suffix in [(False, "static_no_reoptimization"), (True, "dynamic_reoptimization")]:
        policy = f"surrogate_nsga2_{suffix}"
        metrics, recs, tr = simulate_surrogate_plan_with_events(
            base,
            base_plan,
            profile,
            events,
            dynamic,
            seed + (5000 if dynamic else 6000),
            int(task["population"]),
            int(task["generations"]),
            surrogate,
        )
        metrics["initial_optimization_time_s"] = base_elapsed
        for key, value in base_metrics.items():
            if key.startswith("surrogate_"):
                metrics[f"initial_{key}"] = value
        metrics["event_count"] = len(events)
        metrics["fault_count"] = sum(1 for e in events if e.kind == "fault")
        metrics["recovery_count"] = sum(1 for e in events if e.kind == "recovery")
        metrics["events"] = ";".join(f"{e.kind}:{e.node}@{e.time_h:.2f}" for e in events)
        metrics["optimizer_algorithm"] = "surrogate_nsga2"
        rows.append(base.add_identity(metrics, experiment, sid, policy, 1, seed))
        for rec in recs:
            rec.update({"experiment": experiment, "scenario_id": sid, "policy": policy, "optimizer_algorithm": "surrogate_nsga2"})
            records.append(rec)
        for row in tr:
            row.update({"experiment": experiment, "scenario_id": sid, "policy": policy, "repeat": 1, "optimizer_algorithm": "surrogate_nsga2"})
            traces.append(row)
    return rows, records, traces


def worker_count(value: str) -> int:
    cpu = os.cpu_count() or 1
    if value.lower() == "auto":
        return max(1, math.floor(cpu * 0.80))
    return max(1, int(value))


def run_surrogate_experiment(
    output_dir: Path,
    surrogate_path: Path,
    workers_value: str,
    base_seed: int,
    normal_scenarios: int,
    single_fault_scenarios: int,
    multi_fault_scenarios: int,
    random_repeats: int,
    nsga_repeats: int,
    population: int,
    generations: int,
    roughness: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    params = BASE.Params()
    workers = worker_count(workers_value)
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "surrogate_path": str(surrogate_path),
        "workers": workers,
        "cpu_count": os.cpu_count(),
        "normal_scenarios": normal_scenarios,
        "single_fault_scenarios": single_fault_scenarios,
        "multi_fault_scenarios": multi_fault_scenarios,
        "random_repeats": random_repeats,
        "nsga_repeats": nsga_repeats,
        "population": population,
        "generations": generations,
        "roughness_mm": roughness,
    }
    (output_dir / "surrogate_experiment_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    init_args = (
        NETWORK_PATHS[0],
        NETWORK_PATHS[1],
        params.__dict__,
        roughness,
        str(surrogate_path),
        True,
        NETWORK_PATHS[2],
    )
    normal_tasks = [
        {
            "scenario_id": i,
            "seed": base_seed + i * 100,
            "random_repeats": random_repeats,
            "nsga_repeats": nsga_repeats,
            "population": population,
            "generations": generations,
        }
        for i in range(1, normal_scenarios + 1)
    ]
    fault_tasks = [
        {
            "experiment": "single_fault",
            "scenario_id": i,
            "seed": base_seed + 100000 + i * 100,
            "population": population,
            "generations": generations,
        }
        for i in range(1, single_fault_scenarios + 1)
    ] + [
        {
            "experiment": "multi_fault",
            "scenario_id": i,
            "seed": base_seed + 200000 + i * 100,
            "population": population,
            "generations": generations,
        }
        for i in range(1, multi_fault_scenarios + 1)
    ]
    all_rows: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    all_traces: list[dict[str, Any]] = []
    start = time.perf_counter()
    print(f"Surrogate experiment workers: {workers}/{os.cpu_count() or 1}", flush=True)
    print(f"Normal tasks: {len(normal_tasks)}, fault tasks: {len(fault_tasks)}", flush=True)
    # Model training leaves OpenMP thread pools behind; forked workers deadlock on them, so spawn as on Windows.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=worker_init, initargs=init_args) as executor:
        futures = [executor.submit(run_normal_surrogate_task, task) for task in normal_tasks]
        futures.extend(executor.submit(run_fault_surrogate_task, task) for task in fault_tasks)
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            if len(result) == 2:
                rows, traces = result
                records = []
            else:
                rows, records, traces = result
            all_rows.extend(rows)
            all_records.extend(records)
            all_traces.extend(traces)
            completed += 1
            if completed == 1 or completed % 25 == 0 or completed == len(futures):
                elapsed = time.perf_counter() - start
                print(f"Completed {completed}/{len(futures)} surrogate tasks; elapsed={elapsed/60:.1f} min", flush=True)
    elapsed = time.perf_counter() - start
    metadata["total_wall_time_s"] = elapsed
    (output_dir / "surrogate_experiment_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_df = pd.DataFrame(all_rows)
    records_df = pd.DataFrame(all_records)
    traces_df = pd.DataFrame(all_traces)
    summary_input = BASE.canonical_row_order(metrics_df)
    summary_df = BASE.aggregate_by_policy(summary_input)
    metrics_df.to_csv(output_dir / "surrogate_policy_metrics.csv", index=False, encoding="utf-8-sig")
    metrics_df[metrics_df["experiment"].eq("normal")].to_csv(output_dir / "surrogate_normal_policy_metrics.csv", index=False, encoding="utf-8-sig")
    metrics_df[metrics_df["experiment"].eq("single_fault")].to_csv(output_dir / "surrogate_single_fault_policy_metrics.csv", index=False, encoding="utf-8-sig")
    metrics_df[metrics_df["experiment"].eq("multi_fault")].to_csv(output_dir / "surrogate_multi_fault_policy_metrics.csv", index=False, encoding="utf-8-sig")
    records_df.to_csv(output_dir / "surrogate_fault_execution_records.csv", index=False, encoding="utf-8-sig")
    traces_df.to_csv(output_dir / "surrogate_nsga2_convergence.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "surrogate_policy_summary.csv", index=False, encoding="utf-8-sig")
    return summary_input, summary_df


def paired_surrogate_comparison(output_dir: Path, surrogate_df: pd.DataFrame, baseline_dir: Path = PUBLISHED_MAIN_DATA) -> pd.DataFrame:
    baseline_path = baseline_dir / "all_policy_metrics_after_verification.csv"
    if not baseline_path.exists():
        return pd.DataFrame()
    baseline = pd.read_csv(baseline_path)
    pairs: list[dict[str, Any]] = []
    mapping = [
        ("normal", "surrogate_nsga2", "nsga2_verified"),
        ("single_fault", "surrogate_nsga2_static_no_reoptimization", "nsga2_static_no_reoptimization"),
        ("single_fault", "surrogate_nsga2_dynamic_reoptimization", "nsga2_dynamic_reoptimization"),
        ("multi_fault", "surrogate_nsga2_static_no_reoptimization", "nsga2_static_no_reoptimization"),
        ("multi_fault", "surrogate_nsga2_dynamic_reoptimization", "nsga2_dynamic_reoptimization"),
    ]
    for experiment, s_policy, b_policy in mapping:
        s = surrogate_df[surrogate_df["experiment"].eq(experiment) & surrogate_df["policy"].eq(s_policy)].copy()
        b = baseline[baseline["experiment"].eq(experiment) & baseline["policy"].eq(b_policy)].copy()
        if "optimizer_algorithm" in b.columns and experiment != "normal":
            b = b[b["optimizer_algorithm"].eq("nsga2")]
        if s.empty or b.empty:
            continue
        keys = ["experiment", "scenario_id"]
        if experiment == "normal":
            keys.append("repeat")
        merged = s.merge(b, on=keys, suffixes=("_surrogate", "_exact"))
        for metric in AGRICULTURAL_COLUMNS + ["total_reoptimization_time_s", "initial_optimization_time_s"]:
            left = f"{metric}_surrogate"
            right = f"{metric}_exact"
            if left not in merged.columns or right not in merged.columns:
                continue
            diff = pd.to_numeric(merged[left], errors="coerce") - pd.to_numeric(merged[right], errors="coerce")
            base_abs = np.maximum(np.abs(pd.to_numeric(merged[right], errors="coerce")), 1e-9)
            pairs.append(
                {
                    "experiment": experiment,
                    "surrogate_policy": s_policy,
                    "exact_policy": b_policy,
                    "metric": metric,
                    "n_pairs": int(diff.notna().sum()),
                    "surrogate_mean": float(pd.to_numeric(merged[left], errors="coerce").mean()),
                    "exact_mean": float(pd.to_numeric(merged[right], errors="coerce").mean()),
                    "mean_difference_surrogate_minus_exact": float(diff.mean()),
                    "max_abs_difference": float(diff.abs().max()),
                    "mean_abs_pct_difference": float((diff.abs() / base_abs).mean() * 100.0),
                    "max_abs_pct_difference": float((diff.abs() / base_abs).max() * 100.0),
                }
            )
    out = pd.DataFrame(pairs)
    out.to_csv(output_dir / "surrogate_vs_exact_nsga2_paired_comparison.csv", index=False, encoding="utf-8-sig")
    return out


def compare_against_existing_algorithms(output_dir: Path, surrogate_df: pd.DataFrame, baseline_dir: Path = PUBLISHED_MAIN_DATA) -> pd.DataFrame:
    baseline_path = baseline_dir / "all_policy_metrics_after_verification.csv"
    if not baseline_path.exists():
        return pd.DataFrame()
    baseline = pd.read_csv(baseline_path)
    combined = pd.concat([baseline, surrogate_df], ignore_index=True, sort=False)
    summary = BASE.aggregate_by_policy(combined)
    summary.to_csv(output_dir / "combined_existing_algorithms_and_surrogate_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def run_base_script(output_dir: Path, workers: str, disable_cache: bool, normal: int, single: int, multi: int, seed: int) -> float:
    cmd = [
        sys.executable,
        str(BASE_SCRIPT),
        "--output-dir",
        str(output_dir),
        "--workers",
        workers,
        "--normal-scenarios",
        str(normal),
        "--single-fault-scenarios",
        str(single),
        "--multi-fault-scenarios",
        str(multi),
        "--random-repeats",
        "5",
        "--nsga-repeats",
        "2",
        "--dynamic-algorithms",
        "nsga2",
        "--population",
        "48",
        "--generations",
        "30",
        "--base-seed",
        str(seed),
    ]
    cmd += network_cli_args(*NETWORK_PATHS)
    if disable_cache:
        cmd.append("--disable-cache")
    subprocess.run(cmd, check=True)
    metadata = json.loads((output_dir / "experiment_metadata_after_verification.json").read_text(encoding="utf-8"))
    return float(metadata["total_wall_time_s"])


def run_method_efficiency_benchmark(
    output_dir: Path,
    surrogate_path: Path,
    roughness: dict[str, float],
    seed: int,
) -> pd.DataFrame:
    bench_dir = output_dir / "method_efficiency_benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    serial_dir = bench_dir / "basic_nsga2_serial_no_cache"
    t = run_base_script(serial_dir, "1", True, 3, 3, 6, seed)
    rows.append({"method": "basic_nsga2_serial_no_cache", "normal": 3, "single_fault": 3, "multi_fault": 6, "task_count": 12, "wall_time_s": t})
    parallel_dir = bench_dir / "parallel_nsga2_no_cache"
    t = run_base_script(parallel_dir, "auto", True, 20, 20, 50, seed)
    rows.append({"method": "parallel_nsga2_no_cache", "normal": 20, "single_fault": 20, "multi_fault": 50, "task_count": 90, "wall_time_s": t})
    cache_dir = bench_dir / "cached_parallel_nsga2"
    t = run_base_script(cache_dir, "auto", False, 20, 20, 50, seed)
    rows.append({"method": "cached_parallel_nsga2", "normal": 20, "single_fault": 20, "multi_fault": 50, "task_count": 90, "wall_time_s": t})
    surrogate_dir = bench_dir / "surrogate_cached_nsga2"
    _, _ = run_surrogate_experiment(
        surrogate_dir,
        surrogate_path,
        "auto",
        seed,
        20,
        20,
        50,
        5,
        2,
        48,
        30,
        roughness,
    )
    metadata = json.loads((surrogate_dir / "surrogate_experiment_metadata.json").read_text(encoding="utf-8"))
    rows.append({"method": "surrogate_cached_nsga2", "normal": 20, "single_fault": 20, "multi_fault": 50, "task_count": 90, "wall_time_s": float(metadata["total_wall_time_s"])})
    out = pd.DataFrame(rows)
    out["seconds_per_task"] = out["wall_time_s"] / out["task_count"].clip(lower=1)
    ref = out[out["method"].eq("cached_parallel_nsga2")]["wall_time_s"]
    if not ref.empty:
        ref_value = float(ref.iloc[0])
        out["speedup_vs_cached_parallel_nsga2"] = ref_value / out["wall_time_s"]
    ref_task = out[out["method"].eq("cached_parallel_nsga2")]["seconds_per_task"]
    if not ref_task.empty:
        ref_task_value = float(ref_task.iloc[0])
        out["per_task_speedup_vs_cached_parallel_nsga2"] = ref_task_value / out["seconds_per_task"]
    out.to_csv(output_dir / "method_efficiency_benchmark_summary.csv", index=False, encoding="utf-8-sig")
    return out


def write_report(
    output_dir: Path,
    model_comparison: pd.DataFrame,
    best_name: str,
    surrogate_summary: pd.DataFrame,
    paired: pd.DataFrame,
    combined_summary: pd.DataFrame,
    efficiency: pd.DataFrame,
) -> None:
    lines = [
        "# Surrogate-Assisted NSGA-II Experiments",
        "",
        "## Best Surrogate",
        "",
        f"Selected model: `{best_name}`.",
        "",
        "## Surrogate Prediction Comparison",
        "",
        model_comparison.to_markdown(index=False) if not model_comparison.empty else "No model comparison.",
        "",
        "## Surrogate Policy Summary",
        "",
        surrogate_summary.to_markdown(index=False) if not surrogate_summary.empty else "No surrogate summary.",
        "",
        "## Surrogate vs Exact NSGA-II Paired Comparison",
        "",
        paired.to_markdown(index=False) if not paired.empty else "No paired comparison.",
        "",
        "## Existing Algorithms and Surrogate Combined Summary",
        "",
        combined_summary.to_markdown(index=False) if not combined_summary.empty else "No combined summary.",
        "",
        "## Method Efficiency Benchmark",
        "",
        efficiency.to_markdown(index=False) if not efficiency.empty else "No efficiency benchmark.",
    ]
    (output_dir / "surrogate_assisted_nsga2_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Surrogate-assisted NSGA-II workflow.")
    add_network_args(p, NETWORK_DIR)
    p.add_argument("--output-dir", type=Path, default=DATA_ROOT)
    p.add_argument("--base-seed", type=int, default=20260709)
    p.add_argument("--training-samples", type=int, default=6000)
    p.add_argument("--models", type=str, default="catboost,lightgbm,xgboost")
    p.add_argument("--train-threads", type=int, default=8)
    p.add_argument("--workers", type=str, default="auto")
    p.add_argument("--normal-scenarios", type=int, default=100)
    p.add_argument("--single-fault-scenarios", type=int, default=100)
    p.add_argument("--multi-fault-scenarios", type=int, default=500)
    p.add_argument("--random-repeats", type=int, default=50)
    p.add_argument("--nsga-repeats", type=int, default=5)
    p.add_argument("--population", type=int, default=48)
    p.add_argument("--generations", type=int, default=50)
    p.add_argument("--roughness-upvc-mm", type=float, default=0.018)
    p.add_argument("--roughness-pe-mm", type=float, default=0.020)
    p.add_argument("--skip-efficiency-benchmark", action="store_true")
    p.add_argument("--skip-final-experiment", action="store_true")
    p.add_argument("--baseline-main-dir", type=Path, default=PUBLISHED_MAIN_DATA,
                   help="Main-experiment results the surrogate runs are paired with; must come from the same hydraulic model.")
    return p.parse_args()


def main() -> None:
    global NETWORK_PATHS
    args = parse_args()
    NETWORK_PATHS = network_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    roughness = {"UPVC": args.roughness_upvc_mm, "PE": args.roughness_pe_mm}
    start = time.perf_counter()
    x_path = args.output_dir / "surrogate_training_features.parquet"
    y_path = args.output_dir / "surrogate_training_labels.csv"
    key_path = args.output_dir / "surrogate_training_key.json"
    # Reuse cached training data only if it was generated for the same network and sampling settings.
    training_key = {
        "network_sha256": [file_sha256(p) for p in network_files(*NETWORK_PATHS)],
        "roughness_mm": roughness,
        "training_samples": args.training_samples,
        "base_seed": args.base_seed,
    }
    cached_key = json.loads(key_path.read_text(encoding="utf-8")) if key_path.exists() else None
    if x_path.exists() and y_path.exists() and cached_key == training_key:
        x_df = pd.read_parquet(x_path)
        y_df = pd.read_csv(y_path)
        print(f"Loaded existing surrogate training data: {len(x_df)} samples", flush=True)
    else:
        x_df, y_df = generate_training_dataset(args.output_dir, args.training_samples, args.base_seed, roughness)
        key_path.write_text(json.dumps(training_key, indent=2), encoding="utf-8")
    model_names = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    model_comparison, best_path, best_name = train_and_compare_surrogates(
        args.output_dir, x_df, y_df, model_names, args.base_seed, args.train_threads
    )
    surrogate_summary = pd.DataFrame()
    paired = pd.DataFrame()
    combined_summary = pd.DataFrame()
    if not args.skip_final_experiment:
        surrogate_df, surrogate_summary = run_surrogate_experiment(
            args.output_dir / "surrogate_full_experiment",
            best_path,
            args.workers,
            args.base_seed,
            args.normal_scenarios,
            args.single_fault_scenarios,
            args.multi_fault_scenarios,
            args.random_repeats,
            args.nsga_repeats,
            args.population,
            args.generations,
            roughness,
        )
        paired = paired_surrogate_comparison(args.output_dir, surrogate_df, args.baseline_main_dir)
        combined_summary = compare_against_existing_algorithms(args.output_dir, surrogate_df, args.baseline_main_dir)
    efficiency = pd.DataFrame()
    if not args.skip_efficiency_benchmark:
        efficiency = run_method_efficiency_benchmark(args.output_dir, best_path, roughness, args.base_seed)
    write_report(args.output_dir, model_comparison, best_name, surrogate_summary, paired, combined_summary, efficiency)
    elapsed = time.perf_counter() - start
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_wall_time_s": elapsed,
        "best_surrogate_model": best_name,
        "best_surrogate_path": str(best_path),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "args": vars(args) | {"output_dir": str(args.output_dir)},
    }
    (args.output_dir / "surrogate_workflow_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Finished surrogate workflow in {elapsed/60:.2f} min. Output: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
