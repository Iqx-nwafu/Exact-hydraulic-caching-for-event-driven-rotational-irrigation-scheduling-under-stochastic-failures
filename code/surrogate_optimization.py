from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from generate_grouped_surrogate_dataset import feature_names, plan_features
from revision_core import CacheConfig, EvaluationCaches, Group, HydraulicParams, Network, cache_statistics, evaluate_plan, keys_to_plan, objectives_from_metrics
from revised_optimization import AlgorithmConfig, OptimizationResult, nondominated_indices, normalized_objectives, optimize_plan


@dataclass
class ExactIndividual:
    x: np.ndarray
    objectives: np.ndarray
    violation: float
    metrics: dict[str, float]


def search_objectives(objectives: np.ndarray, violations: np.ndarray) -> np.ndarray:
    penalty_scale = np.array([25.0, 46.8, 25.0], dtype=float)
    return objectives + 0.25 * violations[:, None] * penalty_scale


def crowding_distance(values: np.ndarray) -> np.ndarray:
    count, dimensions = values.shape
    if count <= 2:
        return np.full(count, np.inf)
    distance = np.zeros(count, dtype=float)
    for dimension in range(dimensions):
        order = np.argsort(values[:, dimension], kind="mergesort")
        distance[order[[0, -1]]] = np.inf
        span = values[order[-1], dimension] - values[order[0], dimension]
        if span <= 1.0e-15:
            continue
        distance[order[1:-1]] += (
            values[order[2:], dimension] - values[order[:-2], dimension]
        ) / span
    return distance


def nondominated_sort(values: np.ndarray) -> list[np.ndarray]:
    return [
        np.asarray(front, dtype=int)
        for front in NonDominatedSorting().do(np.asarray(values, dtype=float))
    ]


def survival_order(values: np.ndarray) -> np.ndarray:
    ordered: list[int] = []
    for front in nondominated_sort(values):
        crowding = crowding_distance(values[front])
        ordered.extend(front[np.argsort(-crowding, kind="mergesort")].tolist())
    return np.asarray(ordered, dtype=int)


def canonical_plan(plan: Sequence[Group]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(sorted(group.emitters)) for group in plan)


def exact_evaluate(
    network: Network,
    params: HydraulicParams,
    candidates: Sequence[int],
    source_profile: Sequence[float],
    unavailable: set[int],
    start_time_h: float,
    start_volumes: np.ndarray,
    horizon_h: float,
    x: np.ndarray,
    caches: EvaluationCaches,
) -> ExactIndividual:
    plan = keys_to_plan(x, candidates, params, "surrogate_assisted_nsga2")
    metrics, _, _ = evaluate_plan(
        network,
        params,
        plan,
        source_profile,
        unavailable=unavailable,
        start_time_h=start_time_h,
        start_volumes=start_volumes,
        horizon_h=horizon_h,
        caches=caches,
    )
    objectives, violation = objectives_from_metrics(metrics)
    return ExactIndividual(x.copy(), objectives, violation, metrics)


def offspring_population(parents: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    dimension = parents.shape[1]
    output = np.empty((count, dimension), dtype=float)
    for index in range(count):
        first, second = rng.integers(0, len(parents), size=2)
        weight = rng.random(dimension)
        child = weight * parents[first] + (1.0 - weight) * parents[second]
        mutation = rng.random(dimension) < (1.0 / max(dimension, 1))
        child[mutation] += rng.normal(0.0, 0.12, size=int(np.sum(mutation)))
        if rng.random() < 0.08:
            positions = rng.choice(dimension, size=min(2, dimension), replace=False)
            child[positions] = rng.random(len(positions))
        output[index] = np.clip(child, 0.0, 1.0)
    return output


def optimize_surrogate_assisted_nsga2(
    network: Network,
    params: HydraulicParams,
    candidates: Sequence[int],
    source_profile: Sequence[float],
    unavailable: set[int],
    start_time_h: float,
    start_volumes: np.ndarray,
    horizon_h: float,
    model: Any,
    seed: int,
    reference_evaluation_budget: int,
    population_size: int = 48,
    exact_fraction: float = 0.35,
    exploration_fraction: float = 0.20,
    exact_final_generations: int = 4,
    exact_refinement_budget: int | None = None,
    cache_level: int = 3,
) -> OptimizationResult:
    if not candidates:
        raise ValueError("optimization requires at least one candidate")
    if population_size < 8 or not 0.0 < exact_fraction <= 1.0:
        raise ValueError("invalid surrogate-assisted population settings")
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed + 91)
    caches = EvaluationCaches(CacheConfig(level=cache_level))
    context = {
        "profile": list(source_profile),
        "start_time_h": start_time_h,
        "volumes": np.asarray(start_volumes, dtype=float),
        "unavailable": set(unavailable),
        "candidates": list(candidates),
    }
    if exact_refinement_budget is None:
        exact_refinement_budget = max(population_size * 2, int(round(reference_evaluation_budget * 0.40)))
    exact_refinement_budget = int(max(0, min(exact_refinement_budget, reference_evaluation_budget - population_size)))
    screening_budget = reference_evaluation_budget - exact_refinement_budget
    generations = max(1, math.ceil(screening_budget / population_size) - 1)
    initial_x = rng.random((population_size, len(candidates)))
    archive: dict[tuple[tuple[int, ...], ...], ExactIndividual] = {}
    active_keys: dict[tuple[tuple[int, ...], ...], None] = {}
    exact_calls = 0
    surrogate_predictions = 0

    def add_exact(x_values: np.ndarray) -> None:
        nonlocal exact_calls
        plan = keys_to_plan(x_values, candidates, params, "surrogate_assisted_nsga2")
        key = canonical_plan(plan)
        if key in archive:
            active_keys[key] = None
            return
        archive[key] = exact_evaluate(
            network,
            params,
            candidates,
            source_profile,
            unavailable,
            start_time_h,
            start_volumes,
            horizon_h,
            x_values,
            caches,
        )
        active_keys[key] = None
        exact_calls += 1

    for x_values in initial_x:
        add_exact(x_values)

    convergence: list[dict[str, float]] = []
    for generation in range(1, generations + 1):
        current_keys = list(active_keys)
        individuals = [archive[key] for key in current_keys]
        raw = np.vstack([item.objectives for item in individuals])
        violations = np.asarray([item.violation for item in individuals])
        current_order = survival_order(search_objectives(raw, violations))[:population_size]
        parents = np.vstack([individuals[index].x for index in current_order])
        offspring = offspring_population(parents, population_size, rng)
        plans = [keys_to_plan(x_values, candidates, params, "surrogate_assisted_nsga2") for x_values in offspring]
        x_features = np.vstack([plan_features(network, params, plan, context) for plan in plans])
        prediction_frame = pd.DataFrame(
            x_features,
            columns=feature_names(len(network.emitters), params.group_size),
        )
        predicted = np.asarray(model.predict(prediction_frame), dtype=float)
        predicted[:, :3] = np.maximum(predicted[:, :3], 0.0)
        predicted[:, 3] = np.maximum(predicted[:, 3], 0.0)
        surrogate_predictions += len(offspring)
        predicted_order = survival_order(search_objectives(predicted[:, :3], predicted[:, 3]))
        if generation > generations - exact_final_generations:
            selected = predicted_order
        else:
            exact_count = max(8, int(math.ceil(population_size * exact_fraction)))
            explore_count = min(exact_count - 1, int(math.ceil(exact_count * exploration_fraction)))
            exploit_count = exact_count - explore_count
            selected_list = predicted_order[:exploit_count].tolist()
            remaining = [index for index in range(population_size) if index not in selected_list]
            if explore_count and remaining:
                selected_list.extend(py_rng.sample(remaining, k=min(explore_count, len(remaining))))
            selected = np.asarray(selected_list, dtype=int)
        for index in selected:
            add_exact(offspring[index])

        current_keys = list(active_keys)
        individuals = [archive[key] for key in current_keys]
        raw = np.vstack([item.objectives for item in individuals])
        violations = np.asarray([item.violation for item in individuals])
        if len(individuals) > population_size * 3:
            retained = survival_order(search_objectives(raw, violations))[: population_size * 3]
            active_keys = {current_keys[index]: None for index in retained}
            current_keys = list(active_keys)
            individuals = [archive[key] for key in current_keys]
            raw = np.vstack([item.objectives for item in individuals])
            violations = np.asarray([item.violation for item in individuals])
        normalized = normalized_objectives(raw, violations)
        nd = nondominated_indices(normalized)
        hv = float(HV(ref_point=np.array([1.1, 1.1, 1.1, 1.1]))(normalized[nd]))
        convergence.append(
            {
                "generation": float(generation),
                "function_evaluations": float(exact_calls),
                "reference_candidate_evaluations": float(population_size * (generation + 1)),
                "elapsed_s": time.perf_counter() - started,
                "hypervolume_4d": hv,
                "nondominated_count": float(len(nd)),
                "best_achievement_loss": float(np.min(raw[:, 0])),
                "best_average_deficit": float(np.min(raw[:, 1])),
                "best_pressure_nonuniformity": float(np.min(raw[:, 2])),
                "minimum_hydraulic_cv": float(np.min(violations)),
            }
        )

    individuals = [archive[key] for key in active_keys]
    raw = np.vstack([item.objectives for item in individuals])
    violations = np.asarray([item.violation for item in individuals])
    population_order = survival_order(search_objectives(raw, violations))[:population_size]
    final = [individuals[index] for index in population_order]
    population_x = np.vstack([item.x for item in final])
    population_objectives = np.vstack([item.objectives for item in final])
    population_violations = np.asarray([item.violation for item in final])
    selected_index = min(
        range(len(final)),
        key=lambda index: (
            final[index].violation,
            -final[index].metrics["achievement_pct"],
            final[index].metrics["average_deficit_m3_per_emitter"],
            -final[index].metrics["pressure_uniformity_index_pct"],
        ),
    )
    selected = final[selected_index]
    nd = nondominated_indices(normalized_objectives(population_objectives, population_violations))
    refinement_config = AlgorithmConfig(
        algorithm="nsga2",
        pop_size=population_size,
        crossover_probability=0.9,
        crossover_eta=15.0,
        mutation_probability_factor=1.0,
        mutation_eta=20.0,
    )
    refinement: OptimizationResult | None = None
    if exact_refinement_budget >= population_size:
        refinement = optimize_plan(
            network,
            params,
            candidates,
            source_profile,
            unavailable,
            start_time_h,
            start_volumes,
            horizon_h,
            refinement_config,
            seed + 7_000_001,
            exact_refinement_budget,
            cache_level=cache_level,
            external_caches=caches,
            initial_x=population_x,
        )
        selected = ExactIndividual(
            refinement.selected_x,
            refinement.selected_objectives,
            refinement.selected_violation,
            refinement.selected_metrics,
        )
        population_x = refinement.population_x
        population_objectives = refinement.population_objectives
        population_violations = refinement.population_violations
        nd = refinement.nondominated_indices
        for row in refinement.convergence:
            convergence.append(
                {
                    **row,
                    "generation": float(row["generation"] + generations),
                    "function_evaluations": float(row["function_evaluations"] + exact_calls),
                    "reference_candidate_evaluations": float(screening_budget + row["function_evaluations"]),
                    "elapsed_s": float(time.perf_counter() - started),
                }
            )
    refinement_evaluations = refinement.function_evaluations if refinement is not None else 0
    total_exact_evaluations = exact_calls + refinement_evaluations
    config = AlgorithmConfig(algorithm="surrogate_nsga2", pop_size=population_size)
    metrics = dict(selected.metrics)
    metrics.update(
        {
            "surrogate_predictions": float(surrogate_predictions),
            "surrogate_screening_exact_evaluations": float(exact_calls),
            "surrogate_refinement_exact_evaluations": float(refinement_evaluations),
            "surrogate_exact_evaluations": float(total_exact_evaluations),
            "surrogate_reference_evaluations": float(reference_evaluation_budget),
            "surrogate_exact_reduction_pct": (
                1.0 - total_exact_evaluations / max(reference_evaluation_budget, 1)
            )
            * 100.0,
        }
    )
    return OptimizationResult(
        algorithm="surrogate_nsga2",
        config=config,
        plan=keys_to_plan(selected.x, candidates, params, "surrogate_assisted_nsga2"),
        selected_x=selected.x.copy(),
        selected_metrics=metrics,
        selected_objectives=selected.objectives.copy(),
        selected_violation=float(selected.violation),
        population_x=population_x,
        population_objectives=population_objectives,
        population_violations=population_violations,
        nondominated_indices=nd,
        elapsed_s=time.perf_counter() - started,
        function_evaluations=total_exact_evaluations,
        exact_plan_evaluations=total_exact_evaluations,
        cache_metrics=cache_statistics(caches),
        convergence=convergence,
    )
