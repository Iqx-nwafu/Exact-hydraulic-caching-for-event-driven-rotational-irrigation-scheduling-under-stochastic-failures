from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from pymoo.algorithms.moo.moead import MOEAD, default_decomp
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.core.problem import Problem
from pymoo.decomposition.pbi import PBI
from pymoo.indicators.hv import HV
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from scipy.spatial.distance import cdist

from revision_core import (
    CacheConfig,
    EvaluationCaches,
    Group,
    HydraulicParams,
    Network,
    cache_statistics,
    causal_expected_profile,
    evaluate_plan,
    keys_to_plan,
    objectives_from_metrics,
)


@dataclass(frozen=True)
class AlgorithmConfig:
    algorithm: str
    pop_size: int = 48
    crossover_probability: float = 0.9
    crossover_eta: float = 20.0
    mutation_probability_factor: float = 1.0
    mutation_eta: float = 20.0
    archive_size: int = 48
    rvea_alpha: float = 2.0
    rvea_adapt_frequency: float = 0.1
    moead_neighbors: int = 20
    moead_neighbor_probability: float = 0.9
    moead_pbi_theta: float = 5.0


@dataclass
class OptimizationResult:
    algorithm: str
    config: AlgorithmConfig
    plan: list[Group]
    selected_x: np.ndarray
    selected_metrics: dict[str, float]
    selected_objectives: np.ndarray
    selected_violation: float
    population_x: np.ndarray
    population_objectives: np.ndarray
    population_violations: np.ndarray
    nondominated_indices: np.ndarray
    elapsed_s: float
    function_evaluations: int
    exact_plan_evaluations: int
    cache_metrics: dict[str, float]
    convergence: list[dict[str, float]]


def normalized_objectives(raw_objectives: np.ndarray, violations: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(raw_objectives, dtype=float)
    scale = np.array([100.0, 187.2, 100.0], dtype=float)
    normalized = np.maximum(values / scale, 0.0)
    if violations is None:
        return normalized
    return np.column_stack([normalized, np.maximum(np.asarray(violations, dtype=float), 0.0)])


def nondominated_indices(objectives: np.ndarray) -> np.ndarray:
    values = np.asarray(objectives, dtype=float)
    keep = np.ones(len(values), dtype=bool)
    for index in range(len(values)):
        if not keep[index]:
            continue
        dominated = np.all(values <= values[index], axis=1) & np.any(values < values[index], axis=1)
        if np.any(dominated):
            keep[index] = False
    return np.flatnonzero(keep)


class SchedulingProblem(Problem):
    def __init__(
        self,
        network: Network,
        params: HydraulicParams,
        candidates: Sequence[int],
        source_profile: Sequence[float],
        unavailable: set[int],
        start_time_h: float,
        start_volumes: np.ndarray,
        horizon_h: float,
        caches: EvaluationCaches,
        soft_penalty_weight: float = 0.25,
    ) -> None:
        super().__init__(n_var=len(candidates), n_obj=3, n_ieq_constr=0, xl=0.0, xu=1.0)
        self.network = network
        self.params = params
        self.candidates = tuple(int(value) for value in candidates)
        self.source_profile = tuple(float(value) for value in source_profile)
        self.unavailable = set(unavailable)
        self.start_time_h = float(start_time_h)
        self.start_volumes = np.asarray(start_volumes, dtype=float).copy()
        self.horizon_h = float(horizon_h)
        self.caches = caches
        self.soft_penalty_weight = float(soft_penalty_weight)
        self.exact_plan_evaluations = 0

    def evaluate_keys(self, keys: np.ndarray) -> tuple[np.ndarray, float, dict[str, float]]:
        plan = keys_to_plan(keys, self.candidates, self.params)
        metrics, _, _ = evaluate_plan(
            self.network,
            self.params,
            plan,
            self.source_profile,
            unavailable=self.unavailable,
            start_time_h=self.start_time_h,
            start_volumes=self.start_volumes,
            horizon_h=self.horizon_h,
            caches=self.caches,
        )
        self.exact_plan_evaluations += 1
        objectives, violation = objectives_from_metrics(metrics)
        return objectives, violation, metrics

    def _evaluate(self, x: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        raw_rows: list[np.ndarray] = []
        violations: list[float] = []
        search_rows: list[np.ndarray] = []
        penalty_scale = np.array([25.0, self.params.target_per_emitter_m3 * 0.25, 25.0])
        for keys in np.atleast_2d(x):
            raw, violation, _ = self.evaluate_keys(keys)
            raw_rows.append(raw)
            violations.append(violation)
            search_rows.append(raw + self.soft_penalty_weight * violation * penalty_scale)
        out["F"] = np.vstack(search_rows)
        out["raw_F"] = np.vstack(raw_rows)
        out["hydraulic_cv"] = np.asarray(violations, dtype=float)[:, None]


class ConstrainedAwareMOEAD(MOEAD):
    """MOEA/D with soft-violation fields retained by the common scheduling problem."""

    def _setup(self, problem: Problem, **kwargs: Any) -> None:
        if self.ref_dirs is None:
            raise ValueError("MOEA/D requires fixed reference directions")
        self.pop_size = len(self.ref_dirs)
        self.neighbors = np.argsort(cdist(self.ref_dirs, self.ref_dirs), axis=1)[:, : self.n_neighbors]
        if self.decomposition is None:
            self.decomposition = default_decomp(problem)


def build_algorithm(
    config: AlgorithmConfig,
    n_var: int,
    seed: int,
    initial_x: np.ndarray | None = None,
) -> Any:
    mutation_probability = min(config.mutation_probability_factor / max(n_var, 1), 1.0)
    crossover = SBX(prob=config.crossover_probability, eta=config.crossover_eta)
    mutation = PM(prob=1.0, prob_var=mutation_probability, eta=config.mutation_eta, at_least_once=True)
    sampling: Any = FloatRandomSampling() if initial_x is None else np.asarray(initial_x, dtype=float)
    name = config.algorithm.lower()
    if name == "nsga2":
        return NSGA2(
            pop_size=config.pop_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True,
        )
    if name == "spea2":
        # In pymoo SPEA2 the retained population is the external archive.
        return SPEA2(
            pop_size=config.archive_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True,
        )
    reference_count = config.pop_size
    reference_directions = get_reference_directions("energy", 3, reference_count, seed=seed)
    if name == "rvea":
        return RVEA(
            reference_directions,
            alpha=config.rvea_alpha,
            adapt_freq=config.rvea_adapt_frequency,
            pop_size=config.pop_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=False,
        )
    if name == "moead":
        return ConstrainedAwareMOEAD(
            ref_dirs=reference_directions,
            n_neighbors=min(config.moead_neighbors, reference_count),
            decomposition=PBI(theta=config.moead_pbi_theta),
            prob_neighbor_mating=config.moead_neighbor_probability,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
        )
    raise ValueError(f"unsupported algorithm: {config.algorithm}")


class ConvergenceRecorder:
    def __init__(self) -> None:
        self.rows: list[dict[str, float]] = []
        self.started = time.perf_counter()

    def __call__(self, algorithm: Any) -> None:
        raw = algorithm.pop.get("raw_F")
        violations = algorithm.pop.get("hydraulic_cv")
        if raw is None or violations is None:
            return
        raw = np.asarray(raw, dtype=float)
        violations = np.asarray(violations, dtype=float).reshape(-1)
        front = normalized_objectives(raw, violations)
        nd = nondominated_indices(front)
        hv = float(HV(ref_point=np.array([1.1, 1.1, 1.1, 1.1]))(front[nd]))
        self.rows.append(
            {
                "generation": float(algorithm.n_gen),
                "function_evaluations": float(algorithm.evaluator.n_eval),
                "elapsed_s": time.perf_counter() - self.started,
                "hypervolume_4d": hv,
                "nondominated_count": float(len(nd)),
                "best_achievement_loss": float(np.min(raw[:, 0])),
                "best_average_deficit": float(np.min(raw[:, 1])),
                "best_pressure_nonuniformity": float(np.min(raw[:, 2])),
                "minimum_hydraulic_cv": float(np.min(violations)),
            }
        )


def choose_operational_solution(
    problem: SchedulingProblem,
    population_x: np.ndarray,
) -> tuple[int, list[dict[str, float]], np.ndarray, np.ndarray]:
    metrics_rows: list[dict[str, float]] = []
    raw_rows: list[np.ndarray] = []
    violations: list[float] = []
    for keys in population_x:
        raw, violation, metrics = problem.evaluate_keys(keys)
        raw_rows.append(raw)
        violations.append(violation)
        metrics_rows.append(metrics)
    raw_matrix = np.vstack(raw_rows)
    violation_array = np.asarray(violations, dtype=float)
    order = sorted(
        range(len(population_x)),
        key=lambda index: (
            violation_array[index],
            -metrics_rows[index]["achievement_pct"],
            metrics_rows[index]["average_deficit_m3_per_emitter"],
            -metrics_rows[index]["pressure_uniformity_index_pct"],
        ),
    )
    return order[0], metrics_rows, raw_matrix, violation_array


def optimize_plan(
    network: Network,
    params: HydraulicParams,
    candidates: Sequence[int],
    source_profile: Sequence[float],
    unavailable: set[int],
    start_time_h: float,
    start_volumes: np.ndarray,
    horizon_h: float,
    config: AlgorithmConfig,
    seed: int,
    evaluation_budget: int,
    cache_level: int = 3,
    time_limit_s: float | None = None,
    external_caches: EvaluationCaches | None = None,
    initial_x: np.ndarray | None = None,
) -> OptimizationResult:
    if not candidates:
        raise ValueError("optimization requires at least one candidate")
    caches = external_caches or EvaluationCaches(CacheConfig(level=cache_level))
    if caches.config.level != cache_level:
        raise ValueError("external cache level does not match cache_level")
    problem = SchedulingProblem(
        network,
        params,
        candidates,
        source_profile,
        unavailable,
        start_time_h,
        start_volumes,
        horizon_h,
        caches,
    )
    algorithm = build_algorithm(config, len(candidates), seed, initial_x=initial_x)
    recorder = ConvergenceRecorder()
    pymoo_evaluation_budget = int(evaluation_budget)
    # pymoo RVEA converts n_eval to n_gen with an off-by-one convention; compensate to keep actual FE equal.
    if config.algorithm.lower() == "rvea":
        pymoo_evaluation_budget += config.pop_size
    termination: Any = ("n_eval", pymoo_evaluation_budget)
    if time_limit_s is not None:
        termination = ("time", f"00:00:{max(float(time_limit_s), 0.01):.2f}")
    started = time.perf_counter()
    result = minimize(
        problem,
        algorithm,
        termination=termination,
        seed=seed,
        callback=recorder,
        save_history=False,
        verbose=False,
        copy_algorithm=False,
    )
    elapsed = time.perf_counter() - started
    population_x = np.asarray(result.pop.get("X"), dtype=float)
    selected_index, metrics_rows, raw_matrix, violation_array = choose_operational_solution(problem, population_x)
    combined = normalized_objectives(raw_matrix, violation_array)
    nd = nondominated_indices(combined)
    selected_x = population_x[selected_index].copy()
    plan = keys_to_plan(selected_x, candidates, params)
    return OptimizationResult(
        algorithm=config.algorithm.lower(),
        config=config,
        plan=plan,
        selected_x=selected_x,
        selected_metrics=dict(metrics_rows[selected_index]),
        selected_objectives=raw_matrix[selected_index].copy(),
        selected_violation=float(violation_array[selected_index]),
        population_x=population_x,
        population_objectives=raw_matrix,
        population_violations=violation_array,
        nondominated_indices=nd,
        elapsed_s=elapsed,
        function_evaluations=int(result.algorithm.evaluator.n_eval),
        exact_plan_evaluations=problem.exact_plan_evaluations,
        cache_metrics=cache_statistics(caches),
        convergence=recorder.rows,
    )


def forecast_for_information_mode(
    realized_profile: Sequence[float],
    decision_time_h: float,
    mode: str,
) -> list[float]:
    normalized_mode = mode.lower()
    if normalized_mode == "oracle":
        return [float(value) for value in realized_profile]
    if normalized_mode == "causal_mean":
        return causal_expected_profile(realized_profile, decision_time_h, expected_head_m=25.0)
    if normalized_mode == "causal_persistence":
        current = float(realized_profile[min(int(math.floor(decision_time_h)), len(realized_profile) - 1)])
        profile = causal_expected_profile(realized_profile, decision_time_h, expected_head_m=current)
        return profile
    raise ValueError(f"unsupported information mode: {mode}")
