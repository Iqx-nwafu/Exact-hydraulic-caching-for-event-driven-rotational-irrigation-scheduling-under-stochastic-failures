# Revised experimental design

## Scope

This revision package replaces the pre-verification numerical results. It does not modify the working manuscript. All formal parallel scenario calculations use exactly 20 worker processes on the same workstation.

## Hydraulic model

- Darcy-Weisbach head loss uses the same fixed roughness inputs and Colebrook-White friction-factor calculation used by the EPANET/WNTR reference.
- Pressure-dependent emitter discharge is solved as a closed fixed-point problem. Pipe flows, head losses, node heads, and emitter discharges are recomputed until both flow and head tolerances are satisfied.
- Source head is integrated in hourly subsegments. A four-hour irrigation group therefore uses four realized source-head observations during execution.
- Local loss is excluded from the EPANET equivalence check. The separate 1.1 engineering multiplier remains an optional engineering correction, not part of the friction-only validation.

## Information set

The deployable optimizer is causal. At each decision time it receives:

- observed source heads up to and including the current hour;
- current failed and recovered components;
- cumulative delivered volume for every emitter;
- the fixed 120 h horizon and known network model.

It does not receive future source heads, future fault times, or future recovery times. Its base forecast sets unobserved future source head to 25 m. Oracle and rolling-prediction variants are evaluated only as explicitly labeled controls.

## Main experiments

- Normal operation: 100 source-head scenarios; 50 random rotations and 5 independent tuned NSGA-II runs per scenario; original near-to-far schedule as the deterministic baseline.
- Single fault and recovery: 100 paired scenarios.
- Multiple faults and recoveries: 500 paired scenarios, with up to three concurrent or sequential faults and partial or absent recovery.
- Algorithms: canonical NSGA-II, SPEA2, RVEA, and MOEA/D implementations.
- All algorithm comparisons use the same scenarios and matched evaluation budgets.

## Algorithm tuning and quality

- Each algorithm is tuned on a predefined parameter grid over normal, fault, and recovery contexts.
- Parameter selection uses tuning contexts only. Held-out quality tests use 30 new contexts, 20 independent runs per algorithm and context, and 2,400 function evaluations per run.
- Pareto quality includes 4D hypervolume, IGD+, additive epsilon, nondominated-set size, spacing, convergence traces, equal-function-evaluation comparison, and equal-wall-time comparison.

## Cache ablation

- Level 0: no cache.
- Level 1: pipe/Colebrook cache.
- Level 2: level 1 plus converged irrigation-group PDA cache.
- Level 3: levels 1 and 2 plus complete-plan cache.
- States: cold; warm from a different optimizer seed in the same context; exact hot replay with identical scenario, seed, budget, and task.
- Timing tasks, random seeds, and evaluation budgets are paired across all levels. Peak RSS and cache storage estimates are reported.
- Scalability uses 60, 120, 240, and 480-emitter synthetic tree networks.

## Surrogate leakage control

The legacy 6,000-row data set used a shuffled row split and did not preserve scenario IDs, topology IDs, or plan hashes. It cannot support a scenario-independent or topology-external generalization claim.

The replacement data set contains:

- 30,000 development samples: 1,500 scenarios with 20 candidate plans per scenario;
- scenario-grouped 70/15/15 train, validation, and internal-test splits;
- 12,000 external samples from three fully held-out network topologies;
- exact plan hashes and plan assignments for duplicate and near-duplicate audits.

CatBoost, LightGBM, and XGBoost are tuned using grouped validation scenarios only. Internal and external test sets are not used for parameter selection. In surrogate-assisted NSGA-II, the surrogate screens candidates; final Pareto points and all reported agricultural metrics are recomputed by the exact closed PDA evaluator.

## Statistical unit

Optimizer repeats are first aggregated within each scenario. Confidence intervals and inferential tests then use scenario/context as the independent unit. Reported analyses include mean, standard deviation, 95% scenario bootstrap confidence interval, median, best and worst cases, paired Wilcoxon tests with Holm correction, and Friedman global tests for multi-algorithm comparisons.

## Figure and table limits

The planned main text contains at most nine figures and six tables. Additional parameter grids, per-metric statistical tests, detailed Pareto fronts, and secondary sensitivity outputs are assigned to supplementary material.
