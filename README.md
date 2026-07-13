# Exact hydraulic caching for event-driven rotational irrigation scheduling

This repository contains the code and numerical data used to evaluate event-driven rotational irrigation scheduling under stochastic pressure, pipe-node failures, and partial or complete recovery. It includes only reproducibility materials: source code, network inputs, EPANET/WNTR verification outputs, optimization results, cache benchmarks, surrogate benchmarks, and network-variant sensitivity results. Manuscript files and generated figures are intentionally excluded.

## Repository structure

```text
src/
  hydraulic_check.py                       EPANET/WNTR hydraulic verification
  main_experiment_after_verification.py    verified multi-scenario optimization experiments
  nsga2_cache_benchmark.py                 exact-cache timing and equivalence benchmark
  surrogate_assisted_nsga2.py              CatBoost/LightGBM/XGBoost comparison
  variant_generalization_analysis.py       pressure, failure, and elevation sensitivity
data/
  network/                                  Nodes.xlsx and Pipes.xlsx
  hydraulic_verification/                   100-case verification and EPANET INP files
  main_experiments/                         normal, single-fault, and multi-fault results
  surrogate/                                surrogate and solver-efficiency comparisons
  variant_network/                          network-variant and sensitivity results
```

## Model and experiment scope

- Darcy-Weisbach friction losses use Reynolds-number-dependent friction factors and material roughness values consistent with the EPANET/WNTR verification model.
- EPANET comparison uses friction losses only. The optimization model applies the separate engineering correction `h_total = 1.1 * h_friction` for local losses.
- Source pressure is sampled in the 20-30 m range in stochastic experiments.
- Each emitter has a design discharge of 0.013 m3/s; available group flow is 0.052 m3/s.
- Hydraulic feasibility requires node head at least 11.4 m and active-pipe velocity between 0.3 and 2.5 m/s.
- The scheduling horizon is 120 h. Dynamic experiments include up to three failures and multiple partial, simultaneous, or absent recoveries.
- Default random seed is `20260709`. Main experiments use 100 normal scenarios, 100 single-fault/single-recovery scenarios, and 500 multi-fault/multi-recovery scenarios.

## Environment

The published runs used Python 3.13.5. Install the recorded package versions with:

```bash
python -m pip install -r requirements.txt
```

## Reproduction

Run commands from the repository root. Each script writes new files under `outputs/`, leaving the published data unchanged.

Hydraulic verification:

```bash
python src/hydraulic_check.py
```

Full verified experiment suite with automatic CPU allocation:

```bash
python src/main_experiment_after_verification.py --workers auto
```

Exact cache benchmark:

```bash
python src/nsga2_cache_benchmark.py --workers auto
```

Surrogate-assisted comparison:

```bash
python src/surrogate_assisted_nsga2.py --workers auto
```

Network-variant sensitivity experiment:

```bash
python src/variant_generalization_analysis.py --workers auto
```

Use each script's `--help` option for reduced-scale smoke tests and parameter overrides. The full experiments are CPU intensive; scenario-level multiprocessing is enabled by `--workers auto`.

## Published data

The CSV files retain scenario-level values, convergence traces, bootstrap confidence intervals, paired Wilcoxon tests, agricultural management metrics, exact cache hit/timing statistics, and paired cached-versus-uncached equivalence checks. JSON files record run parameters and software/hardware metadata. EPANET `.inp` files and the original Excel network tables are included to support independent hydraulic reconstruction.

The exact-cache benchmark demonstrates evaluation equivalence using paired agricultural metrics. The surrogate outputs are retained as a comparative acceleration study; final reported surrogate candidates were reevaluated with the exact hydraulic model.
