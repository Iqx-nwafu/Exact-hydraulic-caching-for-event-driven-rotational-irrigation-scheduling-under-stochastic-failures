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
  network_topology.py                      network input, tree-derived topology, platform JSON converter
data/
  network/                                  Nodes.xlsx, Pipes.xlsx, and Emitters.csv
  hydraulic_verification/                   100-case verification and EPANET INP files
  main_experiments/                         normal, single-fault, and multi-fault results
  surrogate/                                surrogate and solver-efficiency comparisons
  variant_network/                          network-variant and sensitivity results
```

## Network input

The network is defined entirely by its tables; no node-naming convention is assumed.

- `Nodes.xlsx`: `node ID, X, Y, Z` (first column is the node ID).
- `Pipes.xlsx`: `Pipe ID, FromNode, ToNode, Length_m, Diameter_m, Material`, plus an optional `Roughness_mm` used for materials without a `--roughness-<material>-mm` value.
- `Emitters.csv`: `emitter_id, node_id`, one row per emitter in evaluation order. Every node with at least one emitter is an irrigation node.

The tree structure is derived from the pipes: the source is the only node without an upstream pipe, each node's supply path is traced to the source, a lateral is a maximal chain of irrigation nodes hanging from a non-irrigation node, and a node failure isolates every emitter in the node's downstream subtree. The conventional rotation (`original` policy) opens each lateral from upstream to downstream, `group_size` emitters at a time.

Every script accepts `--nodes/--pipes/--emitters` (CSV or Excel) or `--network-json` for a Hydraulic Workbench Studio planning JSON (`schema: hydraulic-irrigation-planning`, version 1), in which each `fields` entry is one emitter and pipes use `headloss_formula: D-W` with `roughness_m`. Convert between the two formats with:

```bash
python src/network_topology.py to-json --out outputs/network.json
python src/network_topology.py to-excel --network-json outputs/network.json --out-dir outputs/network_tables --format csv
```

`--format csv` writes a bit-exact copy; `.xlsx` output stores floats with 16 significant digits. The JSON keeps each pipe's `material`, so material roughness overrides still apply after conversion. The published tables and the converted JSON produce identical experiment results.

## Model and experiment scope

- Darcy-Weisbach friction losses use the EPANET 2.2 all-regime friction factor (64/Re, Dunlop transition, Swamee-Jain) and material roughness values, identical to the EPANET/WNTR verification model and the hydraulic workbench.
- EPANET comparison uses friction losses only. The optimization model applies the separate engineering correction `h_total = 1.1 * h_friction` for local losses.
- `--hydraulic-backend platform` routes group hydraulics to the workbench evaluator `hydraulic_moc.irrigation_eval` (same model; requires the workbench package on `PYTHONPATH`). `python src/platform_equivalence_check.py` compares both backends group by group.
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
