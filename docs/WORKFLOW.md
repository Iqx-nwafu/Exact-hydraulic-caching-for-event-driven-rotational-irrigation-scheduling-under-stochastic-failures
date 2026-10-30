# Research workflow with the Hydraulic Workbench Studio

This page describes the chain from a network drawn or imported in the workbench to the published experiment tables, how to check that the chain runs on a new network or machine, and how other code calls the same hydraulic evaluator.

```text
workbench planning JSON ──► network_topology.py ──► hydraulic_check.py (EPANET/WNTR)
  (or Nodes/Pipes/Emitters)        │
                                   ├─► main_experiment_after_verification.py ─┬─► surrogate_assisted_nsga2.py
                                   │      --hydraulic-backend research|platform│
                                   ├─► nsga2_cache_benchmark.py                │
                                   └─► variant_generalization_analysis.py      └─► summaries, CI, Wilcoxon tests
```

## 1. Setup

- Python 3.13 with `python -m pip install -r requirements.txt`.
- For the platform steps, the workbench source (`hydraulic-workbench-studio-source-code`) on `PYTHONPATH`. Only `numpy` is needed to import `hydraulic_moc.irrigation_eval`; the GUI, licensing and online modules are not loaded.

```bash
export PYTHONPATH=/path/to/hydraulic-workbench-studio-source-code      # PowerShell: $env:PYTHONPATH = "C:\path\to\..."
```

## 2. Network input

Every script takes either the research tables (`--nodes/--pipes/--emitters`) or a workbench planning JSON (`--network-json`). The JSON is the shared format: the workbench saves it, the research code reads it, and `network_topology.py` converts in both directions.

```bash
python src/network_topology.py to-json --out outputs/network.json          # tables -> JSON
python src/network_topology.py to-excel --network-json outputs/network.json --out-dir outputs/tables --format csv
```

What the research code reads from the JSON (`schema: hydraulic-irrigation-planning`, `schema_version: 1`):

| JSON | Meaning |
|---|---|
| `topology.nodes[]` | `node_id`, `elevation_m`, `x`, `y` |
| `topology.pipes[]` | `pipe_id`, `start_node`, `end_node`, `length_m`, `diameter_m`, `headloss_formula: "D-W"`, `roughness_m` |
| `topology.source_node_id` | must be the tree root (the only node without an upstream pipe) |
| `fields[]` | one entry per emitter: `field_id` is the emitter id, `node_id` its node; row order is the emitter index |
| `metadata.research_pipe_material` | pipe materials, so `--roughness-<material>-mm` overrides still apply (the workbench loader rejects unknown pipe keys) |

Requirements: a single-source tree, Darcy-Weisbach pipes with roughness, and no emitter on the source node. The workbench's bundled `planning_tree71.json` uses Hazen-Williams pipes and cannot be used as is.

## 3. Check the chain on a new network or machine

```bash
python src/run_workflow.py --network-json outputs/network.json --output-dir outputs/workflow
```

Without `--network-json` it converts `data/network` first. Every step runs at reduced scale (a few scenarios, population 16, 4 generations), and `outputs/workflow/workflow_report.json` records the commands, wall times and these checks:

| Check | Passes when |
|---|---|
| `platform_load` | the workbench loader accepts the JSON and returns one field per emitter |
| `verification` | max \|model − EPANET\| active-node pressure is below 0.05 m |
| `backend_equivalence` | the research and platform evaluators agree on 2000 random groups with fault-isolated emitters (no feasibility mismatch, pressure difference below 1e-9 m) |
| `backend_experiments` | the optimisation experiment gives the same metrics with either backend |
| `cache_equivalence` | cached and uncached runs give identical agricultural metrics |
| `surrogate` | the surrogate comparison is paired with this run's main experiment |
| `variant` | the network-variant summary is written |

The run takes a few minutes on four cores. The surrogate step skips its fixed-size wall-time benchmark (`--skip-efficiency-benchmark`), whose code paths the cache benchmark and the surrogate experiment already run. Platform checks are skipped when `hydraulic_moc` is not importable (`"platform_available": false` in the report); the research-only chain still runs.

## 4. Full-scale experiments

Run in this order; each script writes under `outputs/` and leaves `data/` unchanged.

```bash
python src/hydraulic_check.py                          --network-json outputs/network.json
python src/main_experiment_after_verification.py       --network-json outputs/network.json --workers auto
python src/nsga2_cache_benchmark.py                    --network-json outputs/network.json --workers auto
python src/surrogate_assisted_nsga2.py                 --network-json outputs/network.json --workers auto \
       --baseline-main-dir outputs/main_experiments
python src/variant_generalization_analysis.py          --network-json outputs/network.json --workers auto
```

`--baseline-main-dir` pairs the surrogate runs with the main experiment just produced; its default is the published `data/main_experiments`. Wall-time results (cache benchmark, surrogate efficiency) depend on the machine; compare timings only within one machine and one worker count. The cache's wall-time saving scales with the cost of the hydraulic solve it skips: with the explicit EPANET 2.2 law the full benchmark on the 24-core machine gives 53.6 s uncached and 50.6 s cached (1.06×), against 1.50× for the iterated Colebrook-White law of the published runs. Cached and uncached runs give identical agricultural metrics under both laws.

## 5. Model conventions

The research backend and the workbench evaluator implement the same model, checked group by group by `platform_equivalence_check.py`.

| Item | Convention |
|---|---|
| Hydraulics | steady, demand-driven, single-source tree: pipe flow is the sum of the design discharges of the open emitters downstream |
| Friction | Darcy-Weisbach with material roughness and the EPANET 2.2 friction factor (64/Re, Dunlop transition, Swamee-Jain), the same law as EPANET and the workbench evaluator's `friction="epanet"`; kinematic viscosity 1.0e-6 m²/s |
| Local losses | `h = 1.1 × h_friction` (`local_loss_ratio = 0.1`); the EPANET check compares friction losses only |
| Source head | hydraulic grade line at the source, sampled in 20–30 m |
| Pressure | `p = H_s − path loss − z`, a pressure head; at least 11.4 m at every node with an open emitter |
| Velocity | 0.3–2.5 m/s on pipes that carry flow (workbench planner equivalent: `velocity_scope = "active_only"`) |
| Faults | a failed node cuts off every emitter in its downstream subtree; a fault inside a group's slot ends the slot at the fault time, then the dynamic policy re-optimises the remaining emitters from their delivered volumes and the static policy runs the interrupted group again for its full duration |
| Emitters | design discharge 0.013 m³/s, four per group (0.052 m³/s) |

## 6. Hydraulic backends

- `--hydraulic-backend research` (default): the scalar research implementation with three exact caches (group, plan, per-pipe loss). Use it for the cache benchmark: the per-pipe loss cache exists only here.
- `--hydraulic-backend platform`: group hydraulics from `hydraulic_moc.irrigation_eval.GroupEvaluator` (see `src/hydraulics_backend.py`). Group and plan caches stay in front of it. Results match the research backend to floating-point round-off.

## 7. Calling the evaluator from other code

The research code uses the workbench module `hydraulic_moc.irrigation_eval`; `examples/evaluate_groups.py` shows the two calls it relies on:

```bash
python examples/evaluate_groups.py outputs/network.json
```

```python
from hydraulic_moc.irrigation_data import load_planning_project
from hydraulic_moc.irrigation_eval import Emitter, GroupEvaluator, Limits, SteadyModel, TreeNetwork

project = load_planning_project("outputs/network.json")
emitters = [Emitter(f.field_id, f.node_id, f.design_flow_m3_h / 3600.0, 11.4) for f in project.fields]
net = TreeNetwork(project.topology, emitters)
ev = GroupEvaluator(net, SteadyModel(friction="epanet", local_loss_ratio=0.1), Limits(11.4, 0.3, 2.5))

r = ev.evaluate([0, 1, 2, 3], source_head_m=25.0, unavailable=net.downstream_emitters("J12"))
r.feasible, r.min_pressure_m, r.node_pressure_m, r.velocity_m_s, r.model_digest
```

Emitter indices follow the order of `fields`; `evaluate` caches exactly on the open-emitter set (hydraulics do not depend on the source head), `evaluate_batch` takes `(B, E)` boolean masks and is vectorised. Every result carries `model_digest` (network, model and limits hashes) to tag outputs.

New studies should start from the workbench's general headless API (`hydraulic_moc.api`, documented in the workbench's `docs/headless_api.md`), which adds looped networks, Hazen-Williams pipes, time-line simulation and Monte Carlo runs. Its defaults follow the workbench planner (no proportional local loss, velocity checked on all pipes); to reproduce this study's model set `local_loss_ratio = 0.1`, velocity on flowing pipes only and emitter design head 11.4 m in `settings.hydraulic_api` or in the call. For the plan objectives, call `api.simulate(..., head_sampling="step_start", pressure_weighting="segment")` and `api.plan_objectives(sim.metrics)` (API 1.1): one source head per irrigation slot and equal weight per slot, as in `plan_objectives` here; `src/platform_metrics_check.py` checks the match (100 head profiles × original and random rotations, largest difference 3e-14). The default time-weighted convention gives the same water metrics but a pressure non-uniformity up to about 2 percentage points different. Its time line continues an interrupted step for the remaining duration, whereas this study's static policy reruns the interrupted group for its full duration (Section 5).

## 8. Limits of the current chain

- Single-source trees with Darcy-Weisbach pipes only; no loops, pumps, valves or tanks.
- Demand-driven: emitter outflow reduction at low pressure is reported, not fed back into pipe flows.
- `hydraulic-moc irrigation-eval --project` reads only the topology; emitters come from a separate CSV and limits from command-line options.
- The workbench GUI planner is a separate implementation (EPANET friction, per-pipe K only, velocity on all pipes by default); its results match this model only with the same options.
