# Figure contract

## Main Figure 1: closed-PDA hydraulic validation

- Core conclusion: the iterative pressure-dependent hydraulic solver reproduces EPANET/WNTR pressure, demand, flow, and friction-loss results over operational and low-pressure stress cases.
- Archetype: quantitative grid.
- Hero evidence: estimated versus EPANET active-node pressure with identity line.
- Supporting evidence: Bland-Altman limits of agreement, cross-quantity error intervals, and convergence ECDF.
- Reviewer risk: local-loss correction must not be mixed into the friction-only equivalence comparison.

## Main Figure 2: agricultural value of optimization and dynamic reoptimization

- Core conclusion: tuned optimization removes normal-operation water deficits and dynamic reoptimization recovers delivery after faults relative to static scheduling.
- Archetype: quantitative grid with a dominant normal-operation comparison.
- Hero evidence: scenario-level mean agricultural outcomes with 95% bootstrap confidence intervals.
- Supporting evidence: paired dynamic-minus-static effect distributions and a median recovery-effect matrix.
- Statistics: scenario is the independent unit; optimizer repeats are averaged within scenario.

## Main Figure 3: multiobjective algorithm quality

- Core conclusion: algorithm preference depends on Pareto quality, distribution, and time; no deterministic agricultural ranking alone proves convergence.
- Archetype: quantitative grid.
- Hero evidence: a representative held-out Pareto-front projection pooled over 20 independent runs.
- Supporting evidence: cross-context ranks for HV, IGD+, epsilon, spacing and time; convergence ribbons; and the quality-cost frontier.
- Statistics: 30 held-out contexts, 20 independent runs per algorithm and context, equal 2,400-function-evaluation budget.

## Main Figure 4: cache ablation and scalability

- Core conclusion: group-level caching provides the largest cold-cache speedup, while plan caching matters primarily for exact hot replay; measured time scales approximately linearly over 60-480 emitters.
- Archetype: asymmetric quantitative grid.
- Hero evidence: a cold-cache waterfall attributing incremental time saved to each layer.
- Supporting evidence: runtime-memory efficiency, highest-layer reuse heatmap, and emitter-count scaling.
- Reviewer risk: cold, cross-seed warm, and exact hot replay must remain separately labeled.

## Main Figure 5: decision information set

- Core conclusion: the deployable causal policy performs close to the perfect-head-information control, while rolling reoptimization trades modest delivery gains for substantially greater computation.
- Archetype: quantitative grid.
- Hero evidence: the achievement-versus-computation operating frontier under each information policy.
- Supporting evidence: paired effects relative to the deployable causal-mean policy and scenario-level policy win rates.
- Reviewer risk: the oracle knows future source head only, not future fault or recovery events.

## Supplementary Figure 1: surrogate predictive generalization

- Core conclusion: surrogate prediction is strong for delivery objectives but weak for pressure/constraint transfer to unseen topologies.
- Archetype: quantitative grid.
- Evidence: internal and external test NRMSE and feasibility classification.

## Supplementary Figure 2: surrogate equivalence and runtime gate

- Core conclusion: the surrogate-assisted method reduces exact evaluations but fails the 5% worst-case agricultural-equivalence gate and increases wall-clock time.
- Archetype: quantitative grid.
- Evidence: normalized maximum differences, runtime, and exact evaluation count.

## Export contract

- Backend: R only.
- Main size: 183 mm wide, 118-130 mm high.
- Supplementary size: 183 mm wide, 105-118 mm high.
- Formats: editable SVG, PDF, and 600 dpi TIFF.
- Font: Arial with sans-serif fallback.
- Source-data unit: scenario/context after within-scenario repeat aggregation where applicable.
