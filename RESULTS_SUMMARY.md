# Revised results summary

## Hydraulic consistency

- The closed pressure-dependent solver converged in all 100 validation cases.
- Fifty cases used operational source heads of 20-30 m and fifty used low-pressure stress heads of 8-20 m.
- Mean case-level active-node pressure MAE was approximately 0.00435 m under operational heads and 0.00311 m under stress heads.
- Pipe-flow MAE was on the order of 10^-8 m3/s. No local-loss multiplier was included in this EPANET friction-only equivalence check.

## Normal operation

Scenario-level means across 100 source-head scenarios:

| Method | Achievement (%) | Unmet emitters (%) | Mean deficit (m3/emitter) | Pressure uniformity (%) | Feasible groups (%) |
|---|---:|---:|---:|---:|---:|
| Original near-to-far | 99.630 | 24.467 | 0.693 | 77.730 | 0.000 |
| Random rotation | 100.000 | 0.001 | 0.000 | 90.059 | 17.666 |
| Tuned NSGA-II | 100.000 | 0.000 | 0.000 | 90.244 | 74.673 |

Tuned NSGA-II improved all six primary agricultural/hydraulic metrics relative to the original schedule in paired scenario tests after Holm correction. Random scheduling delivered the target volume in nearly all cases but produced far fewer hydraulically feasible groups.

## Dynamic reoptimization

Across algorithms, dynamic versus static scheduling changed the scenario means as follows:

| Disturbance | Decision | Achievement (%) | Unmet emitters (%) | Mean deficit (m3/emitter) | Affected target met (%) |
|---|---|---:|---:|---:|---:|
| Single fault/recovery | Static | 96.361 | 5.433 | 6.813 | 61.035 |
| Single fault/recovery | Dynamic | 98.448 | 2.792 | 2.905 | 87.027 |
| Multiple faults/recoveries | Static | 93.074 | 9.999 | 12.965 | 59.587 |
| Multiple faults/recoveries | Dynamic | 96.807 | 4.700 | 5.977 | 75.524 |

Dynamic reoptimization substantially improved delivery and affected-emitter recovery. Pressure uniformity decreased by less than 0.1 percentage point in the NSGA-II paired comparisons, so the result should be described as a delivery-recovery gain with a small hydraulic-uniformity tradeoff.

## Multiobjective algorithms

No algorithm dominates every criterion:

- SPEA2 had the highest mean 4D hypervolume under equal function evaluations.
- MOEA/D had the lowest mean IGD+ and the largest nondominated set.
- RVEA had the lowest mean wall time and the highest mean equal-wall-time hypervolume.
- NSGA-II remained competitive in agricultural delivery but had lower held-out Pareto-quality means and higher runtime than RVEA/MOEA-D.

The manuscript should therefore report criterion-specific strengths and Pareto distributions, not a single deterministic winner derived from one operational ranking rule.

## Cache ablation and scaling

- Cold-cache mean speedup relative to no cache: L1 pipe cache 2.21x; L1+L2 group cache 8.56x; full L1+L2+L3 cache 8.48x.
- Exact hot replay with full caching averaged approximately 0.062 s versus approximately 24.1 s without caching.
- L2/L3 cold-cache storage was approximately 22-23 MB per task, with roughly 52 MB mean peak RSS increment.
- Empirical time exponents over 60-480 emitters were 1.03-1.07 with R2 at least 0.9966, indicating near-linear measured scaling over the tested range.

Cold, cross-seed warm, and exact hot replay results must remain separately labeled. They cannot be collapsed into one acceleration ratio.

## Information set

The deployable causal event-driven policy used observed current/past source heads, current faults, and cumulative delivered volumes, with a 25 m expectation for future heads. It did not know future source heads or future event times. The oracle control knew future source heads only. Rolling causal optimization produced modest delivery gains in some cases but required roughly 68-72 s total optimization time, versus roughly 10-14 s for event-driven causal policies.

## Surrogate model

The legacy 6,000-row data set cannot demonstrate leakage-free generalization because it lacks scenario IDs, topology IDs, and plan hashes and used a shuffled row split.

The replacement contains 30,000 grouped development samples and 12,000 samples from three fully held-out topologies. It has zero scenario overlap, zero exact-plan-hash overlap, and no near duplicate below Hamming distance 0.05. However, external-topology pressure/constraint prediction remained weak.

Surrogate-assisted NSGA-II reduced exact evaluations by approximately 32.65% in normal operation but increased mean optimization time from 10.88 s to 19.11 s and reduced mean feasible-group ratio from 74.67% to 58.18%. It failed the predefined 5% worst-case equivalence gate and should be retained as a negative supplementary result, not recommended as an accelerator.

## Statistical interpretation

- Independent inference unit: scenario/context.
- Optimizer repeats were averaged within scenario before inference.
- Uncertainty: 10,000-replicate scenario bootstrap confidence interval.
- Pairwise tests: paired two-sided Wilcoxon signed-rank with Holm correction.
- Multi-algorithm global tests: Friedman tests on scenario-aggregated results.
