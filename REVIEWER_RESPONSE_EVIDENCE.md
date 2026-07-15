# Reviewer-comment evidence map

This file records completed revisions and their evidence. It is not a response letter and does not modify the manuscript.

## 1. Independent cache ablations, complexity, memory, and scalability

Completed:

- Four levels: no cache, pipe cache, pipe+group cache, complete three-level cache.
- Three separately labeled states: cold, cross-seed warm, and exact hot replay.
- Same scenario, task class, evaluation budget, and paired optimizer seeds across levels.
- 3,240 cache timing runs, 160 scalability runs, peak RSS, estimated cache storage, and empirical power-law fits over 60-480 emitters.

Primary evidence: Main Table 6, Main Figure 4, Supplementary Tables S7-S8, cache metadata and raw runs.

## 2. Physically closed low-pressure discharge

Accepted and corrected:

- Replaced post-hoc discharge reduction with an iterative closed PDA solve.
- Recomputed pipe flows, Darcy-Weisbach/Colebrook losses, node heads, and emitter discharge to convergence.
- Validated 100 random operational/stress cases against EPANET/WNTR PDA.
- Recomputed all revised experiments with the closed solver.

Primary evidence: Main Figure 1, Main Table 2, hydraulic-validation raw comparisons and metadata.

## 3. Information available at reoptimization

Accepted and clarified:

- Deployable method is causal and does not know future source heads or future fault/recovery times.
- Added oracle future-head control, causal mean forecast, causal persistence forecast, and causal rolling control.
- Four modes use paired scenario and optimizer seeds.

Primary evidence: Main Figure 5 and Supplementary Table S4.

## 5. Pseudoreplication in bootstrap inference

Accepted and corrected:

- Algorithm repeats are first aggregated within scenario.
- Bootstrap and paired tests resample/use scenario or context as the independent unit.
- Added 10,000-replicate bootstrap intervals, paired Wilcoxon tests, Holm correction, and scenario-level Friedman tests.

Primary evidence: statistical-analysis metadata and Supplementary Tables S2-S3.

## 6. Unfair timing comparison

Accepted and corrected:

- Removed the 12-task versus 90-task acceleration comparison.
- All cache levels now use identical task composition, seeds, evaluation budgets, and cold/warm/hot definitions.
- Reports mean, standard deviation, confidence interval, worst case, peak RSS, and cache storage.

Primary evidence: Main Table 6, Main Figure 4, cache ablation raw runs.

## 7. Pareto-set quality versus deterministic selected solution

Accepted and corrected:

- Replaced selection-only algorithm variants with canonical pymoo NSGA-II, SPEA2, RVEA, and MOEA/D.
- Systematically tuned each algorithm and froze selected parameters before held-out comparison.
- Added HV, IGD+, additive epsilon, nondominated count, spacing, Pareto fronts, convergence traces, equal-FE comparison, equal-wall-time comparison, and repeated-run statistics.
- Retained agricultural selected-solution outcomes as a separate operational layer.

Primary evidence: Main Figure 3, Main Table 5, tuning files, algorithm-quality fronts/traces, Supplementary Tables S1-S3.

## Additional surrogate audit

- Demonstrated why the legacy 6,000-row shuffled split cannot support scenario-independent or topology-external claims.
- Generated a 42,000-row grouped/topology-aware replacement data set.
- Tested three tuned tree surrogates on internal grouped and external-topology sets.
- The surrogate-assisted optimizer failed the predefined equivalence/time gate and is reported as a negative supplementary result.

Primary evidence: Supplementary Figures S1-S2 and Supplementary Tables S5-S6, S9-S10.
