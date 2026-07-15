# Data replacement manifest

The files in this package supersede numerical results produced before closed-PDA hydraulic verification. The working manuscript has not been edited.

## Use for revised main text

- Hydraulic validation: `data/hydraulic_validation/`
- Main normal/single-fault/multiple-fault outcomes: `data/main_experiments/`
- Held-out Pareto algorithm quality: `data/algorithm_quality/`
- Cache ablation and scaling: `data/cache_scalability/`
- Causal/oracle/rolling information set: `data/information_set/`
- Scenario-level inference: `data/paper_statistics/`
- Main tables: `data/paper_tables/main/`
- Main figures: `figures/main/`

## Use for supplementary material

- Algorithm tuning: `data/algorithm_tuning/`
- Surrogate grouped/external data and model comparison: `data/surrogate_grouped/`
- Surrogate-assisted negative result: `data/surrogate_experiments/`
- Detailed supplementary tables: `data/paper_tables/supplementary/`
- Supplementary figures: `figures/supplementary/`

## Do not reuse as revised evidence

- Pre-verification hydraulic calculations using post-hoc pressure-flow reduction.
- Timing ratios based on unequal 12-task versus 90-task workloads.
- Cache results that do not distinguish cold, cross-seed warm, and exact hot replay.
- Algorithm comparisons implemented only as selection variants without canonical algorithm mechanics.
- Bootstrap confidence intervals treating repeated runs from one scenario as independent observations.
- The legacy 6,000-row surrogate shuffled train/test split as evidence of topology-general prediction.

## Manuscript status

`Manuscript.docx` remains unchanged by design. Use the main/supplementary table and figure manifests when the text revision is authorized.
