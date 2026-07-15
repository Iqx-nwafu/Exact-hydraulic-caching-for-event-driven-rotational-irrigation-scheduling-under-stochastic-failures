# Closed-PDA rotational irrigation scheduling under stochastic failures

This repository contains the revised, reviewer-audited implementation and numerical evidence for event-driven rotational irrigation scheduling. The published experiments use a hydraulically closed pressure-dependent analysis (PDA), causal fault/recovery decisions, canonical multiobjective algorithms, three-level exact caching, and scenario-level statistical inference.

## What changed

The current revision supersedes all pre-verification results. In particular, it:

- iterates emitter discharge, pipe flow, Darcy-Weisbach loss, and node pressure to hydraulic convergence;
- validates 100 operational and low-pressure cases against EPANET/WNTR;
- distinguishes causal, oracle, and rolling-prediction information sets;
- compares NSGA-II, SPEA2, RVEA, and MOEA/D under matched evaluation and wall-time budgets;
- reports independent L0-L3 cache ablations, memory use, and scaling from 60 to 480 emitters;
- aggregates optimizer repeats within scenarios before bootstrap and paired inference;
- treats surrogate-assisted NSGA-II as a negative supplementary result because it failed the predefined operational equivalence gate.

The main numerical conclusions are summarized in [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md). The experimental protocol and reviewer-evidence map are in [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) and [REVIEWER_RESPONSE_EVIDENCE.md](REVIEWER_RESPONSE_EVIDENCE.md).

## Repository layout

~~~text
code/                         experiment, analysis, table, figure, and audit scripts
data/network/                 Nodes.xlsx and Pipes.xlsx
data/hydraulic_validation/    closed-PDA versus EPANET/WNTR validation
data/main_experiments/        normal, single-fault, and multi-fault policy results
data/algorithm_tuning/        tuning grids and frozen algorithm parameters
data/algorithm_quality/       Pareto fronts, convergence, HV, IGD+, epsilon, and spacing
data/cache_scalability/       L0-L3 ablations, timing, memory, and network scaling
data/information_set/         causal, oracle, persistence, and rolling controls
data/surrogate_grouped/       leakage-controlled surrogate data and fitted models
data/surrogate_experiments/   exact-final-evaluation surrogate-assisted results
data/paper_statistics/        scenario-level summaries and inferential tests
data/paper_tables/            main and supplementary tables
data/figure_source_data/      source values for every published figure
figures/                      PNG, SVG, and PDF publication figures
~~~

High-resolution TIFF exports and raw console logs are intentionally excluded from Git history. They are generated artifacts, and the TIFF files alone add about 260 MB. The figure script can regenerate them for submission.

## Environment

The formal runs used Python 3.13.5 and 20 worker processes on a 24-logical-core workstation. Exact package versions are recorded in [environment_versions.json](environment_versions.json) and [requirements.txt](requirements.txt).

~~~bash
python -m pip install -r requirements.txt
~~~

Publication figures additionally require R with ggplot2, patchwork, dplyr, tidyr, readr, scales, and ggrepel.

## Reproduction

Run commands from the repository root. Computational scripts default to outputs/, so published data under data/ is not overwritten.

~~~bash
python code/validate_closed_pda.py
python code/tune_algorithms.py --workers 20
python code/run_revised_main_experiments.py --workers 20
python code/run_algorithm_quality_benchmark.py --workers 20
python code/run_cache_scalability.py --workers 20
python code/run_information_set_experiment.py --workers 20
python code/generate_grouped_surrogate_dataset.py --workers 20
python code/train_grouped_surrogates.py
python code/run_surrogate_assisted_experiments.py --workers 20
python code/analyze_revised_results.py
~~~

Regenerate tables and figures from the published data:

~~~bash
python code/generate_paper_tables.py
Rscript code/make_publication_figures.R .
~~~

Validate the complete published package:

~~~bash
python code/final_audit.py
~~~

The full experiments are CPU intensive. Use each script's --help output to select reduced scenario counts for smoke tests.

## Published evidence

The repository includes 10,400 main policy rows, 2,400 independent algorithm-quality runs, 3,240 cache runs, 160 scalability runs, 240 information-set rows, 42,000 leakage-controlled surrogate samples, six main tables, ten supplementary tables, five main figures, and two supplementary figures. FINAL_AUDIT.json records the machine-checked package counts.

See [DATA_REPLACEMENT_MANIFEST.md](DATA_REPLACEMENT_MANIFEST.md) for the authoritative replacement map and the results that must not be reused.
