from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
PUBLISHED_FIGURE_FORMATS = {".svg", ".pdf", ".png"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    main_policy = pd.read_csv(DATA / "main_experiments" / "revised_policy_metrics.csv")
    require(len(main_policy) == 10_400, "unexpected main policy row count")
    require(main_policy.groupby("experiment").size().to_dict() == {"multi_fault": 4000, "normal": 5600, "single_fault": 800}, "main experiment composition mismatch")
    require(float(main_policy["pda_nonconvergence_rate_pct"].max()) == 0.0, "PDA nonconvergence detected")
    require(main_policy[["achievement_pct", "unmet_emitter_ratio_pct", "average_deficit_m3_per_emitter", "max_deficit_m3", "pressure_uniformity_index_pct", "feasible_group_ratio_pct"]].notna().all().all(), "missing agricultural metrics")

    quality = pd.read_csv(DATA / "algorithm_quality" / "pareto_quality_metrics.csv")
    require(len(quality) == 2400, "unexpected algorithm quality row count")
    require(set(quality["algorithm"]) == {"nsga2", "spea2", "rvea", "moead"}, "algorithm set mismatch")
    require(np.isfinite(quality[["hv_4d", "igd_plus_4d", "additive_epsilon_4d", "spacing_4d"]]).all().all(), "non-finite Pareto metric")

    cache = pd.read_csv(DATA / "cache_scalability" / "cache_ablation_runs.csv")
    scale = pd.read_csv(DATA / "cache_scalability" / "network_scalability_runs.csv")
    require(len(cache) == 3240 and len(scale) == 160, "cache/scalability row count mismatch")
    require(set(cache["cache_level"]) == {0.0, 1.0, 2.0, 3.0}, "cache levels incomplete")
    require(set(cache["cache_state"]) == {"cold", "warm_cross_seed", "hot_exact_replay"}, "cache states incomplete")
    require(set(cache["actual_function_evaluations"]) == {480}, "cache FE mismatch")

    information = pd.read_csv(DATA / "information_set" / "information_set_policy_metrics.csv")
    require(len(information) == 240 and information["scenario_id"].nunique() == 30, "information-set data incomplete")

    surrogate_metadata = pd.read_csv(DATA / "surrogate_grouped" / "surrogate_sample_metadata.csv")
    surrogate_features = pd.read_parquet(DATA / "surrogate_grouped" / "surrogate_features.parquet")
    surrogate_labels = pd.read_parquet(DATA / "surrogate_grouped" / "surrogate_labels.parquet")
    require(len(surrogate_metadata) == len(surrogate_features) == len(surrogate_labels) == 42_000, "surrogate rows misaligned")
    require(surrogate_metadata.groupby("scenario_id")["split"].nunique().max() == 1, "scenario leakage detected")
    require(int((surrogate_metadata.groupby("plan_hash")["split"].nunique() > 1).sum()) == 0, "plan hash leakage detected")

    surrogate_policy = pd.read_csv(DATA / "surrogate_experiments" / "surrogate_policy_metrics.csv")
    require(len(surrogate_policy) == 1100, "surrogate policy row count mismatch")
    require(float(surrogate_policy["pda_nonconvergence_rate_pct"].max()) == 0.0, "surrogate execution PDA nonconvergence")

    paired = pd.read_csv(DATA / "paper_statistics" / "paired_wilcoxon_tests.csv")
    require(paired[["p_value", "p_holm"]].notna().all().all(), "missing inferential p values")

    main_tables = sorted((DATA / "paper_tables" / "main").glob("*.csv"))
    supplementary_tables = sorted((DATA / "paper_tables" / "supplementary").glob("*.csv"))
    require(len(main_tables) == 6, "main table limit/composition mismatch")
    require(len(supplementary_tables) == 10, "supplementary table composition mismatch")

    required_figure_stems = [
        *(f"Figure{index}_" for index in range(1, 6)),
        "FigureS1_",
        "FigureS2_",
    ]
    figure_files = [
        path
        for path in FIGURES.rglob("*")
        if path.is_file() and path.suffix.lower() in PUBLISHED_FIGURE_FORMATS
    ]
    for prefix in required_figure_stems:
        matching = [path for path in figure_files if path.name.startswith(prefix)]
        require(
            {path.suffix.lower() for path in matching} == PUBLISHED_FIGURE_FORMATS,
            f"published figure formats incomplete: {prefix}",
        )
        require(all(path.stat().st_size > 1000 for path in matching), f"empty figure export: {prefix}")

    smoke_directories = [path.name for path in DATA.iterdir() if path.is_dir() and ("smoke" in path.name or "speedtest" in path.name)]
    require(not smoke_directories, "temporary data directories remain")

    manifest_path = DATA / "MANIFEST.sha256"
    manifest_files = sorted(path for path in DATA.rglob("*") if path.is_file() and path != manifest_path)
    manifest_contents = "\n".join(
        f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in manifest_files
    ) + "\n"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(manifest_contents)

    audit = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "PASS",
        "formal_parallel_workers": 20,
        "logical_cpu_count": os.cpu_count(),
        "main_policy_rows": len(main_policy),
        "algorithm_quality_runs": len(quality),
        "cache_runs": len(cache),
        "scalability_runs": len(scale),
        "information_set_rows": len(information),
        "surrogate_samples": len(surrogate_features),
        "surrogate_policy_rows": len(surrogate_policy),
        "main_tables": len(main_tables),
        "supplementary_tables": len(supplementary_tables),
        "main_figures": 5,
        "supplementary_figures": 2,
        "figure_export_files": len(figure_files),
        "published_figure_formats": sorted(PUBLISHED_FIGURE_FORMATS),
        "data_manifest_files": len(manifest_files),
        "temporary_data_directories": smoke_directories,
        "manuscript_edited_by_revision_pipeline": False,
    }
    (ROOT / "FINAL_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
