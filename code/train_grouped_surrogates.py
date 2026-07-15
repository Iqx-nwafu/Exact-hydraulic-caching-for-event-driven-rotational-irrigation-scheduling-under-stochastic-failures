from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from revision_core import DEFAULT_WORKERS


TARGETS = [
    "obj_achievement_loss",
    "obj_average_deficit",
    "obj_pressure_nonuniformity",
    "constraint_violation",
]


GRIDS: dict[str, list[dict[str, Any]]] = {
    "catboost": [
        {"iterations": 500, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3.0},
        {"iterations": 800, "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 5.0},
        {"iterations": 600, "depth": 8, "learning_rate": 0.04, "l2_leaf_reg": 5.0},
        {"iterations": 900, "depth": 8, "learning_rate": 0.025, "l2_leaf_reg": 8.0},
    ],
    "lightgbm": [
        {"n_estimators": 500, "num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 20},
        {"n_estimators": 800, "num_leaves": 31, "learning_rate": 0.03, "min_child_samples": 30},
        {"n_estimators": 600, "num_leaves": 63, "learning_rate": 0.04, "min_child_samples": 30},
        {"n_estimators": 900, "num_leaves": 63, "learning_rate": 0.025, "min_child_samples": 50},
    ],
    "xgboost": [
        {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05, "min_child_weight": 1.0},
        {"n_estimators": 800, "max_depth": 6, "learning_rate": 0.03, "min_child_weight": 3.0},
        {"n_estimators": 600, "max_depth": 8, "learning_rate": 0.04, "min_child_weight": 3.0},
        {"n_estimators": 900, "max_depth": 8, "learning_rate": 0.025, "min_child_weight": 5.0},
    ],
}


def build_model(family: str, config: dict[str, Any], seed: int) -> Any:
    if family == "catboost":
        return CatBoostRegressor(
            **config,
            loss_function="MultiRMSE",
            random_seed=seed,
            thread_count=DEFAULT_WORKERS,
            verbose=False,
            allow_writing_files=False,
        )
    if family == "lightgbm":
        base = LGBMRegressor(
            **config,
            objective="regression",
            random_state=seed,
            n_jobs=DEFAULT_WORKERS,
            subsample=0.9,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            verbosity=-1,
        )
        return MultiOutputRegressor(base, n_jobs=1)
    if family == "xgboost":
        base = XGBRegressor(
            **config,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=DEFAULT_WORKERS,
            subsample=0.9,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            tree_method="hist",
        )
        return MultiOutputRegressor(base, n_jobs=1)
    raise ValueError(f"unknown family: {family}")


def normalized_rmse(y_true: np.ndarray, y_pred: np.ndarray, scales: np.ndarray) -> np.ndarray:
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    return rmse / np.maximum(scales, 1.0e-12)


def select_feasibility_threshold(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    true_feasible = y_true <= 1.0e-9
    if np.unique(true_feasible).size < 2:
        return 1.0e-9, 1.0
    positive_predictions = np.maximum(y_pred, 0.0)
    candidates = np.unique(
        np.concatenate(
            ([0.0, 1.0e-9], np.quantile(positive_predictions, np.linspace(0.01, 0.99, 99)))
        )
    )
    scores = [
        balanced_accuracy_score(true_feasible, positive_predictions <= threshold)
        for threshold in candidates
    ]
    best = int(np.argmax(scores))
    return float(candidates[best]), float(scores[best])


def split_metrics(
    family: str,
    split_name: str,
    topology_id: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scales: np.ndarray,
    feasibility_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(TARGETS):
        true = y_true[:, target_index]
        predicted = y_pred[:, target_index]
        nonzero = np.abs(true) > 1.0e-9
        rows.append(
            {
                "model_family": family,
                "split": split_name,
                "topology_id": topology_id,
                "target": target,
                "n": len(true),
                "mae": mean_absolute_error(true, predicted),
                "rmse": np.sqrt(mean_squared_error(true, predicted)),
                "nrmse_iqr": normalized_rmse(
                    true.reshape(-1, 1), predicted.reshape(-1, 1), scales[target_index : target_index + 1]
                )[0],
                "r2": r2_score(true, predicted),
                "mape_nonzero_pct": (
                    float(np.mean(np.abs((predicted[nonzero] - true[nonzero]) / true[nonzero])) * 100.0)
                    if np.any(nonzero)
                    else np.nan
                ),
            }
        )
    true_feasible = y_true[:, -1] <= 1.0e-9
    predicted_feasible = np.maximum(y_pred[:, -1], 0.0) <= feasibility_threshold
    rows.append(
        {
            "model_family": family,
            "split": split_name,
            "topology_id": topology_id,
            "target": "feasibility_classification",
            "n": len(y_true),
            "mae": np.nan,
            "rmse": np.nan,
            "nrmse_iqr": np.nan,
            "r2": np.nan,
            "mape_nonzero_pct": np.nan,
            "balanced_accuracy": (
                balanced_accuracy_score(true_feasible, predicted_feasible)
                if np.unique(true_feasible).size > 1
                else float(np.mean(true_feasible == predicted_feasible))
            ),
            "false_feasible_rate_pct": float(np.mean(predicted_feasible & ~true_feasible) * 100.0),
            "false_infeasible_rate_pct": float(np.mean(~predicted_feasible & true_feasible) * 100.0),
        }
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grouped and external-topology surrogate comparison")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "surrogate_grouped",
    )
    parser.add_argument("--seed", type=int, default=2026072101)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    features = pd.read_parquet(args.data_dir / "surrogate_features.parquet")
    labels = pd.read_parquet(args.data_dir / "surrogate_labels.parquet")
    metadata = pd.read_csv(args.data_dir / "surrogate_sample_metadata.csv")
    if not (len(features) == len(labels) == len(metadata)):
        raise ValueError("surrogate feature, label, and metadata rows are not aligned")
    if metadata.groupby("scenario_id")["split"].nunique().max() != 1:
        raise ValueError("scenario leakage detected across data splits")

    x = features.to_numpy(dtype=np.float32)
    y = labels[TARGETS].to_numpy(dtype=np.float64)
    masks = {name: metadata["split"].eq(name).to_numpy() for name in ["train", "validation", "test", "external_test"]}
    scales = np.subtract(*np.quantile(y[masks["train"]], [0.75, 0.25], axis=0))
    scales = np.maximum(scales, np.std(y[masks["train"]], axis=0) * 0.1)

    tuning_rows: list[dict[str, Any]] = []
    selected_configs: dict[str, dict[str, Any]] = {}
    selected_thresholds: dict[str, tuple[float, float]] = {}
    for family, configs in GRIDS.items():
        best_score = np.inf
        best_config: dict[str, Any] | None = None
        for config_id, config in enumerate(configs):
            fit_started = time.perf_counter()
            model = build_model(family, config, args.seed + config_id)
            model.fit(x[masks["train"]], y[masks["train"]])
            predicted = np.asarray(model.predict(x[masks["validation"]]), dtype=float)
            nrmse = normalized_rmse(y[masks["validation"]], predicted, scales)
            score = float(np.mean(nrmse))
            threshold, threshold_score = select_feasibility_threshold(
                y[masks["validation"], -1], predicted[:, -1]
            )
            operational_score = score + 0.5 * (1.0 - threshold_score)
            tuning_rows.append(
                {
                    "model_family": family,
                    "config_id": config_id,
                    "config_json": json.dumps(config, sort_keys=True),
                    "validation_mean_nrmse_iqr": score,
                    "validation_operational_score": operational_score,
                    "validation_feasibility_threshold": threshold,
                    "validation_feasibility_balanced_accuracy": threshold_score,
                    **{f"validation_nrmse_{target}": nrmse[index] for index, target in enumerate(TARGETS)},
                    "fit_and_predict_time_s": time.perf_counter() - fit_started,
                }
            )
            print(f"{family} config {config_id + 1}/{len(configs)}: validation NRMSE={score:.6f}", flush=True)
            if operational_score < best_score:
                best_score = operational_score
                best_config = config
                selected_thresholds[family] = (threshold, threshold_score)
            del model
        if best_config is None:
            raise RuntimeError(f"no valid configuration for {family}")
        selected_configs[family] = best_config

    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    final_model_rows: list[dict[str, Any]] = []
    fit_mask = masks["train"] | masks["validation"]
    for family, config in selected_configs.items():
        fit_started = time.perf_counter()
        model = build_model(family, config, args.seed + 10_000)
        model.fit(x[fit_mask], y[fit_mask])
        threshold, validation_balanced_accuracy = selected_thresholds[family]
        model_path = args.data_dir / f"best_{family}_surrogate.joblib"
        joblib.dump(model, model_path, compress=3)
        final_model_rows.append(
            {
                "model_family": family,
                "selected_config_json": json.dumps(config, sort_keys=True),
                "feasibility_threshold": threshold,
                "validation_feasibility_balanced_accuracy": validation_balanced_accuracy,
                "final_fit_time_s": time.perf_counter() - fit_started,
                "model_size_mb": model_path.stat().st_size / 1024**2,
                "model_path": str(model_path),
            }
        )
        for split_name in ["test", "external_test"]:
            split_indices = np.flatnonzero(masks[split_name])
            predicted = np.asarray(model.predict(x[split_indices]), dtype=float)
            prediction = metadata.iloc[split_indices][
                ["sample_id", "scenario_id", "topology_id", "context_type", "split"]
            ].reset_index(drop=True)
            prediction.insert(0, "model_family", family)
            for target_index, target in enumerate(TARGETS):
                prediction[f"actual_{target}"] = y[split_indices, target_index]
                prediction[f"predicted_{target}"] = predicted[:, target_index]
            prediction_rows.append(prediction)
            for topology_id in ["all", *sorted(prediction["topology_id"].unique())]:
                local = np.ones(len(prediction), dtype=bool) if topology_id == "all" else prediction["topology_id"].eq(topology_id).to_numpy()
                metric_rows.extend(
                    split_metrics(
                        family,
                        split_name,
                        topology_id,
                        y[split_indices][local],
                        predicted[local],
                        scales,
                        threshold,
                    )
                )
        del model

    tuning = pd.DataFrame(tuning_rows)
    final_models = pd.DataFrame(final_model_rows)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    tuning.to_csv(args.data_dir / "surrogate_model_tuning.csv", index=False, encoding="utf-8-sig")
    final_models.to_csv(args.data_dir / "surrogate_selected_models.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(args.data_dir / "surrogate_test_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_parquet(args.data_dir / "surrogate_test_predictions.parquet", index=False)
    best_family = (
        tuning.sort_values("validation_operational_score")
        .groupby("model_family", as_index=False)
        .first()
        .sort_values("validation_operational_score")
        .iloc[0]["model_family"]
    )
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.replace("\n", " "),
        "logical_cpu_count": os.cpu_count(),
        "training_parallelism": "CatBoost thread_count=20; LightGBM/XGBoost targets fitted sequentially with 20 threads per target",
        "selection_rule": "lowest grouped-validation operational score = mean target-wise IQR-normalized RMSE + 0.5*(1-feasibility balanced accuracy)",
        "selected_family": best_family,
        "selected_configs": selected_configs,
        "train_samples": int(np.sum(masks["train"])),
        "validation_samples": int(np.sum(masks["validation"])),
        "internal_test_samples": int(np.sum(masks["test"])),
        "external_test_samples": int(np.sum(masks["external_test"])),
        "wall_time_s": time.perf_counter() - started,
    }
    (args.data_dir / "surrogate_training_metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
