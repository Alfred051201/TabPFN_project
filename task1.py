"""Compare local TabPFN with baselines on selected OpenML tasks.

The selected tasks include classification and regression datasets:
- credit-g (task 31)
- blood-transfusion-service-center (task 1464)
- segment (task 1468)
- phishing_websites (task 1496)
- optdigits (task 28)
- cpu_act (task 361099)
- abalone (task 361077)
- houses (task 361089)
- pol (task 361110)
- elevators (task 361084)

All results are written to JSON. The script intentionally does not print
progress or metrics to stdout.
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBClassifier, XGBRegressor

from src.dataset import TASKS, TaskKind, load_openml_data
from src.plot import (
    plot_latency_by_training_rows,
    plot_log_loss_by_training_rows,
    plot_r2_by_training_rows,
    plot_roc_auc_by_training_rows,
)
from src.metrics import classification_metrics, regression_metrics
from src.utils import json_ready


column_transformer = make_column_transformer(
    (
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        make_column_selector(dtype_include=["object", "category"]),
    ),
    remainder="passthrough",
)

SCRIPT_DIR = Path(__file__).resolve().parent
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local TabPFN and baseline models on selected OpenML tasks."
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=3,
        help="Number of CV splits for model comparison.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for train/test split and model initialization.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=SCRIPT_DIR / "results/task1/task1_results.json",
        help="Path where metrics and metadata will be written as JSON.",
    )
    parser.add_argument(
        "--from-json",
        type=Path,
        default=SCRIPT_DIR / "results/task1/task1_results.json",
        help="Load existing results JSON and only generate requested plots.",
    )
    parser.add_argument(
        "--log-loss-plot",
        type=Path,
        default=SCRIPT_DIR / "results/task1/log_loss_by_training_rows.png",
        help="Optional path for a log-loss-vs-training-rows plot.",
    )
    parser.add_argument(
        "--roc-auc-plot",
        type=Path,
        default=SCRIPT_DIR / "results/task1/roc_auc_by_training_rows.png",
        help="Optional path for a ROC-AUC-vs-training-rows plot.",
    )
    parser.add_argument(
        "--r2-plot",
        type=Path,
        default=SCRIPT_DIR / "results/task1/r2_by_training_rows.png",
        help="Optional path for an R2-vs-training-rows plot for regression datasets.",
    )
    parser.add_argument(
        "--latency-plot",
        type=Path,
        default=SCRIPT_DIR / "results/task1/latency_by_training_rows.png",
        help="Optional path for fit/predict latency vs training rows plot.",
    )
    return parser.parse_args()


def evaluate_train_test(
    model: object,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    kind: TaskKind,
    random_state: int,
) -> dict[str, float | int]:

    train_start = time.perf_counter()
    model.fit(X_train, y_train)
    train_end = time.perf_counter()

    predict_start = time.perf_counter()
    if kind == "classification":
        predictions = model.predict_proba(X_test)
    else:
        predictions = model.predict(X_test)
    predict_end = time.perf_counter()

    metrics = (
        classification_metrics(y_test, predictions)
        if kind == "classification"
        else regression_metrics(y_test, predictions)
    )
    return {
        **metrics,
        "train_latency_seconds": train_end - train_start,
        "predict_latency_seconds": predict_end - predict_start,
        "end_to_end_latency_seconds": predict_end - train_start,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }


def build_tabpfn_model(kind: TaskKind, random_state: int):
    if kind == "classification":
        from tabpfn import TabPFNClassifier

        return TabPFNClassifier(random_state=random_state)

    from tabpfn import TabPFNRegressor

    return TabPFNRegressor(random_state=random_state)


def build_models(kind: TaskKind, random_state: int) -> list[tuple[str, object]]:
    if kind == "classification":
        return [
            (
                "TabPFN",
                build_tabpfn_model(kind, random_state),
            ),
            (
                "RandomForest",
                make_pipeline(
                    column_transformer,
                    RandomForestClassifier(random_state=random_state),
                ),
            ),
            (
                "XGBoost",
                make_pipeline(
                    column_transformer,
                    XGBClassifier(random_state=random_state),
                ),
            ),
            (
                "CatBoost",
                make_pipeline(
                    column_transformer,
                    CatBoostClassifier(random_state=random_state, verbose=0),
                ),
            ),
            (
                "Logistic Regression",
                make_pipeline(
                    column_transformer,
                    StandardScaler(),
                    LogisticRegression(
                        random_state=random_state,
                        max_iter=5000,
                        solver="saga",
                    ),
                ),
            ),
        ]

    return [
        (
            "TabPFN",
            make_pipeline(
                column_transformer,
                build_tabpfn_model(kind, random_state),
            ),
        ),
        (
            "RandomForest",
            make_pipeline(
                column_transformer,
                RandomForestRegressor(random_state=random_state),
            ),
        ),
        (
            "XGBoost",
            make_pipeline(
                column_transformer,
                XGBRegressor(random_state=random_state),
            ),
        ),
        (
            "CatBoost",
            make_pipeline(
                column_transformer,
                CatBoostRegressor(random_state=random_state, verbose=0),
            ),
        ),
        (
            "Ridge Regression",
            make_pipeline(
                column_transformer,
                StandardScaler(),
                Ridge(random_state=random_state),
            ),
        ),
    ]


def compare_models(
    models: list[tuple[str, object]],
    X: pd.DataFrame,
    y: pd.Series,
    kind: TaskKind,
    cv_splits: int,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, list[float]]]]:
    if kind == "classification":
        cv = StratifiedKFold(n_splits=cv_splits, random_state=random_state, shuffle=True)
        scoring = ["neg_log_loss", "roc_auc_ovr"]
        y_for_cv = LabelEncoder().fit_transform(y)
    else:
        cv = KFold(n_splits=cv_splits, random_state=random_state, shuffle=True)
        scoring = ["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"]
        y_for_cv = pd.to_numeric(y, errors="coerce")
        if y_for_cv.isna().any():
            y_for_cv = y_for_cv.fillna(y_for_cv.median())

    summary_rows = []
    per_split_results = {}
    for name, model in models:
        results = cross_validate(
            model,
            X,
            y_for_cv,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            verbose=0,
            return_train_score=False,
        )

        if kind == "classification":
            per_split_results[name] = {
                "fit_time": results["fit_time"].tolist(),
                "score_time": results["score_time"].tolist(),
                "roc_auc": results["test_roc_auc_ovr"].tolist(),
                "log_loss": (-results["test_neg_log_loss"]).tolist(),
            }
            summary_rows.append(
                {
                    "model": name,
                    "roc_auc_mean": results["test_roc_auc_ovr"].mean(),
                    "log_loss_mean": -results["test_neg_log_loss"].mean(),
                    "log_loss_std": (-results["test_neg_log_loss"]).std(),
                    "train_latency_mean_seconds": results["fit_time"].mean(),
                    "predict_score_latency_mean_seconds": results["score_time"].mean(),
                    "end_to_end_cv_latency_seconds": results["fit_time"].sum()
                    + results["score_time"].sum(),
                }
            )
        else:
            per_split_results[name] = {
                "fit_time": results["fit_time"].tolist(),
                "score_time": results["score_time"].tolist(),
                "rmse": (-results["test_neg_root_mean_squared_error"]).tolist(),
                "mae": (-results["test_neg_mean_absolute_error"]).tolist(),
                "r2": results["test_r2"].tolist(),
            }
            summary_rows.append(
                {
                    "model": name,
                    "rmse_mean": -results["test_neg_root_mean_squared_error"].mean(),
                    "mae_mean": -results["test_neg_mean_absolute_error"].mean(),
                    "r2_mean": results["test_r2"].mean(),
                    "train_latency_mean_seconds": results["fit_time"].mean(),
                    "predict_score_latency_mean_seconds": results["score_time"].mean(),
                    "end_to_end_cv_latency_seconds": results["fit_time"].sum()
                    + results["score_time"].sum(),
                }
            )

    return pd.DataFrame(summary_rows), per_split_results


def run_task(task_config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    kind = task_config["kind"]
    X, y = load_openml_data(task_config)
    stratify = y if kind == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=args.random_state,
        stratify=stratify,
    )

    train_test_metrics = evaluate_train_test(
        build_tabpfn_model(kind, args.random_state),
        X_train,
        X_test,
        y_train,
        y_test,
        kind,
        args.random_state,
    )

    comparison_df, per_split_results = compare_models(
        build_models(kind, args.random_state),
        X,
        y,
        kind,
        cv_splits=args.cv_splits,
        random_state=args.random_state,
    )

    return {
        "metadata": {
            **task_config,
            "random_state": args.random_state,
            "cv_splits": args.cv_splits,
            "n_rows": len(X),
            "n_features": X.shape[1],
        },
        "tabpfn_train_test": train_test_metrics,
        "cross_validation_summary": comparison_df,
        "cross_validation_per_split": per_split_results,
    }


def write_results_json(results: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=json_ready)


def load_results_json(input_path: Path) -> dict[str, Any]:
    with input_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_requested_plots(results: dict[str, Any], args: argparse.Namespace) -> None:
    if args.log_loss_plot is not None:
        plot_log_loss_by_training_rows(results, args.log_loss_plot)

    if args.roc_auc_plot is not None:
        plot_roc_auc_by_training_rows(results, args.roc_auc_plot)

    if args.r2_plot is not None:
        plot_r2_by_training_rows(results, args.r2_plot)

    if args.latency_plot is not None:
        plot_latency_by_training_rows(results, args.latency_plot)


def main() -> None:
    args = parse_args()

    print(args.from_json)

    if args.from_json is not None:
        results = load_results_json(args.from_json)
        write_requested_plots(results, args)
        return

    results = {
        "metadata": {
            "tabpfn_backend": "local",
            "random_state": args.random_state,
            "cv_splits": args.cv_splits,
            "tasks": TASKS,
        },
        "task_results": [],
    }

    for task_config in TASKS:
        print(f'Working on {task_config["name"]}')
        results["task_results"].append(run_task(task_config, args))
        write_results_json(results, args.output_json)
        print(f'Finished {task_config["name"]}')

    write_requested_plots(results, args)
    write_results_json(results, args.output_json)


if __name__ == "__main__":
    main()
