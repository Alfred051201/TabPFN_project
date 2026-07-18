"""Compare local TabPFN with baselines on selected OpenML tasks.

The selected tasks include classification and regression datasets:
- credit-g (task 31)
- blood-transfusion-service-center (task 1464)
- segment (task 1468)
- phishing_websites (task 1496)
- Fashion-MNIST (task 146825)
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


import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    log_loss,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBClassifier, XGBRegressor

from dataset import TASKS, TaskKind, load_openml_data
from plot import (
    plot_latency_by_training_rows,
    plot_log_loss_by_training_rows,
    plot_r2_by_training_rows,
    plot_roc_auc_by_training_rows,
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
        default=SCRIPT_DIR / "task1_results.json",
        help="Path where metrics and metadata will be written as JSON.",
    )
    parser.add_argument(
        "--from-json",
        type=Path,
        default=None,
        help="Load existing results JSON and only generate requested plots.",
    )
    parser.add_argument(
        "--log-loss-plot",
        type=Path,
        default=None,
        help="Optional path for a log-loss-vs-training-rows plot.",
    )
    parser.add_argument(
        "--roc-auc-plot",
        type=Path,
        default=None,
        help="Optional path for a ROC-AUC-vs-training-rows plot.",
    )
    parser.add_argument(
        "--r2-plot",
        type=Path,
        default=None,
        help="Optional path for an R2-vs-training-rows plot for regression datasets.",
    )
    parser.add_argument(
        "--latency-plot",
        type=Path,
        default=None,
        help="Optional path for fit/predict latency vs training rows plot.",
    )
    return parser.parse_args()


def preprocess_features(X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    categorical_columns = X.select_dtypes(include=["object", "category", "bool"]).columns
    numeric_columns = X.drop(columns=categorical_columns).columns

    parts = []
    if len(numeric_columns) > 0:
        numeric_X = X[numeric_columns].apply(pd.to_numeric, errors="coerce")
        numeric_X = numeric_X.fillna(numeric_X.median(numeric_only=True))
        numeric_X = numeric_X.fillna(0)
        parts.append(numeric_X)

    if len(categorical_columns) > 0:
        categorical_X = X[categorical_columns].astype("string").fillna("__missing__")
        ordinal_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        encoded_data = ordinal_encoder.fit_transform(categorical_X)
        encoded_X = pd.DataFrame(
            encoded_data,
            columns=categorical_columns,
            index=X.index,
        )
        parts.append(encoded_X)

    final_X = pd.concat(parts, axis=1).astype(float)
    metadata = {
        "categorical_columns": categorical_columns.tolist(),
        "numeric_columns": numeric_columns.tolist(),
    }
    return final_X, metadata


def preprocess_target(
    y: pd.Series,
    kind: TaskKind,
) -> tuple[pd.Series, dict[str, Any]]:
    if kind == "classification":
        label_encoder = LabelEncoder()
        encoded_y = pd.Series(label_encoder.fit_transform(y), index=y.index, name="target")
        return encoded_y, {"target_classes": label_encoder.classes_.tolist()}

    numeric_y = pd.to_numeric(y, errors="coerce")
    if numeric_y.isna().any():
        numeric_y = numeric_y.fillna(numeric_y.median())
    return pd.Series(numeric_y, index=y.index, name="target"), {}


def classification_metrics(y_true: pd.Series, y_pred_proba: np.ndarray) -> dict[str, float]:
    labels = sorted(pd.unique(y_true))
    if len(labels) == 2:
        roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
    else:
        roc_auc = roc_auc_score(
            y_true,
            y_pred_proba,
            multi_class="ovr",
            average="macro",
        )
    return {
        "roc_auc": roc_auc,
        "log_loss": log_loss(y_true, y_pred_proba, labels=labels),
    }


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def evaluate_train_test(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    kind: TaskKind,
    random_state: int,
) -> dict[str, float | int]:
    stratify = y if kind == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state,
        stratify=stratify,
    )

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
            ("TabPFN", build_tabpfn_model(kind, random_state)),
            ("RandomForest", RandomForestClassifier(random_state=random_state)),
            ("XGBoost", XGBClassifier(random_state=random_state)),
            ("CatBoost", CatBoostClassifier(random_state=random_state, verbose=0)),
            (
                "Logistic Regression",
                LogisticRegression(
                    random_state=random_state,
                    max_iter=5000,
                    solver="saga",
                ),
            ),
        ]

    return [
        ("TabPFN", build_tabpfn_model(kind, random_state)),
        ("RandomForest", RandomForestRegressor(random_state=random_state)),
        ("XGBoost", XGBRegressor(random_state=random_state)),
        ("CatBoost", CatBoostRegressor(random_state=random_state, verbose=0)),
        ("Ridge Regression", Ridge(random_state=random_state)),
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
    else:
        cv = KFold(n_splits=cv_splits, random_state=random_state, shuffle=True)
        scoring = ["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"]

    summary_rows = []
    per_split_results = {}
    for name, model in models:
        results = cross_validate(
            model,
            X,
            y,
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
    X, y, openml_metadata = load_openml_data(task_config)
    final_X, feature_metadata = preprocess_features(X)
    encoded_y, target_metadata = preprocess_target(y, kind)

    train_test_metrics = evaluate_train_test(
        build_tabpfn_model(kind, args.random_state),
        final_X,
        encoded_y,
        kind,
        args.random_state,
    )

    comparison_df, per_split_results = compare_models(
        build_models(kind, args.random_state),
        final_X,
        encoded_y,
        kind,
        cv_splits=args.cv_splits,
        random_state=args.random_state,
    )

    return {
        "metadata": {
            **task_config,
            **openml_metadata,
            "random_state": args.random_state,
            "cv_splits": args.cv_splits,
            "n_rows": len(final_X),
            "n_features": final_X.shape[1],
            **feature_metadata,
            **target_metadata,
        },
        "tabpfn_train_test": train_test_metrics,
        "cross_validation_summary": comparison_df,
        "cross_validation_per_split": per_split_results,
    }


def json_ready(value):
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


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
        results["task_results"].append(run_task(task_config, args))
        write_results_json(results, args.output_json)

    write_requested_plots(results, args)
    write_results_json(results, args.output_json)


if __name__ == "__main__":
    main()
