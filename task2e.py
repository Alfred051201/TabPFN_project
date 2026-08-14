"""Task 2e: robustness to messy data for TabPFN.

It varies the percentage of missing values shown to TabPFN while holding the test split fixed
per seed, then records accuracy, ROC AUC, log loss, and latency.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from TabPFN_project.src.dataset import TASKS, load_openml_data


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TabPFN context-size vs accuracy/latency experiment."
    )
    parser.add_argument(
        "--dataset",
        default="credit-g",
        help="Classification dataset name from dataset.py TASKS.",
    )
    parser.add_argument(
        "--missing-percentage",
        type=int,
        nargs="+",
        default=[50, 100, 200, 300, 400, 500, 600, 700, 800],
        help="Training-context sizes to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Random seeds for repeated train/test splits and context samples.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of rows held out for testing.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=SCRIPT_DIR / "results" / "task2a" / "context_size_results.json",
        help="Path where raw and summarized results will be written.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=SCRIPT_DIR / "results" / "task2a",
        help="Directory where context-size plots will be written.",
    )
    return parser.parse_args()


def get_task_config(dataset_name: str) -> dict[str, Any]:
    for task_config in TASKS:
        if task_config["name"] == dataset_name:
            if task_config["kind"] != "classification":
                raise ValueError(f"{dataset_name!r} is not a classification dataset.")
            return task_config
    available = ", ".join(task["name"] for task in TASKS if task["kind"] == "classification")
    raise ValueError(f"Unknown classification dataset {dataset_name!r}. Available: {available}")


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
        encoded_X = pd.DataFrame(
            ordinal_encoder.fit_transform(categorical_X),
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


def preprocess_target(y: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    label_encoder = LabelEncoder()
    encoded_y = pd.Series(label_encoder.fit_transform(y), index=y.index, name="target")
    return encoded_y, {"target_classes": label_encoder.classes_.tolist()}


def build_tabpfn_classifier(random_state: int):
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(random_state=random_state)


def normalize_probabilities(y_pred_proba: np.ndarray) -> np.ndarray:
    y_pred_proba = np.clip(y_pred_proba, 1e-15, 1.0)
    return y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)


def run_missing_value_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    missing_value: list[int],
    seeds: list[int],
    test_size: float = 0.20,
) -> pd.DataFrame:
    """Sample different training-context sizes and measure quality plus latency."""

    rows = []
    y_array = np.asarray(y)

    for seed in seeds:
        X_train_pool, X_test, y_train_pool, y_test = train_test_split(
            X,
            y_array,
            test_size=test_size,
            random_state=seed,
            stratify=y_array,
        )
        y_train_pool = np.asarray(y_train_pool)
        y_test = np.asarray(y_test)
        rng = np.random.default_rng(seed)

        for context_size in context_sizes:
            if context_size > len(X_train_pool):
                raise ValueError(
                    f"context_size={context_size} is larger than the available "
                    f"training pool of {len(X_train_pool)} rows"
                )

            sample_idx = rng.choice(len(X_train_pool), size=context_size, replace=False)
            X_train_context = X_train_pool.iloc[sample_idx]
            y_train_context = y_train_pool[sample_idx]

            model = build_tabpfn_classifier(seed)

            train_start = time.perf_counter()
            model.fit(X_train_context, y_train_context)
            train_end = time.perf_counter()

            predict_start = time.perf_counter()
            y_pred_proba = normalize_probabilities(
                np.asarray(model.predict_proba(X_test), dtype=float)
            )
            predict_end = time.perf_counter()

            y_pred = model.classes_[np.argmax(y_pred_proba, axis=1)]
            train_latency = train_end - train_start
            predict_latency = predict_end - predict_start

            rows.append(
                {
                    "seed": seed,
                    "training_rows": context_size,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "roc_auc": roc_auc_score(y_test, y_pred_proba[:, 1]),
                    "log_loss": log_loss(y_test, y_pred_proba),
                    "train_latency_seconds": train_latency,
                    "predict_latency_seconds": predict_latency,
                    "end_to_end_latency_seconds": train_latency + predict_latency,
                    "test_rows": len(X_test),
                }
            )

    return pd.DataFrame(rows)


def summarize_context_results(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby("training_rows")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_sem=("accuracy", "sem"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_sem=("roc_auc", "sem"),
            log_loss_mean=("log_loss", "mean"),
            log_loss_sem=("log_loss", "sem"),
            train_latency_mean=("train_latency_seconds", "mean"),
            train_latency_sem=("train_latency_seconds", "sem"),
            predict_latency_mean=("predict_latency_seconds", "mean"),
            predict_latency_sem=("predict_latency_seconds", "sem"),
            end_to_end_latency_mean=("end_to_end_latency_seconds", "mean"),
            end_to_end_latency_sem=("end_to_end_latency_seconds", "sem"),
        )
        .reset_index()
        .fillna(0)
    )


def configure_matplotlib_cache() -> None:
    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "tabpfn_matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))


def plot_context_summary(summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    configure_matplotlib_cache()

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot_specs = [
        {
            "title": "Accuracy vs Context Size",
            "mean_col": "accuracy_mean",
            "sem_col": "accuracy_sem",
            "ylabel": "Accuracy",
            "filename": "accuracy_vs_context_size.png",
        },
        {
            "title": "ROC AUC vs Context Size",
            "mean_col": "roc_auc_mean",
            "sem_col": "roc_auc_sem",
            "ylabel": "ROC AUC",
            "filename": "roc_auc_vs_context_size.png",
        },
        {
            "title": "Log Loss vs Context Size",
            "mean_col": "log_loss_mean",
            "sem_col": "log_loss_sem",
            "ylabel": "Log Loss",
            "filename": "log_loss_vs_context_size.png",
        },
        {
            "title": "Train Latency vs Context Size",
            "mean_col": "train_latency_mean",
            "sem_col": "train_latency_sem",
            "ylabel": "Seconds",
            "filename": "train_latency_vs_context_size.png",
        },
        {
            "title": "Predict Latency vs Context Size",
            "mean_col": "predict_latency_mean",
            "sem_col": "predict_latency_sem",
            "ylabel": "Seconds",
            "filename": "predict_latency_vs_context_size.png",
        },
        {
            "title": "End-to-End Latency vs Context Size",
            "mean_col": "end_to_end_latency_mean",
            "sem_col": "end_to_end_latency_sem",
            "ylabel": "Seconds",
            "filename": "end_to_end_latency_vs_context_size.png",
        },
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for spec in plot_specs:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(
            summary["training_rows"],
            summary[spec["mean_col"]],
            yerr=summary[spec["sem_col"]],
            marker="o",
            linewidth=2,
            capsize=4,
        )
        ax.set_title(spec["title"])
        ax.set_xlabel("Training rows sampled")
        ax.set_ylabel(spec["ylabel"])
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        plot_path = output_dir / spec["filename"]
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(plot_path)

    return saved_paths


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


def main() -> None:
    args = parse_args()

    task_config = get_task_config(args.dataset)
    X, y, openml_metadata = load_openml_data(task_config)
    X_processed, feature_metadata = preprocess_features(X)
    y_processed, target_metadata = preprocess_target(y)

    raw_results = run_missing_value_experiment(
        X_processed,
        y_processed,
        missing_percentage=args.missing_percentage,
        seeds=args.seeds,
        test_size=args.test_size,
    )
    summary = summarize_context_results(raw_results)
    plot_paths = plot_context_summary(summary, args.plots_dir)

    output = {
        "metadata": {
            **task_config,
            **openml_metadata,
            "context_sizes": args.context_sizes,
            "seeds": args.seeds,
            "test_size": args.test_size,
            "n_rows": len(X_processed),
            "n_features": X_processed.shape[1],
            **feature_metadata,
            **target_metadata,
        },
        "raw_results": raw_results,
        "summary": summary,
        "plot_paths": plot_paths,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=json_ready)


if __name__ == "__main__":
    main()
