"""Task 2a: context size vs accuracy and latency for TabPFN.

This script extracts the context-size experiment from task2.ipynb. It varies
the number of training rows shown to TabPFN while holding the test split fixed
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

from src.dataset import TASKS, load_openml_data
from src.plot import plot_context_summary
from src.metrics import normalize_probabilities
from src.utils import get_task_config
from src.utils import json_ready


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
        "--context-sizes",
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


def build_tabpfn_classifier(random_state: int):
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(random_state=random_state)


def run_context_size_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    context_sizes: list[int],
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


def main() -> None:
    args = parse_args()

    task_config = get_task_config(TASKS, args.dataset, kind="classification")
    X, y = load_openml_data(task_config)

    raw_results = run_context_size_experiment(
        X,
        y,
        context_sizes=args.context_sizes,
        seeds=args.seeds,
        test_size=args.test_size,
    )
    summary = summarize_context_results(raw_results)
    plot_paths = plot_context_summary(summary, args.plots_dir)

    output = {
        "metadata": {
            **task_config,
            "context_sizes": args.context_sizes,
            "seeds": args.seeds,
            "test_size": args.test_size,
            "n_rows": len(X),
            "n_features": X.shape[1],
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
