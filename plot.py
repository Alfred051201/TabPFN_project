"""Plotting utilities for Task 1 benchmark results."""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def configure_matplotlib_cache() -> None:
    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "tabpfn_matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))


def get_summary_rows(task_result: dict[str, Any]) -> list[dict[str, Any]]:
    summary = task_result["cross_validation_summary"]
    if isinstance(summary, pd.DataFrame):
        return summary.to_dict(orient="records")
    return summary


def get_cv_train_rows(task_result: dict[str, Any], results: dict[str, Any]) -> int:
    metadata = task_result["metadata"]
    cv_splits = metadata.get("cv_splits", results["metadata"].get("cv_splits", 2))
    return int(round(metadata["n_rows"] * (cv_splits - 1) / cv_splits))


def plot_log_loss_by_training_rows(
    results: dict[str, Any],
    output_path: Path,
    title: str = "Classification log loss by training rows",
) -> None:
    """Plot classification CV log loss against train-set size for each model."""

    configure_matplotlib_cache()

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    colors = {
        "TabPFN": "#7b2cbf",
        "XGBoost": "#2fbf71",
        "RandomForest": "#e88c2a",
        "CatBoost": "#8a9b0f",
        "Logistic Regression": "#3498ff",
    }

    model_points: dict[str, list[dict[str, float]]] = {}
    for task_result in results["task_results"]:
        metadata = task_result["metadata"]
        if metadata["kind"] != "classification":
            continue

        train_rows = get_cv_train_rows(task_result, results)
        for row in get_summary_rows(task_result):
            if "log_loss_mean" not in row:
                continue
            model_points.setdefault(row["model"], []).append(
                {
                    "train_rows": train_rows,
                    "mean": row["log_loss_mean"],
                    "std": row["log_loss_std"],
                }
            )

    if not model_points:
        raise ValueError("No classification log-loss results are available to plot.")

    fig, ax = plt.subplots(figsize=(6.5, 3.2))

    for name, points in model_points.items():
        sorted_points = sorted(points, key=lambda item: item["train_rows"])
        x = np.array([point["train_rows"] for point in sorted_points], dtype=float)
        mean = np.array([point["mean"] for point in sorted_points], dtype=float)
        std = np.array([point["std"] for point in sorted_points], dtype=float)
        color = colors.get(name, None)

        ax.plot(x, mean, color=color, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.22)
        ax.text(x[0] * 1.05, mean[0], name, color=color, fontsize=8, va="center")
        ax.axhline(
            mean[-1],
            color=color,
            linestyle=(0, (1, 3)),
            linewidth=1.5,
            alpha=0.8,
        )

    all_x = np.array(
        [
            point["train_rows"]
            for points in model_points.values()
            for point in points
        ],
        dtype=float,
    )
    all_y = np.array(
        [point["mean"] for points in model_points.values() for point in points],
        dtype=float,
    )

    ax.set_xscale("log")
    ax.set_xlim(all_x.min() * 0.8, all_x.max() * 1.2)
    ax.set_ylim(max(0.0, all_y.min() * 0.85), all_y.max() * 1.15)
    ax.set_xlabel("Number of training rows")
    ax.set_ylabel("Cross-validation log loss")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_roc_auc_by_training_rows(
    results: dict[str, Any],
    output_path: Path,
    title: str = "Classification ROC AUC by training rows",
) -> None:
    """Plot classification CV ROC AUC against train-set size for each model."""

    configure_matplotlib_cache()

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    colors = {
        "TabPFN": "#7b2cbf",
        "XGBoost": "#2fbf71",
        "RandomForest": "#e88c2a",
        "CatBoost": "#8a9b0f",
        "Logistic Regression": "#3498ff",
    }

    model_points: dict[str, list[dict[str, float]]] = {}
    for task_result in results["task_results"]:
        metadata = task_result["metadata"]
        if metadata["kind"] != "classification":
            continue

        train_rows = get_cv_train_rows(task_result, results)
        for row in get_summary_rows(task_result):
            if "roc_auc_mean" not in row:
                continue
            per_split = task_result["cross_validation_per_split"].get(row["model"], {})
            roc_auc_values = per_split.get("roc_auc", [])
            roc_auc_std = float(np.std(roc_auc_values)) if roc_auc_values else 0.0
            model_points.setdefault(row["model"], []).append(
                {
                    "train_rows": train_rows,
                    "mean": row["roc_auc_mean"],
                    "std": roc_auc_std,
                }
            )

    if not model_points:
        raise ValueError("No classification ROC AUC results are available to plot.")

    fig, ax = plt.subplots(figsize=(6.5, 3.2))

    for name, points in model_points.items():
        sorted_points = sorted(points, key=lambda item: item["train_rows"])
        x = np.array([point["train_rows"] for point in sorted_points], dtype=float)
        mean = np.array([point["mean"] for point in sorted_points], dtype=float)
        std = np.array([point["std"] for point in sorted_points], dtype=float)
        color = colors.get(name, None)

        ax.plot(x, mean, color=color, linewidth=2)
        ax.fill_between(
            x,
            np.clip(mean - std, 0.0, 1.0),
            np.clip(mean + std, 0.0, 1.0),
            color=color,
            alpha=0.22,
        )
        ax.text(x[0] * 1.05, mean[0], name, color=color, fontsize=8, va="center")
        ax.axhline(
            mean[-1],
            color=color,
            linestyle=(0, (1, 3)),
            linewidth=1.5,
            alpha=0.8,
        )

    all_x = np.array(
        [
            point["train_rows"]
            for points in model_points.values()
            for point in points
        ],
        dtype=float,
    )
    all_y = np.array(
        [point["mean"] for points in model_points.values() for point in points],
        dtype=float,
    )

    ax.set_xscale("log")
    ax.set_xlim(all_x.min() * 0.8, all_x.max() * 1.2)
    ax.set_ylim(max(0.0, all_y.min() - 0.08), min(1.0, all_y.max() + 0.08))
    ax.set_xlabel("Number of training rows")
    ax.set_ylabel("Cross-validation ROC AUC")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_r2_by_training_rows(
    results: dict[str, Any],
    output_path: Path,
    title: str = "Regression R2 by training rows",
) -> None:
    """Plot regression CV R2 against train-set size for each model."""

    configure_matplotlib_cache()

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    colors = {
        "TabPFN": "#7b2cbf",
        "XGBoost": "#2fbf71",
        "RandomForest": "#e88c2a",
        "CatBoost": "#8a9b0f",
        "Ridge Regression": "#3498ff",
    }

    model_points: dict[str, list[dict[str, float]]] = {}
    for task_result in results["task_results"]:
        metadata = task_result["metadata"]
        if metadata["kind"] != "regression":
            continue

        train_rows = get_cv_train_rows(task_result, results)
        for row in get_summary_rows(task_result):
            if "r2_mean" not in row:
                continue
            per_split = task_result["cross_validation_per_split"].get(row["model"], {})
            r2_values = per_split.get("r2", [])
            r2_std = float(np.std(r2_values)) if r2_values else 0.0
            model_points.setdefault(row["model"], []).append(
                {
                    "train_rows": train_rows,
                    "mean": row["r2_mean"],
                    "std": r2_std,
                }
            )

    if not model_points:
        raise ValueError("No regression R2 results are available to plot.")

    fig, ax = plt.subplots(figsize=(6.5, 3.2))

    for name, points in model_points.items():
        sorted_points = sorted(points, key=lambda item: item["train_rows"])
        x = np.array([point["train_rows"] for point in sorted_points], dtype=float)
        mean = np.array([point["mean"] for point in sorted_points], dtype=float)
        std = np.array([point["std"] for point in sorted_points], dtype=float)
        color = colors.get(name, None)

        ax.plot(x, mean, color=color, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.22)
        ax.text(x[0] * 1.05, mean[0], name, color=color, fontsize=8, va="center")
        ax.axhline(
            mean[-1],
            color=color,
            linestyle=(0, (1, 3)),
            linewidth=1.5,
            alpha=0.8,
        )

    all_x = np.array(
        [
            point["train_rows"]
            for points in model_points.values()
            for point in points
        ],
        dtype=float,
    )
    all_y = np.array(
        [point["mean"] for points in model_points.values() for point in points],
        dtype=float,
    )

    y_margin = max(0.05, (all_y.max() - all_y.min()) * 0.15)
    ax.set_xscale("log")
    ax.set_xlim(all_x.min() * 0.8, all_x.max() * 1.2)
    ax.set_ylim(all_y.min() - y_margin, min(1.0, all_y.max() + y_margin))
    ax.set_xlabel("Number of training rows")
    ax.set_ylabel("Cross-validation R2")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_latency_by_training_rows(
    results: dict[str, Any],
    output_path: Path,
    title: str = "Fit and predict latency by training rows",
) -> None:
    """Plot CV fit and predict/score latency against train-set size."""

    configure_matplotlib_cache()

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    colors = {
        "TabPFN": "#7b2cbf",
        "XGBoost": "#2fbf71",
        "RandomForest": "#e88c2a",
        "CatBoost": "#8a9b0f",
        "Logistic Regression": "#3498ff",
        "Ridge Regression": "#3498ff",
    }

    fit_points: dict[str, list[dict[str, float]]] = {}
    predict_points: dict[str, list[dict[str, float]]] = {}
    for task_result in results["task_results"]:
        train_rows = get_cv_train_rows(task_result, results)
        for row in get_summary_rows(task_result):
            model = row["model"]
            per_split = task_result["cross_validation_per_split"].get(model, {})
            fit_values = per_split.get("fit_time", [])
            score_values = per_split.get("score_time", [])

            fit_mean = row.get("train_latency_mean_seconds")
            if fit_mean is None and fit_values:
                fit_mean = float(np.mean(fit_values))

            score_mean = row.get("predict_score_latency_mean_seconds")
            if score_mean is None and score_values:
                score_mean = float(np.mean(score_values))

            if fit_mean is not None:
                fit_points.setdefault(model, []).append(
                    {
                        "train_rows": train_rows,
                        "mean": fit_mean,
                        "std": float(np.std(fit_values)) if fit_values else 0.0,
                    }
                )
            if score_mean is not None:
                predict_points.setdefault(model, []).append(
                    {
                        "train_rows": train_rows,
                        "mean": score_mean,
                        "std": float(np.std(score_values)) if score_values else 0.0,
                    }
                )

    if not fit_points and not predict_points:
        raise ValueError("No latency results are available to plot.")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharex=True)
    panels = [
        (axes[0], fit_points, "Fit latency", "CV fit time (seconds)"),
        (axes[1], predict_points, "Predict/score latency", "CV score time (seconds)"),
    ]

    for ax, model_points, panel_title, ylabel in panels:
        all_x = []
        all_y = []
        for name, points in model_points.items():
            sorted_points = sorted(points, key=lambda item: item["train_rows"])
            x = np.array([point["train_rows"] for point in sorted_points], dtype=float)
            mean = np.array([point["mean"] for point in sorted_points], dtype=float)
            std = np.array([point["std"] for point in sorted_points], dtype=float)
            color = colors.get(name, None)

            ax.plot(x, mean, color=color, linewidth=2)
            ax.fill_between(
                x,
                np.maximum(mean - std, 0.0),
                mean + std,
                color=color,
                alpha=0.22,
            )
            ax.text(x[0] * 1.05, mean[0], name, color=color, fontsize=7, va="center")
            all_x.extend(x.tolist())
            all_y.extend(mean.tolist())

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(panel_title)
        ax.set_xlabel("Number of training rows")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)
        if all_x:
            ax.set_xlim(min(all_x) * 0.8, max(all_x) * 1.2)
        if all_y:
            positive_y = [value for value in all_y if value > 0]
            if positive_y:
                ax.set_ylim(min(positive_y) * 0.7, max(positive_y) * 1.4)

    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
