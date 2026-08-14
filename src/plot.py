"""Plotting utilities for benchmark task results."""

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


def save_shap_bar_plot(explanation, output_dir: Path) -> Path | None:
    configure_matplotlib_cache()

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import shap
    except ImportError:
        return None

    shap.plots.bar(explanation, show=False)
    plot_path = output_dir / "shap_feature_bar.png"
    plt.gcf().savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    return plot_path


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


def plot_missing_value_performance(results: pd.DataFrame, output_dir: Path) -> list[Path]:
    configure_matplotlib_cache()

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    metric_specs = [
        {
            "metric": "rmse",
            "ylabel": "RMSE",
            "title": "RMSE",
            "filename_prefix": "rmse",
        },
        {
            "metric": "mae",
            "ylabel": "MAE",
            "title": "MAE",
            "filename_prefix": "mae",
        },
        {
            "metric": "r2",
            "ylabel": "R2",
            "title": "R2",
            "filename_prefix": "r2",
        },
        {
            "metric": "train_latency_seconds",
            "ylabel": "Seconds",
            "title": "Train Latency",
            "filename_prefix": "train_latency",
        },
        {
            "metric": "predict_latency_seconds",
            "ylabel": "Seconds",
            "title": "Predict Latency",
            "filename_prefix": "predict_latency",
        },
        {
            "metric": "end_to_end_latency_seconds",
            "ylabel": "Seconds",
            "title": "End-to-End Latency",
            "filename_prefix": "end_to_end_latency",
        },
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    plot_results = results.sort_values("missing_percentage")
    for metric_spec in metric_specs:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(
            plot_results["missing_percentage"] * 100,
            plot_results[metric_spec["metric"]],
            marker="o",
            linewidth=2,
        )
        ax.set_title(f"{metric_spec['title']} vs Missing Values")
        ax.set_xlabel("Training missing values (%)")
        ax.set_ylabel(metric_spec["ylabel"])
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        plot_path = output_dir / f"{metric_spec['filename_prefix']}_vs_missing_values.png"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(plot_path)

    return saved_paths


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
        "Logistic Regression": "#d6278b",
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

        ax.plot(x, mean, color=color, linewidth=2, label=name)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.22)
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
    ax.legend(loc="best", fontsize=8, frameon=False)

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
        "Logistic Regression": "#d6278b",
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

        ax.plot(x, mean, color=color, linewidth=2, label=name)
        ax.fill_between(
            x,
            np.clip(mean - std, 0.0, 1.0),
            np.clip(mean + std, 0.0, 1.0),
            color=color,
            alpha=0.22,
        )
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
    ax.legend(loc="best", fontsize=8, frameon=False)

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
        "Ridge Regression": "#1f77b4",
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

        ax.plot(x, mean, color=color, linewidth=2, label=name)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.22)
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
    ax.legend(loc="best", fontsize=8, frameon=False)

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
        "Logistic Regression": "#d6278b",
        "Ridge Regression": "#1f77b4",
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

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), sharex=True)
    panels = [
        (axes[0], fit_points, "Fit latency"),
        (axes[1], predict_points, "Predict latency"),
    ]
    legend_handles = {}

    for ax, model_points, panel_title in panels:
        all_x = []
        all_y = []
        for name, points in model_points.items():
            sorted_points = sorted(points, key=lambda item: item["train_rows"])
            x = np.array([point["train_rows"] for point in sorted_points], dtype=float)
            mean = np.array([point["mean"] for point in sorted_points], dtype=float)
            std = np.array([point["std"] for point in sorted_points], dtype=float)
            color = colors.get(name, None)

            (line,) = ax.plot(x, mean, color=color, linewidth=2, label=name)
            legend_handles.setdefault(name, line)
            ax.fill_between(
                x,
                np.maximum(mean - std, 0.0),
                mean + std,
                color=color,
                alpha=0.22,
            )
            all_x.extend(x.tolist())
            all_y.extend(mean.tolist())

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(panel_title)
        ax.set_xlabel("Number of training rows")
        ax.grid(True, which="both", alpha=0.25)
        if all_x:
            ax.set_xlim(min(all_x) * 0.8, max(all_x) * 1.2)
        if all_y:
            positive_y = [value for value in all_y if value > 0]
            if positive_y:
                ax.set_ylim(min(positive_y) * 0.7, max(positive_y) * 1.4)

    fig.suptitle(title)
    fig.supylabel("CV latency (seconds)")
    if legend_handles:
        fig.legend(
            legend_handles.values(),
            legend_handles.keys(),
            loc="center left",
            bbox_to_anchor=(0.86, 0.5),
            fontsize=8,
            frameon=False,
            title="Model",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.03, 0.0, 0.84, 0.94))
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
