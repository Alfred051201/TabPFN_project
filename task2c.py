"""Task 2c: feature dimensionality vs TabPFN performance on credit-g.

This script follows the SHAP workflow from task2c.ipynb, ranks credit-g
features by mean absolute SHAP value, then varies the number of top-ranked
features used to fit TabPFN.

Feature counts 21-25 append synthetic noise columns after the 20 real features:
uniform [-1, 1], normal mean 0/std 1, binary 0/1, sparse binary with p=0.05,
and heavy-tailed Student-t noise with df=2. Feature-count plots include these
noise features, while SHAP-value plots are capped at the 20 real features.
"""

import argparse
import os
import time
import warnings
from pathlib import Path
from typing import Any


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but TabPFNClassifier was fitted with feature names",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Feature .* is not numerical\. Adding it to categorical features\.",
    category=UserWarning,
)


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from src.dataset import TASKS, load_openml_data
from src.metrics import normalize_probabilities
from src.utils import get_ranked_features, get_task_config
from src.utils import load_or_create_feature_summary


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "task2c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank credit-g features by SHAP value, then measure TabPFN "
            "performance and latency as feature dimensionality increases."
        )
    )
    parser.add_argument(
        "--dataset",
        default="credit-g",
        help="Classification dataset name from dataset.py TASKS.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=200,
        help="Number of credit-g rows held out for evaluation.",
    )
    parser.add_argument(
        "--n-explain",
        type=int,
        default=50,
        help="Number of held-out rows to explain when SHAP summary is missing.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=256,
        help="shapiq approximation budget passed to shapiq_to_shap_explanation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random state for train/test split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where SHAP summary, performance CSV, and plots are written.",
    )
    parser.add_argument(
        "--feature-counts",
        type=int,
        nargs="+",
        default=[4, 6, 8, 10, 12, 15, 20, 21, 22, 23, 24, 25],
        help="Top-k SHAP-ranked feature counts to evaluate.",
    )
    return parser.parse_args()


def compute_shap_explanation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_explain: pd.DataFrame,
    feature_names: list[str],
    budget: int,
):
    try:
        from tabpfn_extensions import TabPFNClassifier
        from tabpfn_extensions.interpretability import (
            shapiq,
            shapiq_to_shap_explanation,
        )
    except ImportError as exc:
        raise ImportError(
            "task2c.py requires tabpfn-extensions for SHAP values. "
            "Install it in this environment, then rerun the script."
        ) from exc

    X_train_array = X_train.to_numpy()
    X_explain_array = X_explain.to_numpy()

    clf = TabPFNClassifier(fit_mode="fit_with_cache")
    clf.fit(X_train_array, y_train)

    explainer = shapiq.get_tabpfn_imputation_explainer(
        model=clf,
        data=X_train_array,
        index="SV",
        max_order=1,
    )

    return shapiq_to_shap_explanation(
        explainer,
        X_explain_array,
        budget=budget,
        feature_names=feature_names,
    )


def build_tabpfn_classifier():
    try:
        from tabpfn_extensions import TabPFNClassifier
    except ImportError as exc:
        raise ImportError(
            "task2c.py requires tabpfn-extensions to fit TabPFN on credit-g. "
            "Install it in this environment, then rerun the script."
        ) from exc

    return TabPFNClassifier(fit_mode="fit_with_cache")


def run_feature_count_experiment(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    ranked_features: pd.DataFrame,
    feature_counts: list[int],
    random_state: int,
) -> pd.DataFrame:
    rows = []
    available_features = ranked_features["feature_name"].tolist()
    extra_feature_generators = {
        "synthetic_uniform_feature": lambda rng, size: rng.uniform(
            -1.0,
            1.0,
            size=size,
        ),
        "synthetic_normal_feature": lambda rng, size: rng.normal(
            loc=0.0,
            scale=1.0,
            size=size,
        ),
        "synthetic_binary_feature": lambda rng, size: rng.integers(
            0,
            2,
            size=size,
        ),
        "synthetic_sparse_binary_feature": lambda rng, size: rng.binomial(
            1,
            0.05,
            size=size,
        ),
        "synthetic_heavy_tailed_feature": lambda rng, size: rng.standard_t(
            df=2,
            size=size,
        ),
    }
    max_feature_count = len(available_features) + len(extra_feature_generators)
    valid_feature_counts = sorted(
        feature_count
        for feature_count in set(feature_counts)
        if feature_count <= max_feature_count
    )
    if not valid_feature_counts:
        raise ValueError(
            "No feature counts are valid for "
            f"{len(available_features)} available features plus "
            f"{len(extra_feature_generators)} synthetic features."
        )

    rng = np.random.default_rng(random_state)
    extra_feature_names = list(extra_feature_generators)
    for feature_count in valid_feature_counts:
        real_feature_count = min(feature_count, len(available_features))
        extra_feature_count = max(0, feature_count - len(available_features))
        selected_features = available_features[:real_feature_count]
        selected_summary = ranked_features.iloc[:real_feature_count]
        X_train_subset = X_train[selected_features].to_numpy()
        X_test_subset = X_test[selected_features].to_numpy()

        for extra_feature_name in extra_feature_names[:extra_feature_count]:
            generator = extra_feature_generators[extra_feature_name]
            X_train_random = generator(rng, size=(len(X_train_subset), 1))
            X_test_random = generator(rng, size=(len(X_test_subset), 1))
            X_train_subset = np.column_stack([X_train_subset, X_train_random])
            X_test_subset = np.column_stack([X_test_subset, X_test_random])
            selected_features = [*selected_features, extra_feature_name]

        model = build_tabpfn_classifier()

        train_start = time.perf_counter()
        model.fit(X_train_subset, y_train)
        train_end = time.perf_counter()

        predict_start = time.perf_counter()
        y_pred_proba = normalize_probabilities(
            np.asarray(model.predict_proba(X_test_subset), dtype=float)
        )
        predict_end = time.perf_counter()

        y_pred = model.classes_[np.argmax(y_pred_proba, axis=1)]
        train_latency = train_end - train_start
        predict_latency = predict_end - predict_start

        rows.append(
            {
                "feature_count": feature_count,
                "accuracy": accuracy_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_pred_proba[:, 1]),
                "log_loss": log_loss(y_test, y_pred_proba),
                "train_latency_seconds": train_latency,
                "predict_latency_seconds": predict_latency,
                "end_to_end_latency_seconds": train_latency + predict_latency,
                "cumulative_mean_abs_shap_value": selected_summary[
                    "mean_abs_shap_value"
                ].sum(),
                "mean_abs_shap_value_at_k": (
                    0.0
                    if extra_feature_count > 0
                    else selected_summary["mean_abs_shap_value"].iloc[-1]
                ),
                "selected_features": ",".join(selected_features),
            }
        )

    return pd.DataFrame(rows)


def plot_feature_performance(results: pd.DataFrame, output_dir: Path) -> list[Path]:

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    metric_specs = [
        {
            "metric": "accuracy",
            "ylabel": "Accuracy",
            "title": "Accuracy",
            "filename_prefix": "accuracy",
        },
        {
            "metric": "roc_auc",
            "ylabel": "ROC AUC",
            "title": "ROC AUC",
            "filename_prefix": "roc_auc",
        },
        {
            "metric": "log_loss",
            "ylabel": "Log Loss",
            "title": "Log Loss",
            "filename_prefix": "log_loss",
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
    x_specs = [
        {
            "column": "feature_count",
            "xlabel": "Number of Top SHAP-Ranked Features",
            "filename_suffix": "feature_count",
            "title_suffix": "Feature Count",
        },
        {
            "column": "cumulative_mean_abs_shap_value",
            "xlabel": "Cumulative Mean Absolute SHAP Value",
            "filename_suffix": "shap_value",
            "title_suffix": "Cumulative SHAP Value",
        },
    ]

    saved_paths = []
    for metric_spec in metric_specs:
        for x_spec in x_specs:
            plot_results = results
            if x_spec["filename_suffix"] == "feature_count":
                plot_results = results[results["feature_count"] <= 25]
            elif x_spec["filename_suffix"] == "shap_value":
                plot_results = results[results["feature_count"] <= 20]

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(
                plot_results[x_spec["column"]],
                plot_results[metric_spec["metric"]],
                marker="o",
                linewidth=2,
            )
            ax.set_title(f"{metric_spec['title']} vs {x_spec['title_suffix']}")
            ax.set_xlabel(x_spec["xlabel"])
            ax.set_ylabel(metric_spec["ylabel"])
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            plot_path = (
                output_dir
                / f"{metric_spec['filename_prefix']}_vs_{x_spec['filename_suffix']}.png"
            )
            fig.savefig(plot_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(plot_path)

    return saved_paths


def main() -> None:
    args = parse_args()

    task_config = get_task_config(TASKS, args.dataset, kind="classification")
    X, y  = load_openml_data(task_config)
    feature_names = X.columns.astype(str).tolist()

    X_train_pool, X_test, y_train_pool, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    X_train, X_explain, y_train, y_explain = train_test_split(
        X_train_pool,
        y_train_pool,
        test_size=args.n_explain,
        random_state=args.random_state,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_dir / "shap_feature_summary.csv"
    performance_csv = args.output_dir / "feature_count_performance.csv"

    feature_summary = load_or_create_feature_summary(
        summary_csv=summary_csv,
        X_train=X_train,
        y_train=y_train,
        X_explain=X_explain,
        y_explain=y_explain,
        feature_names=feature_names,
        budget=args.budget,
        output_dir=args.output_dir,
        compute_shap_explanation=compute_shap_explanation,
        log_message=f"Computing Shapley values for {len(X_explain)} credit-g rows...",
    )
    ranked_features = get_ranked_features(feature_summary)
    performance_results = run_feature_count_experiment(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        ranked_features=ranked_features,
        feature_counts=args.feature_counts,
        random_state=args.random_state,
    )
    performance_results.to_csv(performance_csv, index=False)
    plot_paths = plot_feature_performance(performance_results, args.output_dir)

    print(f"Wrote feature-count performance results to {performance_csv}")
    for plot_path in plot_paths:
        print(f"Wrote performance plot to {plot_path}")


if __name__ == "__main__":
    main()
