import argparse
from pathlib import Path
from typing import Any
import time
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from src.plot import plot_missing_value_performance
from src.metrics import regression_metrics
from src.utils import apply_missing_values, format_rate_label
from src.utils import get_ranked_features, load_or_create_feature_summary

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "task3"

from tabpfn import TabPFNRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank Medical insurance cost features by SHAP value to evaluate if TabPFN can identify dominant cost-driving features." \
            "Afterwards, mimic the real world scenario where scenarios of missing values often occur due to privacy concerns to test" \
            "if model is still able to derive important features amid incomplete information."
        )
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=400,
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
        "--missing-rates",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
        help="Fractions of training feature values to randomly replace with missing values.",
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
        from tabpfn_extensions import TabPFNRegressor
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

    clf = TabPFNRegressor(fit_mode="fit_with_cache")
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

def run_missing_value_experiment(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    missing_rates: list[float],
    random_state: int,
) -> pd.DataFrame:
    """Evaluate TabPFN regression under different training missing-value rates."""

    rows = []
    y_train = pd.to_numeric(y_train, errors="coerce")
    y_test = pd.to_numeric(y_test, errors="coerce")
    if y_train.isna().any():
        y_train = y_train.fillna(y_train.median())
    if y_test.isna().any():
        y_test = y_test.fillna(y_train.median())

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    for rate_index, rate in enumerate(missing_rates):
        X_messy = apply_missing_values(
            X_train,
            rate=rate,
            random_state=random_state + rate_index,
        )

        model = TabPFNRegressor(random_state=random_state)

        train_start = time.perf_counter()
        model.fit(X_messy, y_train)
        train_end = time.perf_counter()

        predict_start = time.perf_counter()
        y_pred = np.asarray(model.predict(X_test), dtype=float)
        predict_end = time.perf_counter()

        metrics = regression_metrics(y_test, y_pred)
        train_latency = train_end - train_start
        predict_latency = predict_end - predict_start

        rows.append(
            {
                "missing_percentage": rate,
                **metrics,
                "train_latency_seconds": train_latency,
                "predict_latency_seconds": predict_latency,
                "end_to_end_latency_seconds": train_latency + predict_latency,
                "test_rows": len(X_test),
            }
        )

    return pd.DataFrame(rows)

def main():
    args = parse_args()

    df = pd.read_csv(SCRIPT_DIR / "medical_insurance_dataset.csv")
    X, y = df.drop(["insurance_cost"], axis=1), df["insurance_cost"] 
    feature_names = X.columns.astype(str).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    _, X_explain, _, y_explain = train_test_split(
        X_train,
        y_train,
        test_size=args.n_explain,
        random_state=args.random_state,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    performance_csv = args.output_dir / "missing_value_performance.csv"

    for rate_index, rate in enumerate(args.missing_rates):
        rate_label = format_rate_label(rate)
        shap_output_dir = args.output_dir / f"missing_{rate_label}"
        summary_csv = args.output_dir / f"shap_feature_summary_missing_{rate_label}.csv"
        X_train_explain = apply_missing_values(
            X_train,
            rate=rate,
            random_state=args.random_state + rate_index,
        )
        X_explain_messy = apply_missing_values(
            X_explain,
            rate=rate,
            random_state=args.random_state + 10_000 + rate_index,
        )

        feature_summary = load_or_create_feature_summary(
            summary_csv=summary_csv,
            X_train=X_train_explain,
            y_train=y_train,
            X_explain=X_explain_messy,
            y_explain=y_explain,
            feature_names=feature_names,
            budget=args.budget,
            output_dir=shap_output_dir,
            compute_shap_explanation=compute_shap_explanation,
            log_message=(
                f"Computing Shapley values with {rate * 100:.1f}% "
                "missing values in X_explain..."
            ),
        )

        ranked_features = get_ranked_features(feature_summary)
        print(
            "\nFeature importance ranked by mean absolute SHAP value "
            f"({rate * 100:.1f}% missing):"
        )
        print(
            ranked_features[
                ["feature_name", "mean_abs_shap_value", "mean_shap_value"]
            ].to_string(index=False)
        )

    performance_results = run_missing_value_experiment(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        missing_rates=args.missing_rates,
        random_state=args.random_state,
    )
    performance_results.to_csv(performance_csv, index=False)
    plot_paths = plot_missing_value_performance(performance_results, args.output_dir)
    print(f"Wrote missing-value performance results to {performance_csv}")
    for plot_path in plot_paths:
        print(f"Wrote performance plot to {plot_path}")


if __name__ == "__main__":
    main()
