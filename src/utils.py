"""General helpers shared across task scripts."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


def get_task_config(
    tasks: list[dict[str, Any]],
    dataset_name: str,
    kind: str | None = None,
) -> dict[str, Any]:
    for task_config in tasks:
        if task_config["name"] == dataset_name:
            if kind is not None and task_config["kind"] != kind:
                raise ValueError(f"{dataset_name!r} is not a {kind} dataset.")
            return task_config

    available_tasks = (
        [task for task in tasks if kind is None]
        if kind is None
        else [task for task in tasks if task["kind"] == kind]
    )
    available = ", ".join(task["name"] for task in available_tasks)
    dataset_type = "dataset" if kind is None else f"{kind} dataset"
    raise ValueError(f"Unknown {dataset_type} {dataset_name!r}. Available: {available}")


def build_per_feature_values(
    explanation,
    X_explain: pd.DataFrame,
    y_explain: pd.Series,
    feature_names: list[str],
) -> pd.DataFrame:
    shap_values = np.asarray(explanation.values, dtype=float)
    if shap_values.ndim != 2:
        raise ValueError(f"Expected a 2D SHAP matrix, got shape {shap_values.shape}.")

    rows = []
    for explained_position, original_index in enumerate(X_explain.index):
        for feature_position, feature_name in enumerate(feature_names):
            rows.append(
                {
                    "explained_position": explained_position,
                    "row_index": original_index,
                    "target": y_explain.iloc[explained_position],
                    "feature_index": feature_position,
                    "feature_name": feature_name,
                    "feature_value": X_explain.iloc[explained_position, feature_position],
                    "shap_value": shap_values[explained_position, feature_position],
                }
            )

    return pd.DataFrame(rows)


def summarize_feature_values(per_feature_values: pd.DataFrame) -> pd.DataFrame:
    return (
        per_feature_values.groupby(["feature_index", "feature_name"], as_index=False)
        .agg(
            mean_shap_value=("shap_value", "mean"),
            mean_abs_shap_value=("shap_value", lambda values: values.abs().mean()),
            std_shap_value=("shap_value", "std"),
            min_shap_value=("shap_value", "min"),
            max_shap_value=("shap_value", "max"),
        )
        .fillna(0)
        .sort_values("mean_abs_shap_value", ascending=False)
    )


def get_ranked_features(feature_summary: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"feature_name", "mean_abs_shap_value"}
    missing_columns = required_columns - set(feature_summary.columns)
    if missing_columns:
        raise ValueError(
            "SHAP feature summary is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return feature_summary.sort_values("mean_abs_shap_value", ascending=False).reset_index(
        drop=True
    )


def load_or_create_feature_summary(
    summary_csv: Path,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_explain: pd.DataFrame,
    y_explain: pd.Series,
    feature_names: list[str],
    budget: int,
    output_dir: Path,
    compute_shap_explanation,
    log_message: str = "Computing Shapley values...",
) -> pd.DataFrame:
    from src.plot import save_shap_bar_plot

    output_dir.mkdir(parents=True, exist_ok=True)
    if summary_csv.exists():
        print(f"Using existing SHAP feature summary from {summary_csv}")
        return pd.read_csv(summary_csv)

    print(log_message)
    explanation = compute_shap_explanation(
        X_train=X_train,
        y_train=y_train,
        X_explain=X_explain,
        feature_names=feature_names,
        budget=budget,
    )
    per_feature_values = build_per_feature_values(
        explanation=explanation,
        X_explain=X_explain,
        y_explain=y_explain,
        feature_names=feature_names,
    )
    feature_summary = summarize_feature_values(per_feature_values)
    feature_summary.to_csv(summary_csv, index=False)
    plot_path = save_shap_bar_plot(explanation, output_dir)

    print(f"Wrote aggregate feature summary to {summary_csv}")
    if plot_path is not None:
        print(f"Wrote SHAP bar plot to {plot_path}")

    return feature_summary


def format_rate_label(rate: float) -> str:
    return f"{rate:.2f}".replace(".", "p")


def apply_missing_values(
    X: pd.DataFrame,
    rate: float,
    random_state: int,
) -> pd.DataFrame:
    if rate <= 0:
        return X.copy()

    rng = np.random.default_rng(random_state)
    missing_mask = rng.random(X.shape) < rate
    missing_mask = pd.DataFrame(missing_mask, index=X.index, columns=X.columns)
    X_missing = X.copy()
    categorical_columns = X_missing.select_dtypes(
        include=["object", "category", "bool"]
    ).columns
    numeric_columns = X_missing.columns.difference(categorical_columns)

    if len(numeric_columns) > 0:
        X_missing[numeric_columns] = X_missing[numeric_columns].mask(
            missing_mask[numeric_columns]
        )

    if len(categorical_columns) > 0:
        categorical_values = X_missing[categorical_columns].astype("object")
        X_missing[categorical_columns] = categorical_values.mask(
            missing_mask[categorical_columns],
            "__missing__",
        )

    return X_missing
