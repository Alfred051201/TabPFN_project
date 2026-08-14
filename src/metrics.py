"""Metric helpers shared across benchmark tasks."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    log_loss,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
    roc_auc_score,
)


def normalize_probabilities(y_pred_proba: np.ndarray) -> np.ndarray:
    y_pred_proba = np.clip(y_pred_proba, 1e-15, 1.0)
    return y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred_proba: np.ndarray,
) -> dict[str, float]:
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


def regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
