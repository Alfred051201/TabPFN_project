"""Selected OpenML tasks and dataset loading utilities."""

import shutil
import time
from pathlib import Path
from typing import Any, Literal

import pandas as pd


TaskKind = Literal["classification", "regression"]
PROJECT_DIR = Path(__file__).resolve().parent
OPENML_CACHE_DIR = PROJECT_DIR / ".openml_cache"

TASKS = [
    {
        "name": "credit-g",
        "source": "task",
        "task_id": 31,
        "dataset_id": 31,
        "kind": "classification",
    },
    {
        "name": "blood-transfusion",
        "source": "dataset",
        "dataset_id": 1464,
        "target": "Class",
        "kind": "classification",
    },
    {
        "name": "segment",
        "source": "dataset",
        "dataset_id": 36,
        "target": "class",
        "kind": "classification",
        "requested_id": 1468,
    },
    {
        "name": "phishing_websites",
        "source": "dataset",
        "dataset_id": 4534,
        "target": "Result",
        "kind": "classification",
        "requested_id": 1496,
    },
    {
        "name": "optdigits",
        "source": "dataset",
        "dataset_id": 28,
        "target": "class",
        "kind": "classification",
        "replaces": "fashion-mnist",
    },
    {
        "name": "cpu_act",
        "source": "dataset",
        "dataset_id": 197,
        "target": "usr",
        "kind": "regression",
        "requested_id": 361099,
    },
    {
        "name": "abalone",
        "source": "dataset",
        "dataset_id": 42726,
        "target": "Class_number_of_rings",
        "kind": "regression",
        "requested_id": 361077,
    },
    {
        "name": "houses",
        "source": "dataset",
        "dataset_id": 537,
        "target": "median_house_value",
        "kind": "regression",
        "requested_id": 361089,
    },
    {
        "name": "pol",
        "source": "dataset",
        "dataset_id": 201,
        "target": "foo",
        "kind": "regression",
        "requested_id": 361110,
    },
    {
        "name": "elevators",
        "source": "dataset",
        "dataset_id": 216,
        "target": "Goal",
        "kind": "regression",
        "requested_id": 361084,
    },
]


def configure_openml_cache() -> None:
    import openml

    OPENML_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    openml.config.set_root_cache_directory(str(OPENML_CACHE_DIR))


def dataset_cache_dir(dataset_id: int) -> Path:
    return OPENML_CACHE_DIR / "org" / "openml" / "www" / "datasets" / str(dataset_id)


def clear_dataset_cache(dataset_id: int | None) -> None:
    if dataset_id is None:
        return
    shutil.rmtree(dataset_cache_dir(dataset_id), ignore_errors=True)


def load_openml_task(task_id: int) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    import openml

    configure_openml_cache()
    task = openml.tasks.get_task(task_id)
    X, y = task.get_X_and_y(dataset_format="dataframe")

    return X, pd.Series(y, name=task.target_name)


def load_openml_dataset(
    dataset_id: int,
    target: str,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    import openml

    configure_openml_cache()
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=target)

    return X, pd.Series(y, name=target)


def load_openml_data(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    attempts = 3
    dataset_id = config.get("dataset_id")
    for attempt in range(1, attempts + 1):
        try:
            if config["source"] == "task":
                return load_openml_task(config["task_id"])
            if config["source"] == "dataset":
                return load_openml_dataset(config["dataset_id"], config["target"])
            raise ValueError(f"Unsupported OpenML source: {config['source']}")
        except Exception:
            if attempt == attempts:
                raise
            clear_dataset_cache(dataset_id)
            time.sleep(2 * attempt)

    raise RuntimeError("OpenML loading retry loop exited unexpectedly.")
