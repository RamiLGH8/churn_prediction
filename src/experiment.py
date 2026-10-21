"""Compare model configurations using DVC-preprocessed feature data."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline


MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Experiment configuration must be a YAML mapping.")

    required = {"experiment", "data", "models"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")
    return config


def build_model(model_name: str, parameters: dict[str, Any]):
    try:
        model_class = MODEL_REGISTRY[model_name]
    except KeyError as error:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models: {supported}"
        ) from error
    return model_class(**parameters)


def build_experiment_pipeline(model, number_of_features: int | str) -> Pipeline:
    """Build feature selection plus model; preprocessing is already done."""
    return Pipeline(
        steps=[
            ("feature_selection", SelectKBest(f_classif, k=number_of_features)),
            ("model", model),
        ]
    )


def calculate_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
    }


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def validate_config(config: dict[str, Any], available_features: int) -> None:
    experiment = config["experiment"]
    thresholds = experiment.get("thresholds", [0.5])
    if not thresholds or any(not 0 < float(value) < 1 for value in thresholds):
        raise ValueError("Every decision threshold must be between 0 and 1.")

    valid_metrics = {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}
    primary_metric = experiment.get("primary_metric", "f1")
    if primary_metric not in valid_metrics:
        raise ValueError(
            f"primary_metric must be one of {sorted(valid_metrics)}."
        )

    feature_counts = experiment.get("number_of_features", ["all"])
    for count in feature_counts:
        if count == "all":
            continue
        if not isinstance(count, int) or count < 1 or count > available_features:
            raise ValueError(
                "number_of_features values must be 'all' or integers between "
                f"1 and {available_features}. Got: {count!r}"
            )


def candidate_configurations(config: dict[str, Any]):
    feature_counts = config["experiment"].get(
        "number_of_features", ["all"]
    )
    for model_name, model_config in config["models"].items():
        if not model_config.get("enabled", True):
            continue
        fixed_parameters = model_config.get("fixed_params", {})
        parameter_grid = model_config.get("param_grid", {})
        for tuned_parameters in ParameterGrid(parameter_grid):
            parameters = {**fixed_parameters, **tuned_parameters}
            for feature_count in feature_counts:
                yield {
                    "model_name": model_name,
                    "model_parameters": parameters,
                    "number_of_features": feature_count,
                }


def run_experiment(config_path: Path) -> None:
    config = load_config(config_path)
    experiment_config = config["experiment"]
    data_config = config["data"]

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or experiment_config.get(
        "tracking_uri"
    )
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_config["name"])

    data = pd.read_csv(data_config["train_features_path"])
    target_column = data_config.get("target", "Churn")
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    X = data.drop(columns=target_column)
    y = pd.to_numeric(data[target_column], errors="raise").astype(int)
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("The target must contain only binary values 0 and 1.")
    validate_config(config, X.shape[1])

    folds = int(experiment_config.get("cv_folds", 5))
    random_state = int(experiment_config.get("random_state", 42))
    n_jobs = int(experiment_config.get("n_jobs", -1))
    thresholds = [
        float(value) for value in experiment_config.get("thresholds", [0.5])
    ]
    primary_metric = experiment_config.get("primary_metric", "f1")
    cross_validation = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )

    best_result: dict[str, Any] | None = None
    with mlflow.start_run(run_name=experiment_config.get("run_name")):
        mlflow.set_tags(
            {
                "git_commit": git_revision(),
                "dataset_path": data_config["train_features_path"],
                "preprocessing": "dvc-feature-engineering-stage",
                "experiment_type": "model-selection",
            }
        )
        mlflow.log_artifact(str(config_path), artifact_path="config")
        if Path("dvc.lock").exists():
            mlflow.log_artifact("dvc.lock", artifact_path="data_version")
        preprocessor_path = data_config.get("preprocessor_path")
        if preprocessor_path and Path(preprocessor_path).exists():
            mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")

        for index, candidate in enumerate(candidate_configurations(config), start=1):
            model = build_model(
                candidate["model_name"], candidate["model_parameters"]
            )
            pipeline = build_experiment_pipeline(
                model, candidate["number_of_features"]
            )
            run_name = (
                f"{candidate['model_name']}-"
                f"k{candidate['number_of_features']}-{index}"
            )

            with mlflow.start_run(run_name=run_name, nested=True):
                probabilities = cross_val_predict(
                    pipeline,
                    X,
                    y,
                    cv=cross_validation,
                    method="predict_proba",
                    n_jobs=n_jobs,
                )[:, 1]
                threshold_results = {
                    str(threshold): calculate_metrics(y, probabilities, threshold)
                    for threshold in thresholds
                }
                best_threshold = max(
                    thresholds,
                    key=lambda threshold: threshold_results[str(threshold)][
                        primary_metric
                    ],
                )
                metrics = threshold_results[str(best_threshold)]

                mlflow.log_params(
                    {
                        "model": candidate["model_name"],
                        "number_of_features": str(
                            candidate["number_of_features"]
                        ),
                        "decision_threshold": best_threshold,
                        **{
                            f"model__{key}": "none" if value is None else value
                            for key, value in candidate["model_parameters"].items()
                        },
                    }
                )
                mlflow.log_metrics(
                    {f"cv_{key}": value for key, value in metrics.items()}
                )
                mlflow.log_dict(threshold_results, "threshold_metrics.json")

                result = {
                    **candidate,
                    "threshold": best_threshold,
                    "metrics": metrics,
                }
                if (
                    best_result is None
                    or metrics[primary_metric]
                    > best_result["metrics"][primary_metric]
                ):
                    best_result = result

        if best_result is None:
            raise ValueError("No enabled model configurations were found.")

        best_model = build_model(
            best_result["model_name"], best_result["model_parameters"]
        )
        best_pipeline = build_experiment_pipeline(
            best_model, best_result["number_of_features"]
        )
        best_pipeline.fit(X, y)
        selected_mask = best_pipeline.named_steps[
            "feature_selection"
        ].get_support()
        selected_features = X.columns[selected_mask].tolist()

        mlflow.log_params(
            {
                "best_model": best_result["model_name"],
                "best_number_of_features": str(
                    best_result["number_of_features"]
                ),
                "best_decision_threshold": best_result["threshold"],
            }
        )
        mlflow.log_metrics(
            {
                f"best_cv_{key}": value
                for key, value in best_result["metrics"].items()
            }
        )
        mlflow.log_dict(
            {
                "decision_threshold": best_result["threshold"],
                "selected_features": selected_features,
            },
            "best_model_metadata.json",
        )
        mlflow.sklearn.log_model(best_pipeline, artifact_path="best_model")

        print("Best experiment configuration")
        print(f"  model: {best_result['model_name']}")
        print(f"  selected feature count: {len(selected_features)}")
        print(f"  decision threshold: {best_result['threshold']}")
        print(
            f"  CV {primary_metric}: "
            f"{best_result['metrics'][primary_metric]:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare models using DVC-preprocessed churn features."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_experiment(arguments.config)
