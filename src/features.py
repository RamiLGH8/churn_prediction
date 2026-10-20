"""Cleaning and feature preprocessing for the churn model.

The transformer in this module is deliberately row preserving so it can be
used safely in a scikit-learn Pipeline. Use ``clean_dataset`` before splitting
the raw dataset when dataset-level cleaning (for example, duplicate removal
and target encoding) is required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BINARY_MAPS = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

SERVICE_COLUMNS = [
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

SERVICE_MAP = {
    "No": 0,
    "No phone service": 0,
    "No internet service": 0,
    "Yes": 1,
}

# These remain binary 0/1 and are intentionally not standardized.
BINARY_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
    "is_month_to_month",
]

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "avg_monthly_spend",
    "charge_delta",
    "n_services",
]

# InternetService also has more than two categories, so it must be encoded
# alongside the two explicitly requested multi-category columns.
ONE_HOT_FEATURES = ["InternetService", "Contract", "PaymentMethod"]

REQUIRED_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    *SERVICE_COLUMNS,
    "InternetService",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


def _strip_strings(series: pd.Series) -> pd.Series:
    """Trim strings and turn empty strings into missing values."""
    if not (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        return series

    stripped = series.map(
        lambda value: value.strip() if isinstance(value, str) else value
    )
    return stripped.replace(r"^\s*$", np.nan, regex=True)


def _encode_binary(series: pd.Series, mapping: dict[str, int]) -> pd.Series:
    """Encode labels while preserving input that is already binary."""
    mapped = series.map(mapping)
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.where(numeric.isin([0, 1]))
    return mapped.fillna(numeric).astype("float64")


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Clean raw feature values, encode binary fields, and add features."""

    def fit(self, X: pd.DataFrame, y=None):
        self._validate_input(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(X)
        transformed = X.copy()
        transformed.columns = transformed.columns.str.strip()
        transformed = transformed.drop(columns=["customerID"], errors="ignore")

        for column in transformed.columns:
            transformed[column] = _strip_strings(transformed[column])

        for column in [
            "SeniorCitizen",
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
        ]:
            transformed[column] = pd.to_numeric(
                transformed[column], errors="coerce"
            )

        for column, mapping in BINARY_MAPS.items():
            transformed[column] = _encode_binary(
                transformed[column], mapping
            )

        for column in SERVICE_COLUMNS:
            transformed[column] = _encode_binary(
                transformed[column], SERVICE_MAP
            )

        tenure = transformed["tenure"].replace(0, np.nan)
        average_spend = transformed["TotalCharges"] / tenure
        transformed["avg_monthly_spend"] = average_spend.fillna(
            transformed["MonthlyCharges"]
        )
        transformed["charge_delta"] = (
            transformed["MonthlyCharges"]
            - transformed["avg_monthly_spend"]
        )

        service_count_columns = [
            "PhoneService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        transformed["n_services"] = (
            transformed[service_count_columns].fillna(0).sum(axis=1)
        )
        transformed["is_month_to_month"] = (
            transformed["Contract"] == "Month-to-month"
        ).astype("int8")

        return transformed

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineer expects a pandas DataFrame.")

        normalized_columns = {str(column).strip() for column in X.columns}
        missing = sorted(set(REQUIRED_FEATURES) - normalized_columns)
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

    def get_feature_names_out(
        self, input_features: Iterable[str] | None = None
    ) -> np.ndarray:
        if input_features is None:
            input_features = self.feature_names_in_
        names = [
            str(column).strip()
            for column in input_features
            if str(column).strip() != "customerID"
        ]
        for engineered in [
            "avg_monthly_spend",
            "charge_delta",
            "n_services",
            "is_month_to_month",
        ]:
            if engineered not in names:
                names.append(engineered)
        return np.asarray(names, dtype=object)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a complete raw dataset before train/holdout splitting.

    Unlike ``FeatureEngineer.transform``, this function may remove duplicate
    rows. It also converts the optional Churn target to 0/1.
    """
    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    target = None
    if "Churn" in cleaned.columns:
        target = _strip_strings(cleaned.pop("Churn"))

    cleaned = FeatureEngineer().fit_transform(cleaned)

    if target is not None:
        target = target.reset_index(drop=True)
        encoded_target = _encode_binary(target, {"No": 0, "Yes": 1})
        if encoded_target.isna().any():
            invalid = sorted(target[encoded_target.isna()].astype(str).unique())
            raise ValueError(
                f"Churn contains missing or invalid values: {invalid}"
            )
        cleaned["Churn"] = encoded_target.astype("int8")

    return cleaned


def build_preprocessor() -> ColumnTransformer:
    """Create the fitted-on-training-data preprocessing transformer."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    binary_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("binary", binary_pipeline, BINARY_FEATURES),
            ("categorical", categorical_pipeline, ONE_HOT_FEATURES),
        ],
        remainder="drop",
    )


def build_feature_pipeline() -> Pipeline:
    """Build an end-to-end cleaning and feature preprocessing pipeline."""
    return Pipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("preprocessor", build_preprocessor()),
        ]
    )


def _to_dataframe(values, columns: list[str]) -> pd.DataFrame:
    """Convert dense or sparse transformer output to a DataFrame."""
    if hasattr(values, "toarray"):
        values = values.toarray()
    return pd.DataFrame(values, columns=columns)


def main() -> None:
    """Fit preprocessing on the training split and transform both splits."""
    with Path("params.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)["data"]

    train_path = Path(config["train_path"])
    holdout_path = Path(config["holdout_path"])
    train_features_path = Path(config["train_features_path"])
    holdout_features_path = Path(config["holdout_features_path"])
    preprocessor_path = Path(config["preprocessor_path"])

    train = pd.read_csv(train_path)
    holdout = pd.read_csv(holdout_path)

    if "Churn" not in train.columns or "Churn" not in holdout.columns:
        raise ValueError("Both input splits must contain the Churn target.")

    X_train = train.drop(columns="Churn")
    y_train = train["Churn"].reset_index(drop=True)
    X_holdout = holdout.drop(columns="Churn")
    y_holdout = holdout["Churn"].reset_index(drop=True)

    feature_pipeline = build_feature_pipeline()
    transformed_train = feature_pipeline.fit_transform(X_train)
    transformed_holdout = feature_pipeline.transform(X_holdout)

    columns = list(
        feature_pipeline.named_steps[
            "preprocessor"
        ].get_feature_names_out()
    )
    train_features = _to_dataframe(transformed_train, columns)
    holdout_features = _to_dataframe(transformed_holdout, columns)
    train_features["Churn"] = y_train.to_numpy()
    holdout_features["Churn"] = y_holdout.to_numpy()

    train_features_path.parent.mkdir(parents=True, exist_ok=True)
    holdout_features_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)

    train_features.to_csv(train_features_path, index=False)
    holdout_features.to_csv(holdout_features_path, index=False)
    joblib.dump(feature_pipeline, preprocessor_path)

    print(f"Training features: {train_features.shape} -> {train_features_path}")
    print(f"Holdout features: {holdout_features.shape} -> {holdout_features_path}")
    print(f"Saved fitted preprocessor to: {preprocessor_path}")


if __name__ == "__main__":
    main()
