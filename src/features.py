"""Reusable cleaning, feature engineering, and model preprocessing."""

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

BASE_BINARY_FEATURES = [
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
]

BASE_NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
ONE_HOT_FEATURES = ["InternetService", "Contract", "PaymentMethod"]

ENGINEERED_FEATURES = (
    "avg_monthly_spend",
    "charge_delta",
    "n_services",
    "is_month_to_month",
)

ENGINEERED_NUMERIC_FEATURES = {
    "avg_monthly_spend",
    "charge_delta",
    "n_services",
}

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
    mapped = series.map(mapping)
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.where(numeric.isin([0, 1]))
    return mapped.fillna(numeric).astype("float64")


def normalize_engineered_features(
    enabled_features: Iterable[str] | None,
) -> tuple[str, ...]:
    """Validate and normalize the requested engineered feature names."""
    if enabled_features is None:
        return ENGINEERED_FEATURES

    normalized = tuple(dict.fromkeys(enabled_features))
    unknown = sorted(set(normalized) - set(ENGINEERED_FEATURES))
    if unknown:
        raise ValueError(f"Unknown engineered features: {unknown}")
    return normalized


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Clean raw values and create a configurable set of features."""

    def __init__(self, enabled_features: Iterable[str] | None = None):
        self.enabled_features = enabled_features

    def fit(self, X: pd.DataFrame, y=None):
        self._validate_input(X)
        self.enabled_features_ = normalize_engineered_features(
            self.enabled_features
        )
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(X)
        enabled = set(
            getattr(
                self,
                "enabled_features_",
                normalize_engineered_features(self.enabled_features),
            )
        )

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
        average_spend = average_spend.fillna(transformed["MonthlyCharges"])

        if "avg_monthly_spend" in enabled:
            transformed["avg_monthly_spend"] = average_spend
        if "charge_delta" in enabled:
            transformed["charge_delta"] = (
                transformed["MonthlyCharges"] - average_spend
            )
        if "n_services" in enabled:
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
        if "is_month_to_month" in enabled:
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
        enabled = getattr(
            self,
            "enabled_features_",
            normalize_engineered_features(self.enabled_features),
        )
        for engineered in enabled:
            if engineered not in names:
                names.append(engineered)
        return np.asarray(names, dtype=object)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a complete dataset and encode its optional target."""
    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    target = None
    if "Churn" in cleaned.columns:
        target = _strip_strings(cleaned.pop("Churn"))

    cleaned = FeatureEngineer(enabled_features=[]).fit_transform(cleaned)

    if target is not None:
        encoded_target = _encode_binary(
            target.reset_index(drop=True), {"No": 0, "Yes": 1}
        )
        if encoded_target.isna().any():
            invalid = sorted(target[encoded_target.isna()].astype(str).unique())
            raise ValueError(
                f"Churn contains missing or invalid values: {invalid}"
            )
        cleaned["Churn"] = encoded_target.astype("int8")

    return cleaned


def build_preprocessor(
    enabled_features: Iterable[str] | None = None,
) -> ColumnTransformer:
    """Create preprocessing for one experiment's feature configuration."""
    enabled = normalize_engineered_features(enabled_features)
    numeric_features = BASE_NUMERIC_FEATURES + [
        feature
        for feature in enabled
        if feature in ENGINEERED_NUMERIC_FEATURES
    ]
    binary_features = BASE_BINARY_FEATURES.copy()
    if "is_month_to_month" in enabled:
        binary_features.append("is_month_to_month")

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
            ("numeric", numeric_pipeline, numeric_features),
            ("binary", binary_pipeline, binary_features),
            ("categorical", categorical_pipeline, ONE_HOT_FEATURES),
        ],
        remainder="drop",
    )


def build_feature_pipeline() -> Pipeline:
    """Build the fixed DVC feature-engineering and preprocessing pipeline."""
    return Pipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("preprocessor", build_preprocessor()),
        ]
    )


def _to_dataframe(values, columns: list[str]) -> pd.DataFrame:
    if hasattr(values, "toarray"):
        values = values.toarray()
    return pd.DataFrame(values, columns=columns)


def main() -> None:
    """Fit preprocessing on train data and transform train and holdout."""
    with Path("params.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)["data"]

    train = pd.read_csv(config["train_path"])
    holdout = pd.read_csv(config["holdout_path"])
    if "Churn" not in train.columns or "Churn" not in holdout.columns:
        raise ValueError("Both data splits must contain the Churn target.")

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

    train_features_path = Path(config["train_features_path"])
    holdout_features_path = Path(config["holdout_features_path"])
    preprocessor_path = Path(config["preprocessor_path"])
    train_features_path.parent.mkdir(parents=True, exist_ok=True)
    holdout_features_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)

    train_features.to_csv(train_features_path, index=False)
    holdout_features.to_csv(holdout_features_path, index=False)
    joblib.dump(feature_pipeline, preprocessor_path)

    print(f"Training features: {train_features.shape} -> {train_features_path}")
    print(f"Holdout features: {holdout_features.shape} -> {holdout_features_path}")
    print(f"Fitted preprocessor: {preprocessor_path}")


if __name__ == "__main__":
    main()
