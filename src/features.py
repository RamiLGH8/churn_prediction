import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

SERVICES = ["PhoneService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"]
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges",
           "avg_monthly_spend", "charge_delta", "n_services"]

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0)
        avg = X["TotalCharges"] / X["tenure"].replace(0, np.nan)
        X["avg_monthly_spend"] = avg.fillna(X["MonthlyCharges"])
        X["charge_delta"] = X["MonthlyCharges"] - X["avg_monthly_spend"]
        X["n_services"] = (X[SERVICES] == "Yes").sum(axis=1)
        X["is_month_to_month"] = (X["Contract"] == "Month-to-month").astype(int)
        return X

def build_pipeline(model):
    pre = ColumnTransformer(
        [("num", StandardScaler(), NUMERIC),
         ("cat", OneHotEncoder(handle_unknown="ignore"),
          make_column_selector(dtype_include=["object", "string"]))],
        remainder="passthrough",  # SeniorCitizen, is_month_to_month
    )
    return Pipeline([("fe", FeatureEngineer()), ("pre", pre), ("model", model)])