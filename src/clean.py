import pandas as pd, yaml
from pandas.api.types import is_string_dtype

p = yaml.safe_load(open("params.yaml"))["data"]
df = pd.read_csv(p["raw_path"])

df.columns = df.columns.str.strip()
df = df.drop(columns=["customerID"]).drop_duplicates()
df = df.apply(lambda s: s.str.strip() if is_string_dtype(s) else s)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)  # blanks = tenure 0
df["Churn"] = (df["Churn"] == "Yes").astype(int)

assert df["Churn"].isin([0, 1]).all()
assert df.isna().sum().sum() == 0, "unexpected nulls"
df.to_csv(p["clean_path"], index=False)
print(df.shape, "churn rate:", round(df["Churn"].mean(), 3))