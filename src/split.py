import pandas as pd, yaml
from sklearn.model_selection import train_test_split

p = yaml.safe_load(open("params.yaml"))["data"]
df = pd.read_csv(p["clean_path"])
train, test = train_test_split(df, test_size=p["test_size"],
                               stratify=df["Churn"], random_state=p["seed"])
train.to_csv(p["train_path"], index=False)
test.to_csv(p["holdout_path"], index=False)