import pandas as pd
df = pd.read_csv("../data/data.csv").sample(frac=1, random_state=42)
n = len(df)
df.iloc[:int(.6*n)].to_csv("../data/batch_1.csv", index=False)
df.iloc[int(.6*n):int(.8*n)].to_csv("../data/batch_2.csv", index=False)
df.iloc[int(.8*n):].to_csv("../data/batch_3.csv", index=False)