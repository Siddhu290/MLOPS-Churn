import pandas as pd

df = pd.read_csv("data/raw/churn.csv")
print("Shape:", df.shape)
print(df.dtypes)
print(df["Churn"].value_counts(normalize=True))
print(df.isna().sum().sum(), "nulls")
print((df["TotalCharges"].astype(str).str.strip() == "").sum(), "blank TotalCharges")
