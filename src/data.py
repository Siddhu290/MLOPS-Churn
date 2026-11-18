import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/churn.csv"


def load_raw(path=RAW_PATH):
    df = pd.read_csv(path)
    # 11 blank TotalCharges are brand-new customers (tenure 0), so 0 is correct
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df = df.drop(columns=["customerID"])   # an ID is not a feature
    return df


def load_split(test_size=0.2, random_state=42, path=RAW_PATH):
    df = load_raw(path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    # stratify keeps the 73/27 churn ratio the same in train and test
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_split()
    print("Train:", X_train.shape, "Test:", X_test.shape)
    print("Train churn ratio:", y_train.value_counts(normalize=True).round(3).to_dict())
    print("Test churn ratio:", y_test.value_counts(normalize=True).round(3).to_dict())
