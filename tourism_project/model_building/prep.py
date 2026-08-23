"""
Cleans the tourism dataset and splits it into train/test sets.

Run from the repository root:
    python tourism_project/model_building/prep.py

Writes Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv to the working
directory (repo root). The GitHub Actions workflow uploads these as a
workflow artifact for the model-training job to consume.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"

# Columns that carry no predictive signal (row index / unique ID)
DROP_COLS = ["Unnamed: 0", "CustomerID"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop identifier / index columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Fix inconsistent category labels found during EDA
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    if "MaritalStatus" in df.columns:
        df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    # Drop exact duplicate rows, if any
    df = df.drop_duplicates()

    return df


def prepare_data(path: str = DATA_PATH, test_size: float = 0.2, random_state: int = 42):
    df = pd.read_csv(path)
    df = clean_data(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print(f"Cleaned dataset shape : {df.shape}")
    print(f"Train shape           : {Xtrain.shape}")
    print(f"Test shape            : {Xtest.shape}")
    print(f"Train target balance  :\n{ytrain.value_counts(normalize=True).round(3)}")
    print(f"Test target balance   :\n{ytest.value_counts(normalize=True).round(3)}")
    print("Saved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")

    return Xtrain, Xtest, ytrain, ytest


if __name__ == "__main__":
    prepare_data()
