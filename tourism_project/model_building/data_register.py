"""
Registers the tourism dataset: validates that the expected columns are
present and prints a short summary.

Run from the repository root:
    python tourism_project/model_building/data_register.py
"""
import os
import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]


def register_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")

    na_counts = df.isna().sum()
    na_counts = na_counts[na_counts > 0]

    print("Dataset registered successfully.")
    print(f"Path             : {path}")
    print(f"Rows, Columns    : {df.shape}")
    print(f"Expected columns : all {len(EXPECTED_COLUMNS)} present")
    print(f"Target balance   :\n{df['ProdTaken'].value_counts(normalize=True).round(3)}")
    print(f"Missing values   :\n{na_counts if not na_counts.empty else 'none'}")

    return df


if __name__ == "__main__":
    register_dataset()
