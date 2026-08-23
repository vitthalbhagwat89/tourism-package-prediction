"""
Trains and tunes a model to predict ProdTaken, logs the experiment to
MLflow, evaluates the best model, and saves it for deployment.

Run from the repository root, after prep.py has produced the train/test
splits (Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv):
    python tourism_project/model_building/train.py
"""
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

MODEL_DIR = "tourism_project/deployment"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")

NUMERIC_FEATURES = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]
CATEGORICAL_FEATURES = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

PARAM_GRID = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.05, 0.1],
}


def build_pipeline() -> Pipeline:
    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    model = XGBClassifier(eval_metric="logloss", random_state=42)

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train():
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze()
    ytest = pd.read_csv("ytest.csv").squeeze()

    pipeline = build_pipeline()

    mlflow.set_experiment("tourism-package-prediction")

    with mlflow.start_run():
        grid_search = GridSearchCV(
            pipeline, PARAM_GRID, cv=3, scoring="f1", n_jobs=-1
        )
        grid_search.fit(Xtrain, ytrain)

        best_model = grid_search.best_estimator_

        # Log all tuned parameters
        mlflow.log_params(grid_search.best_params_)

        y_pred = best_model.predict(Xtest)
        acc = accuracy_score(ytest, y_pred)
        f1 = f1_score(ytest, y_pred)

        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1", f1)

        print("Best parameters:", grid_search.best_params_)
        print(f"Test accuracy: {acc:.4f}")
        print(f"Test F1-score: {f1:.4f}")
        print(classification_report(ytest, y_pred))

        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)
        mlflow.sklearn.log_model(best_model, "model")

        print(f"Best model saved to {MODEL_PATH}")

    return best_model


if __name__ == "__main__":
    train()
