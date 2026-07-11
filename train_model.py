"""
train_model.py
---------------
Trains the core PREDICTIVE DEEP LEARNING model (an Artificial Neural Network /
Multi-Layer Perceptron) on the tabular loan-application data.

Why MLPClassifier instead of TensorFlow/PyTorch?
This sandbox has no internet access, so heavy DL frameworks can't be pip-installed
here. scikit-learn's MLPClassifier IS a real feed-forward ANN (backprop, adaptive
learning rate, multiple hidden layers) and needs zero extra dependencies, so the
whole project trains with 0 errors out of the box.
If you have TensorFlow/PyTorch available, swap `build_model()` below for a Keras
Sequential model -- the rest of the pipeline (features, saving, explainability,
Flask app) does not need to change, since it only calls .predict_proba().
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "..", "data", "loan_data.csv")
MODEL_PATH = os.path.join(HERE, "loan_model.joblib")
METRICS_PATH = os.path.join(HERE, "metrics.json")

NUMERIC_FEATURES = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                     "Loan_Amount_Term", "Credit_History"]
CATEGORICAL_FEATURES = ["Gender", "Married", "Dependents", "Education",
                         "Self_Employed", "Property_Area"]
TARGET = "Loan_Status"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalIncome"] = df["ApplicantIncome"].fillna(0) + df["CoapplicantIncome"].fillna(0)
    df["LoanToIncomeRatio"] = (df["LoanAmount"].astype(float) * 1000) / (df["TotalIncome"] + 1)
    return df


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES + ["TotalIncome", "LoanToIncomeRatio"]),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    ann = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        learning_rate_init=1e-3,
        max_iter=800,
        early_stopping=True,
        n_iter_no_change=25,
        random_state=42,
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("ann", ann)])


def main():
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)
    df = df.dropna(subset=[TARGET])

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["TotalIncome", "LoanToIncomeRatio"]]
    y = (df[TARGET] == "Y").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "architecture": "MLP(32,16) - ReLU - Adam - early_stopping",
    }

    joblib.dump(pipe, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")
    print(f"Accuracy: {metrics['accuracy']:.3f}  ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"Model saved -> {MODEL_PATH}")
    print(f"Metrics saved -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
