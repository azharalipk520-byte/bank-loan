"""
generate_data.py
-----------------
Generates a synthetic bank-loan-application dataset that mirrors the
structure of the well-known Kaggle "Loan Prediction" dataset
(https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset).

We generate data locally (instead of downloading it) so the project can be
trained and demoed with zero external dependencies / zero internet access.
If you have internet access, you can instead download the real Kaggle CSV
and place it at data/loan_data.csv with the same column names -- the rest
of the pipeline (train_model.py, app.py) will work unchanged.
"""

import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
N_ROWS = 1200

def generate(n_rows: int = N_ROWS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female"], size=n_rows, p=[0.78, 0.22])
    married = rng.choice(["Yes", "No"], size=n_rows, p=[0.65, 0.35])
    dependents = rng.choice(["0", "1", "2", "3+"], size=n_rows, p=[0.55, 0.18, 0.17, 0.10])
    education = rng.choice(["Graduate", "Not Graduate"], size=n_rows, p=[0.78, 0.22])
    self_employed = rng.choice(["Yes", "No"], size=n_rows, p=[0.14, 0.86])
    property_area = rng.choice(["Urban", "Semiurban", "Rural"], size=n_rows, p=[0.38, 0.38, 0.24])

    applicant_income = rng.gamma(shape=4.5, scale=1200, size=n_rows).round(0) + 1500
    coapplicant_income = np.where(
        married == "Yes",
        rng.gamma(shape=2.5, scale=800, size=n_rows).round(0),
        0.0,
    )
    loan_amount = (rng.gamma(shape=3.0, scale=45, size=n_rows) + 40).round(0)  # in thousands
    loan_term = rng.choice([360, 180, 120, 60, 300, 240, 84], size=n_rows,
                            p=[0.72, 0.09, 0.05, 0.04, 0.04, 0.03, 0.03])
    credit_history = rng.choice([1.0, 0.0], size=n_rows, p=[0.84, 0.16])

    # ---- Latent "true" approval probability (drives a realistic target) ----
    total_income = applicant_income + coapplicant_income
    debt_to_income = (loan_amount * 1000) / (total_income * loan_term / 12 + 1)

    score = (
        2.6 * credit_history
        - 1.9 * debt_to_income
        + 0.00025 * total_income
        - 0.35 * (education == "Not Graduate").astype(float)
        - 0.15 * (self_employed == "Yes").astype(float)
        + 0.10 * (property_area == "Semiurban").astype(float)
        - 0.20 * (property_area == "Rural").astype(float)
        - 0.05 * (dependents == "3+").astype(float)
    )
    prob_approve = 1 / (1 + np.exp(-score))
    loan_status = rng.binomial(1, prob_approve)

    df = pd.DataFrame({
        "Loan_ID": [f"LP{100000+i}" for i in range(n_rows)],
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income.astype(int),
        "CoapplicantIncome": coapplicant_income.astype(int),
        "LoanAmount": loan_amount.astype(int),
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
        "Loan_Status": np.where(loan_status == 1, "Y", "N"),
    })

    # introduce a small amount of realistic missingness (like the real dataset)
    for col, frac in [("Gender", 0.02), ("Dependents", 0.02), ("Self_Employed", 0.03),
                       ("LoanAmount", 0.03), ("Credit_History", 0.08)]:
        idx = rng.choice(n_rows, size=int(n_rows * frac), replace=False)
        df.loc[idx, col] = np.nan

    return df


if __name__ == "__main__":
    out_dir = os.path.dirname(__file__)
    df = generate()
    out_path = os.path.join(out_dir, "loan_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df["Loan_Status"].value_counts(normalize=True))
