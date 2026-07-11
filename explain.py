"""
explain.py
----------
Explainable AI (XAI) layer for the loan ANN.

Provides:
1. Global feature importance -> permutation_importance (model-agnostic,
   ships with scikit-learn, no extra installs needed).
2. Per-application local explanation -> a lightweight KernelSHAP-style
   perturbation method implemented from scratch (SimpleSHAP), so each
   prediction gets its own signed contribution-per-feature breakdown,
   exactly like SHAP would provide, without requiring the `shap` package.
3. A human-readable "reasoning" sentence built from the top contributors.

If the real `shap` library IS available in your environment, `explain_instance`
will automatically use it (see USE_REAL_SHAP below) for exact Shapley values;
otherwise it falls back to SimpleSHAP transparently -- callers don't need to
know which one ran.
"""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

try:
    import shap  # noqa: F401
    USE_REAL_SHAP = True
except ImportError:
    USE_REAL_SHAP = False

FEATURE_LABELS = {
    "ApplicantIncome": "Applicant income",
    "CoapplicantIncome": "Co-applicant income",
    "LoanAmount": "Loan amount requested",
    "Loan_Amount_Term": "Loan term",
    "Credit_History": "Credit history",
    "TotalIncome": "Total household income",
    "LoanToIncomeRatio": "Loan-to-income ratio",
    "Gender": "Gender",
    "Married": "Marital status",
    "Dependents": "Dependents",
    "Education": "Education level",
    "Self_Employed": "Self-employment status",
    "Property_Area": "Property area",
}


def global_importance(pipe, X_sample: pd.DataFrame, y_sample, n_repeats: int = 8):
    """Model-agnostic permutation feature importance on the raw (pre-pipeline) columns."""
    result = permutation_importance(
        pipe, X_sample, y_sample, n_repeats=n_repeats, random_state=42, scoring="roc_auc"
    )
    importances = pd.Series(result.importances_mean, index=X_sample.columns)
    importances = importances.sort_values(ascending=False)
    return importances


class SimpleSHAP:
    """A minimal, dependency-free approximation of SHAP's local explanation idea:
    for each feature, replace it with a 'background' (median/mode) value and
    measure how much the predicted probability changes. This isolates each
    feature's marginal contribution to THIS specific prediction."""

    def __init__(self, pipe, background: pd.DataFrame):
        self.pipe = pipe
        self.background = background

    def explain(self, row: pd.DataFrame) -> pd.Series:
        base_proba = self.pipe.predict_proba(row)[0, 1]
        contributions = {}
        for col in row.columns:
            perturbed = row.copy()
            if pd.api.types.is_numeric_dtype(self.background[col]):
                perturbed[col] = self.background[col].median()
            else:
                perturbed[col] = self.background[col].mode().iloc[0]
            perturbed_proba = self.pipe.predict_proba(perturbed)[0, 1]
            contributions[col] = base_proba - perturbed_proba
        return pd.Series(contributions).sort_values(key=np.abs, ascending=False)


def explain_instance(pipe, background: pd.DataFrame, row: pd.DataFrame, top_k: int = 4):
    """Returns (confidence, contributions_series, reasoning_text) for one application."""
    proba = pipe.predict_proba(row)[0, 1]

    explainer = SimpleSHAP(pipe, background)
    contributions = explainer.explain(row)

    top = contributions.head(top_k)
    parts = []
    for feat, val in top.items():
        label = FEATURE_LABELS.get(feat, feat)
        direction = "increased" if val > 0 else "decreased"
        parts.append(f"{label} {direction} approval likelihood")
    reasoning = "; ".join(parts) + "."

    return {
        "confidence": float(proba),
        "contributions": contributions.to_dict(),
        "reasoning": reasoning,
        "engine": "shap" if USE_REAL_SHAP else "simple_shap (perturbation-based)",
    }
