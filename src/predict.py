# ============================================================
# src/predict.py
# ============================================================

from pathlib import Path
import joblib
import pandas as pd

from .feature_engineering import create_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
LOW_RISK_MAX = 0.30
MEDIUM_RISK_MAX = 0.60


def categorize_risk(probabilities):
    """Apply project-specific probability bands; these are not industry standards."""
    return pd.cut(
        probabilities,
        bins=[-0.001, LOW_RISK_MAX, MEDIUM_RISK_MAX, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
        include_lowest=True,
    )


def predict_risk(data):

    """Return default probabilities and project risk categories for customer rows."""

    model = joblib.load(
        MODEL_DIR / "random_forest.pkl"
    )

    preprocessor = joblib.load(
        MODEL_DIR / "preprocessor.pkl"
    )

    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("Prediction input must be a non-empty pandas DataFrame.")
    feature_data = create_features(data)
    expected_columns = list(preprocessor.feature_names_in_)
    missing_columns = set(expected_columns).difference(feature_data.columns)
    if missing_columns:
        raise ValueError("Prediction input is missing required columns: " + ", ".join(sorted(missing_columns)))
    processed_data = preprocessor.transform(feature_data[expected_columns])

    probability = model.predict_proba(
        processed_data
    )[:, 1]

    prediction = (
        probability >= 0.5
    ).astype(int)

    result = pd.DataFrame({
        "Default_Probability": probability,
        "Prediction": prediction
    })

    result["Risk_Category"] = categorize_risk(result["Default_Probability"])

    return result


if __name__ == "__main__":

    sample = pd.DataFrame({
        "LIMIT_BAL": [50000],
        "SEX": [2],
        "EDUCATION": [2],
        "MARRIAGE": [2],
        "AGE": [30],
        "PAY_0": [0],
        "PAY_2": [0],
        "PAY_3": [0],
        "PAY_4": [0],
        "PAY_5": [0],
        "PAY_6": [0],
        "BILL_AMT1": [40000],
        "BILL_AMT2": [42000],
        "BILL_AMT3": [41000],
        "BILL_AMT4": [39000],
        "BILL_AMT5": [38000],
        "BILL_AMT6": [37000],
        "PAY_AMT1": [2000],
        "PAY_AMT2": [2000],
        "PAY_AMT3": [2000],
        "PAY_AMT4": [2000],
        "PAY_AMT5": [2000],
        "PAY_AMT6": [2000]
    })

    print(
        predict_risk(sample)
    )
