from pathlib import Path
import joblib
import pandas as pd

from src.feature_engineering import create_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"


def predict_risk(data):

    model = joblib.load(
        MODEL_DIR / "random_forest.pkl"
    )

    preprocessor = joblib.load(
        MODEL_DIR / "preprocessor.pkl"
    )

    data = create_features(data)

    processed_data = (
        preprocessor.transform(data)
    )

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

    result["Risk_Category"] = pd.cut(
        result["Default_Probability"],
        bins=[0, 0.30, 0.60, 1.0],
        labels=[
            "Low Risk",
            "Medium Risk",
            "High Risk"
        ],
        include_lowest=True
    )
    
    return result