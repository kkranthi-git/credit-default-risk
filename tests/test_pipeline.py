import numpy as np
import pandas as pd

from src.feature_engineering import create_features
from src.preprocessing import create_preprocessor
from src.predict import predict_risk


def test_feature_engineering_creates_features():

    df = pd.DataFrame({
        "LIMIT_BAL": [100000],
        "AGE": [30],
        "SEX": [1],
        "EDUCATION": [2],
        "MARRIAGE": [1],
        "PAY_0": [0],
        "PAY_2": [0],
        "PAY_3": [1],
        "PAY_4": [0],
        "PAY_5": [0],
        "PAY_6": [0],
        "BILL_AMT1": [50000],
        "BILL_AMT2": [45000],
        "BILL_AMT3": [40000],
        "BILL_AMT4": [35000],
        "BILL_AMT5": [30000],
        "BILL_AMT6": [25000],
        "PAY_AMT1": [10000],
        "PAY_AMT2": [9000],
        "PAY_AMT3": [8000],
        "PAY_AMT4": [7000],
        "PAY_AMT5": [6000],
        "PAY_AMT6": [5000],
    })

    result = create_features(df)

    assert "TOTAL_BILL_AMT" in result.columns
    assert "TOTAL_PAY_AMT" in result.columns
    assert "DELAY_COUNT" in result.columns
    assert "CREDIT_UTILIZATION" in result.columns


def test_feature_engineering_handles_zero_bill():

    df = pd.DataFrame({
        "LIMIT_BAL": [100000],
        "AGE": [30],
        "SEX": [1],
        "EDUCATION": [2],
        "MARRIAGE": [1],
        "PAY_0": [0],
        "PAY_2": [0],
        "PAY_3": [0],
        "PAY_4": [0],
        "PAY_5": [0],
        "PAY_6": [0],
        "BILL_AMT1": [0],
        "BILL_AMT2": [0],
        "BILL_AMT3": [0],
        "BILL_AMT4": [0],
        "BILL_AMT5": [0],
        "BILL_AMT6": [0],
        "PAY_AMT1": [0],
        "PAY_AMT2": [0],
        "PAY_AMT3": [0],
        "PAY_AMT4": [0],
        "PAY_AMT5": [0],
        "PAY_AMT6": [0],
    })

    result = create_features(df)

    assert np.isfinite(
        result["PAYMENT_TO_BILL_RATIO"].fillna(0)
    ).all()


def test_preprocessor_creates_output():

    df = pd.DataFrame({
        "LIMIT_BAL": [100000, 200000],
        "AGE": [30, 40],
        "SEX": [1, 2],
        "EDUCATION": [2, 1],
        "MARRIAGE": [1, 2],
        "PAY_0": [0, 1],
        "PAY_2": [0, 1],
        "PAY_3": [0, 0],
        "PAY_4": [0, 0],
        "PAY_5": [0, 0],
        "PAY_6": [0, 0],
        "BILL_AMT1": [50000, 100000],
        "BILL_AMT2": [45000, 90000],
        "BILL_AMT3": [40000, 80000],
        "BILL_AMT4": [35000, 70000],
        "BILL_AMT5": [30000, 60000],
        "BILL_AMT6": [25000, 50000],
        "PAY_AMT1": [10000, 20000],
        "PAY_AMT2": [9000, 18000],
        "PAY_AMT3": [8000, 16000],
        "PAY_AMT4": [7000, 14000],
        "PAY_AMT5": [6000, 12000],
        "PAY_AMT6": [5000, 10000],
    })

    df = create_features(df)

    preprocessor = create_preprocessor()

    transformed = preprocessor.fit_transform(df)

    assert transformed.shape[0] == 2
    assert transformed.shape[1] > 0


def test_prediction_output():

    sample = {
        "LIMIT_BAL": 100000,
        "SEX": 1,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 30,
        "PAY_0": 0,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 50000,
        "BILL_AMT2": 45000,
        "BILL_AMT3": 40000,
        "BILL_AMT4": 35000,
        "BILL_AMT5": 30000,
        "BILL_AMT6": 25000,
        "PAY_AMT1": 10000,
        "PAY_AMT2": 9000,
        "PAY_AMT3": 8000,
        "PAY_AMT4": 7000,
        "PAY_AMT5": 6000,
        "PAY_AMT6": 5000,
    }

    sample_df = pd.DataFrame([sample])

    result = predict_risk(sample_df)

    assert "Default_Probability" in result.columns
    assert "Prediction" in result.columns
    assert "Risk_Category" in result.columns

    assert 0 <= result["Default_Probability"].iloc[0] <= 1
    assert result["Prediction"].iloc[0] in [0, 1]
    assert result["Risk_Category"].iloc[0] in [
    "Low Risk",
    "Medium Risk",
    "High Risk"
    ]