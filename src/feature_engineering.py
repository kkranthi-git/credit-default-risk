import pandas as pd
import numpy as np


BILL_COLS = [
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6"
]

PAYMENT_COLS = [
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6"
]

PAY_STATUS_COLS = [
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6"
]


ENGINEERED_NUMERICAL_COLS = [
    "TOTAL_BILL_AMT",
    "TOTAL_PAY_AMT",
    "AVG_BILL_AMT",
    "AVG_PAY_AMT",
    "MAX_BILL_AMT",
    "MAX_PAY_AMT",
    "DELAY_COUNT",
    "SEVERE_DELAY_COUNT",
    "AVG_PAY_STATUS",
    "MAX_PAY_STATUS",
    "PAYMENT_TO_BILL_RATIO",
    "CREDIT_UTILIZATION"
]


def create_features(df):

    df = df.copy()

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    df["TOTAL_BILL_AMT"] = (
        df[BILL_COLS].sum(axis=1)
    )

    df["TOTAL_PAY_AMT"] = (
        df[PAYMENT_COLS].sum(axis=1)
    )

    df["AVG_BILL_AMT"] = (
        df[BILL_COLS].mean(axis=1)
    )

    df["AVG_PAY_AMT"] = (
        df[PAYMENT_COLS].mean(axis=1)
    )

    df["MAX_BILL_AMT"] = (
        df[BILL_COLS].max(axis=1)
    )

    df["MAX_PAY_AMT"] = (
        df[PAYMENT_COLS].max(axis=1)
    )

    df["DELAY_COUNT"] = (
        df[PAY_STATUS_COLS] > 0
    ).sum(axis=1)

    df["SEVERE_DELAY_COUNT"] = (
        df[PAY_STATUS_COLS] >= 2
    ).sum(axis=1)

    df["AVG_PAY_STATUS"] = (
        df[PAY_STATUS_COLS].mean(axis=1)
    )

    df["MAX_PAY_STATUS"] = (
        df[PAY_STATUS_COLS].max(axis=1)
    )

    df["PAYMENT_TO_BILL_RATIO"] = (
        df["TOTAL_PAY_AMT"]
        / df["TOTAL_BILL_AMT"].replace(0, np.nan)
    )

    df["CREDIT_UTILIZATION"] = (
        df["BILL_AMT1"]
        / df["LIMIT_BAL"].replace(0, np.nan)
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df