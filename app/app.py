
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from src.predict import predict_risk


st.set_page_config(
    page_title="Credit Default Risk Predictor",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Default Risk Predictor")

st.write(
    "Enter customer information to estimate "
    "the probability of credit default."
)

limit_bal = st.number_input(
    "Credit Limit",
    min_value=0.0,
    value=50000.0
)

sex = st.selectbox(
    "Sex",
    [1, 2]
)

education = st.selectbox(
    "Education",
    [0, 1, 2, 3, 4, 5, 6]
)

marriage = st.selectbox(
    "Marriage",
    [0, 1, 2, 3]
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

pay_0 = st.number_input(
    "Latest Repayment Status",
    min_value=-2,
    max_value=8,
    value=0
)

pay_2 = st.number_input(
    "Repayment Status 2",
    min_value=-2,
    max_value=8,
    value=0
)

pay_3 = st.number_input(
    "Repayment Status 3",
    min_value=-2,
    max_value=8,
    value=0
)

pay_4 = st.number_input(
    "Repayment Status 4",
    min_value=-2,
    max_value=8,
    value=0
)

pay_5 = st.number_input(
    "Repayment Status 5",
    min_value=-2,
    max_value=8,
    value=0
)

pay_6 = st.number_input(
    "Repayment Status 6",
    min_value=-2,
    max_value=8,
    value=0
)

bill_amt1 = st.number_input(
    "Bill Amount 1",
    min_value=0.0,
    value=40000.0
)

bill_amt2 = st.number_input(
    "Bill Amount 2",
    min_value=0.0,
    value=40000.0
)

bill_amt3 = st.number_input(
    "Bill Amount 3",
    min_value=0.0,
    value=40000.0
)

bill_amt4 = st.number_input(
    "Bill Amount 4",
    min_value=0.0,
    value=40000.0
)

bill_amt5 = st.number_input(
    "Bill Amount 5",
    min_value=0.0,
    value=40000.0
)

bill_amt6 = st.number_input(
    "Bill Amount 6",
    min_value=0.0,
    value=40000.0
)

pay_amt1 = st.number_input(
    "Payment Amount 1",
    min_value=0.0,
    value=2000.0
)

pay_amt2 = st.number_input(
    "Payment Amount 2",
    min_value=0.0,
    value=2000.0
)

pay_amt3 = st.number_input(
    "Payment Amount 3",
    min_value=0.0,
    value=2000.0
)

pay_amt4 = st.number_input(
    "Payment Amount 4",
    min_value=0.0,
    value=2000.0
)

pay_amt5 = st.number_input(
    "Payment Amount 5",
    min_value=0.0,
    value=2000.0
)

pay_amt6 = st.number_input(
    "Payment Amount 6",
    min_value=0.0,
    value=2000.0
)


if st.button("Predict Default Risk"):

    input_data = pd.DataFrame({
        "LIMIT_BAL": [limit_bal],
        "SEX": [sex],
        "EDUCATION": [education],
        "MARRIAGE": [marriage],
        "AGE": [age],
        "PAY_0": [pay_0],
        "PAY_2": [pay_2],
        "PAY_3": [pay_3],
        "PAY_4": [pay_4],
        "PAY_5": [pay_5],
        "PAY_6": [pay_6],
        "BILL_AMT1": [bill_amt1],
        "BILL_AMT2": [bill_amt2],
        "BILL_AMT3": [bill_amt3],
        "BILL_AMT4": [bill_amt4],
        "BILL_AMT5": [bill_amt5],
        "BILL_AMT6": [bill_amt6],
        "PAY_AMT1": [pay_amt1],
        "PAY_AMT2": [pay_amt2],
        "PAY_AMT3": [pay_amt3],
        "PAY_AMT4": [pay_amt4],
        "PAY_AMT5": [pay_amt5],
        "PAY_AMT6": [pay_amt6]
    })

    result = predict_risk(input_data)

    probability = (
        result["Default_Probability"].iloc[0]
    )

    category = (
        result["Risk_Category"].iloc[0]
    )

    st.metric(
        "Default Probability",
        f"{probability:.2%}"
    )

    if category == "High Risk":
        st.error(
            f"Risk Category: {category}"
        )

    elif category == "Medium Risk":
        st.warning(
            f"Risk Category: {category}"
        )

    else:
        st.success(
            f"Risk Category: {category}"
        )