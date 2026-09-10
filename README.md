# Credit Default Risk Predictor

This project predicts the chance that a credit-card customer may miss their next payment.

It uses past payment history, bill amounts, payment amounts, credit limit, age, education, and marital status. The result is shown as a default probability and a simple risk level.

## Overview

This is an end-to-end machine-learning portfolio project built around the UCI Default of Credit Card Clients dataset. It includes data preparation, reproducible model training, evaluation, a prediction module, exploratory notebooks, and a Streamlit user interface.

## Business Problem

When a customer misses a credit-card payment, a lender can lose money and must spend time on collections. A risk score helps prioritise manual review and customer-support actions. It must not be the only basis for an automated credit decision.

## Objective

Estimate the probability that a customer will default on their next payment, then group that probability into a simple project risk category.

## What this project can do

- Train machine-learning models from the credit default dataset.
- Compare model performance with accuracy, precision, recall, F1 score, and ROC-AUC.
- Predict risk for one customer in a Streamlit web app.

## Project folders

```text
app/                Streamlit web application
data/raw/           Original dataset file
models/             Trained model files created after training
src/                Training, prediction, preprocessing, and evaluation code
main.py             Runs training and evaluation together
requirements.txt    Python packages needed by the project
```

## Dataset

Put the dataset in this location:

```text
data/raw/credit_default.xls
```

The project uses the UCI `Default of Credit Card Clients` dataset. The workbook has its column names on the second row; the project handles that automatically.

The target is `default payment next month`, where `1` means default and `0` means no default. The source data has 30,000 rows. The training pipeline removes 35 duplicate records after excluding the row identifier, leaving 29,965 records before the train/test split.

## Features and Data Processing

The pipeline removes `ID`, keeps the target separate from model features, performs a stratified 80/20 train/test split, and fits imputation, scaling, and one-hot encoding on the training set only. This avoids preprocessing leakage into the test set.

`SEX`, `EDUCATION`, and `MARRIAGE` are handled as categories. Repayment-status columns are treated as ordered numeric values. Other balance, bill, payment, and age columns are numerical.

## Feature Engineering

The same feature function is used during training and prediction. It creates total, average, and maximum bill/payment values; delay counts; average and maximum repayment status; payment-to-bill ratio; and credit utilisation. Division by zero is converted to a missing value so the imputer can handle it safely.

## Exploratory Data Analysis

The notebooks inspect data quality, class balance, payment status, customer attributes, and bill/payment patterns. Saved charts are available in `reports/figures/`. The notebooks are exploratory material; the code in `src/` is the source of truth for the runnable pipeline.

## Machine Learning Models

- Logistic Regression: interpretable baseline with balanced class weights.
- Random Forest: non-linear model with 300 trees, depth limited to 10, balanced class weights, and a fixed random seed.

The random forest is the selected model because it has stronger ROC-AUC and F1 score than the logistic-regression baseline on the held-out test set. Logistic Regression has higher recall for the default class.

## Model Evaluation

The evaluation reports accuracy, precision, recall, F1 score, ROC-AUC, a classification report, and a confusion matrix. The following results were verified in a clean run of the deduplicated pipeline:

| Model | ROC-AUC | Recall (default class) | F1 (default class) |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.7608 | 0.6192 | 0.5152 |
| Random Forest | 0.7734 | 0.5958 | 0.5298 |

For credit risk, recall is important because a false negative means a customer who may default is labelled as lower risk. The operating threshold remains `0.50` for this project; a real lender would choose it using the cost of false negatives, false positives, and calibration analysis.

## Risk Scoring

The Streamlit app displays the model probability and uses these **project-specific** bands:

- Low Risk: 0% to 30%
- Medium Risk: above 30% to 60%
- High Risk: above 60%

These values are not industry-standard lending thresholds.

## Streamlit Application

The application accepts all raw customer fields required by the model and adds engineered fields automatically before prediction. The underlying dataset can contain negative bill amounts, which are preserved by the data-processing pipeline.

## Setup

You need Python 3.10 or newer installed on your computer.

Open PowerShell in the project folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell does not allow activation, you can run commands with the virtual-environment Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Train and evaluate the models

Run this command from the project folder:

```powershell
.\.venv\Scripts\python.exe main.py
```

This will:

1. Read the dataset.
2. Split it into training and test data.
3. Train logistic-regression and random-forest models.
4. Save the trained random-forest model and preprocessing files in `models/`.
5. Print model evaluation results in the terminal.

## Run the web app

Train the model at least once before starting the app. Then run:

```powershell
.\.venv\Scripts\streamlit.exe run app\app.py
```

Streamlit will show a local web address, usually `http://localhost:8501`. Open that address in your browser.

Enter the customer information and click **Predict Default Risk**.

## Running with Docker

Train the model locally first so that the local `models/` folder contains the generated model and preprocessor. Then build and run:

```powershell
docker build -t credit-default-risk .
docker run --rm -p 8501:8501 credit-default-risk
```

Open http://localhost:8501 in your browser.

Docker status: Docker image build and container smoke testing were successfully completed. The Streamlit application was verified inside the Docker container at http://localhost:8501.

## Example Prediction

A sample customer with a 50,000 credit limit, no recent repayment delays, 40,000 monthly bill amounts, and 2,000 monthly payments produced a default probability of approximately 37.12% in the validated application run, which falls in the Medium Risk band.

## Key Business Insights

- The default class is a minority class, so accuracy alone is not enough to judge the model.
- Recent repayment-status information is likely important for identifying potential default risk.
- The probability should guide review or support actions, not replace accountable human decisions.

## Limitations and Future Improvements

- This historical dataset may not represent a current lender, country, customer population, or policy.
- Probability calibration and threshold selection based on real business costs are not implemented.
- Model/data versioning and CI are not implemented and would be useful for a production workflow.
- Do a fairness assessment before using protected or sensitive attributes in a real decision workflow.

## Understanding the result

- **Default Probability**: the model's estimated chance of a missed next payment.
- **Low Risk**: probability up to 30%.
- **Medium Risk**: probability above 30% and up to 60%.
- **High Risk**: probability above 60%.

This is a learning project. A model prediction should support human decision-making, not make a credit decision by itself.

## Useful commands

```powershell
# Train only
.\.venv\Scripts\python.exe -m src.train

# Evaluate the saved random-forest model only
.\.venv\Scripts\python.exe -m src.evaluate

# Test the prediction code with an example customer
.\.venv\Scripts\python.exe -m src.predict
```

## Technologies Used

Python, pandas, NumPy, scikit-learn, joblib, Streamlit, Matplotlib, Seaborn, Jupyter, Docker, and xlrd.
