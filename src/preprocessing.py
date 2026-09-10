from pathlib import Path
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from .feature_engineering import ENGINEERED_NUMERICAL_COLS, create_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT /
    "data" /
    "raw" /
    "credit_default.xls"
)

MODEL_DIR = PROJECT_ROOT / "models"
TARGET_COLUMN = "default payment next month"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CATEGORICAL_COLS = [
    "SEX",
    "EDUCATION",
    "MARRIAGE"
]

ORDINAL_COLS = [
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6"
]

NUMERICAL_COLS = [
    "LIMIT_BAL",
    "AGE",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6"
] + ENGINEERED_NUMERICAL_COLS

FEATURE_COLUMNS = NUMERICAL_COLS + ORDINAL_COLS + CATEGORICAL_COLS


def create_preprocessor():

    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    ordinal_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "num",
            numerical_pipeline,
            NUMERICAL_COLS
        ),
        (
            "ordinal",
            ordinal_pipeline,
            ORDINAL_COLS
        ),
        (
            "cat",
            categorical_pipeline,
            CATEGORICAL_COLS
        )
    ])

    return preprocessor


def prepare_data():

    """Load, deduplicate, and prepare raw features and target values."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}.")

    # The UCI source workbook stores its column labels on the second row.
    df = pd.read_excel(DATA_PATH, header=1)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset is missing target column: {TARGET_COLUMN}")

    # ID is unique by design and must not hide duplicate observations.
    columns_without_id = [column for column in df.columns if column != "ID"]
    df = df.drop_duplicates(subset=columns_without_id).copy()
    df = create_features(df)

    X = df[FEATURE_COLUMNS]

    y = df[TARGET_COLUMN].astype(int)

    return X, y


if __name__ == "__main__":

    X, y = prepare_data()

    preprocessor = create_preprocessor()

    print(X.shape)
    print(y.shape)
