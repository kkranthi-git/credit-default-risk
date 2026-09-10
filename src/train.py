from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from preprocessing import (
    prepare_data,
    create_preprocessor
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def train_models():

    X, y = prepare_data()

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    )

    preprocessor = create_preprocessor()

    X_train_processed = (
        preprocessor.fit_transform(X_train)
    )

    X_test_processed = (
        preprocessor.transform(X_test)
    )

    models = {
        "logistic_regression":
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ),

        "random_forest":
            RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
    }

    for name, model in models.items():

        model.fit(
            X_train_processed,
            y_train
        )

        joblib.dump(
            model,
            MODEL_DIR / f"{name}.pkl"
        )

    joblib.dump(
        preprocessor,
        MODEL_DIR / "preprocessor.pkl"
    )

    joblib.dump(
        X_test,
        MODEL_DIR / "X_test.pkl"
    )

    joblib.dump(
        y_test,
        MODEL_DIR / "y_test.pkl"
    )

    print("Models trained successfully.")


if __name__ == "__main__":
    train_models()