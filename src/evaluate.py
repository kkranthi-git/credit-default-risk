from pathlib import Path
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"


def evaluate_model(model_name="random_forest"):

    model = joblib.load(
        MODEL_DIR / f"{model_name}.pkl"
    )

    preprocessor = joblib.load(
        MODEL_DIR / "preprocessor.pkl"
    )

    X_test = joblib.load(
        MODEL_DIR / "X_test.pkl"
    )

    y_test = joblib.load(
        MODEL_DIR / "y_test.pkl"
    )

    X_test_processed = (
        preprocessor.transform(X_test)
    )

    y_pred = model.predict(
        X_test_processed
    )

    y_prob = model.predict_proba(
        X_test_processed
    )[:, 1]

    print(
        "Accuracy:",
        accuracy_score(y_test, y_pred)
    )

    print(
        "Precision:",
        precision_score(y_test, y_pred)
    )

    print(
        "Recall:",
        recall_score(y_test, y_pred)
    )

    print(
        "F1:",
        f1_score(y_test, y_pred)
    )

    print(
        "ROC-AUC:",
        roc_auc_score(y_test, y_prob)
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )


if __name__ == "__main__":
    evaluate_model()