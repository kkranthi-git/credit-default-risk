from src.train import train_models
from src.evaluate import evaluate_model


def main():
    print("Starting Credit Default Risk Pipeline...")

    train_models()

    print("Training completed.")

    evaluate_model()

    print("Evaluation completed.")
    print("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
