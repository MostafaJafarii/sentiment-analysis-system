import joblib
import pandas as pd

from src.models.model_factory import get_models
from src.models.trainer import train_model
from src.models.evaluator import evaluate_model


def main():

    X_train = joblib.load(
        "data/processed/X_train.joblib"
    )

    y_train = pd.read_csv(
        "data/processed/y_train.csv"
    )["sentiment"]

    X_validation = joblib.load(
        "data/processed/X_validation.joblib"
    )

    y_validation = pd.read_csv(
        "data/processed/y_validation.csv"
    )["sentiment"]

    model = get_models()["logistic_regression"]

    trained_model = train_model(
        model,
        X_train,
        y_train
    )

    results = evaluate_model(
        trained_model,
        X_validation,
        y_validation
    )

    print()

    print("Accuracy:")

    print(results["accuracy"])

    print()

    print("Precision:")

    print(results["precision"])

    print()

    print("Recall:")

    print(results["recall"])

    print()

    print("F1 Score:")

    print(results["f1_score"])

    print()

    print(results["classification_report"])

    print()

    print(results["confusion_matrix"])


if __name__ == "__main__":
    main()