import joblib
import pandas as pd

from src.models.model_factory import get_models
from src.models.trainer import train_model
from src.models.model_utils import (
    save_model,
    load_model
)


def main():

    X_train = joblib.load(
        "data/processed/X_train.joblib"
    )

    y_train = pd.read_csv(
        "data/processed/y_train.csv"
    )["sentiment"]

    model = get_models()["logistic_regression"]

    trained_model = train_model(
        model,
        X_train,
        y_train
    )

    saved_path = save_model(
        trained_model,
        "logistic_regression"
    )

    print("Model saved at:")
    print(saved_path)

    loaded_model = load_model(
        "logistic_regression"
    )

    print()
    print("Loaded Model:")
    print(type(loaded_model))


if __name__ == "__main__":
    main()