import joblib
import pandas as pd

from src.models.model_factory import get_models
from src.models.trainer import train_model


def main():

    X_train = joblib.load(
        "data/processed/X_train.joblib"
    )

    y_train = pd.read_csv(
        "data/processed/y_train.csv"
    )["sentiment"]

    models = get_models()

    model = models["logistic_regression"]

    trained_model = train_model(
        model,
        X_train,
        y_train
    )

    print()

    print("Model Type:")

    print(type(trained_model))


if __name__ == "__main__":

    main()