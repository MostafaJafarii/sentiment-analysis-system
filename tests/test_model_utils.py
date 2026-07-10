"""
Tests for model saving and loading utilities.
"""

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression

from src.config import (
    X_TRAIN_FILE,
    Y_TRAIN_FILE,
)

from src.models.model_factory import get_models
from src.models.trainer import train_model
from src.models.model_utils import (
    save_model,
    load_model,
)


MODEL_NAME = "logistic_regression"


def get_trained_model():
    """
    Train a Logistic Regression model.
    """

    x_train = joblib.load(
        X_TRAIN_FILE
    )

    y_train = pd.read_csv(
        Y_TRAIN_FILE
    )["sentiment"]

    model = get_models()[
        MODEL_NAME
    ]

    return train_model(
        model,
        x_train,
        y_train
    )


def test_save_model():
    """
    Model should be saved successfully.
    """

    model = get_trained_model()

    model_path = save_model(
        model,
        MODEL_NAME
    )

    assert model_path.exists()


def test_load_model():
    """
    Saved model should be loadable.
    """

    model = get_trained_model()

    save_model(
        model,
        MODEL_NAME
    )

    loaded_model = load_model(
        MODEL_NAME
    )

    assert isinstance(
        loaded_model,
        LogisticRegression
    )


def test_loaded_model_can_predict():
    """
    Loaded model should make predictions.
    """

    model = get_trained_model()

    save_model(
        model,
        MODEL_NAME
    )

    loaded_model = load_model(
        MODEL_NAME
    )

    x_train = joblib.load(
        X_TRAIN_FILE
    )

    predictions = loaded_model.predict(
        x_train[:10]
    )

    assert len(predictions) == 10