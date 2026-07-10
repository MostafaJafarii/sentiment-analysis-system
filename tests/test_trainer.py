"""
Tests for model training.
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


def test_train_logistic_regression():
    """
    Logistic Regression should train successfully.
    """

    x_train = joblib.load(
        X_TRAIN_FILE
    )

    y_train = pd.read_csv(
        Y_TRAIN_FILE
    )["sentiment"]

    model = get_models()[
        "logistic_regression"
    ]

    trained_model = train_model(
        model,
        x_train,
        y_train
    )

    assert isinstance(
        trained_model,
        LogisticRegression
    )

    # Scikit-Learn models have coef_ after fitting
    assert hasattr(
        trained_model,
        "coef_"
    )


def test_train_returns_same_instance():
    """
    Trainer should return the fitted model instance.
    """

    x_train = joblib.load(
        X_TRAIN_FILE
    )

    y_train = pd.read_csv(
        Y_TRAIN_FILE
    )["sentiment"]

    model = get_models()[
        "logistic_regression"
    ]

    trained_model = train_model(
        model,
        x_train,
        y_train
    )

    assert trained_model is model