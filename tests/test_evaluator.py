"""
Tests for model evaluation.
"""

import joblib
import pandas as pd

from src.config import (
    X_TRAIN_FILE,
    X_VALIDATION_FILE,
    Y_TRAIN_FILE,
    Y_VALIDATION_FILE,
)

from src.models.model_factory import get_models
from src.models.trainer import train_model
from src.models.evaluator import evaluate_model


def get_trained_model():
    """
    Train a Logistic Regression model for evaluation tests.
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

    return train_model(
        model,
        x_train,
        y_train
    )


def test_evaluation_returns_dictionary():
    """
    Evaluator should return a dictionary.
    """

    model = get_trained_model()

    x_validation = joblib.load(
        X_VALIDATION_FILE
    )

    y_validation = pd.read_csv(
        Y_VALIDATION_FILE
    )["sentiment"]

    results = evaluate_model(
        model,
        x_validation,
        y_validation
    )

    assert isinstance(
        results,
        dict
    )


def test_metrics_exist():
    """
    Evaluation dictionary should contain all required metrics.
    """

    model = get_trained_model()

    x_validation = joblib.load(
        X_VALIDATION_FILE
    )

    y_validation = pd.read_csv(
        Y_VALIDATION_FILE
    )["sentiment"]

    results = evaluate_model(
        model,
        x_validation,
        y_validation
    )

    expected_keys = {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "classification_report",
        "confusion_matrix",
    }

    assert expected_keys.issubset(
        results.keys()
    )


def test_metric_ranges():
    """
    All metric values should be between 0 and 1.
    """

    model = get_trained_model()

    x_validation = joblib.load(
        X_VALIDATION_FILE
    )

    y_validation = pd.read_csv(
        Y_VALIDATION_FILE
    )["sentiment"]

    results = evaluate_model(
        model,
        x_validation,
        y_validation
    )

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    ]:

        assert 0.0 <= results[metric] <= 1.0


def test_confusion_matrix_shape():
    """
    Confusion matrix should be 2x2.
    """

    model = get_trained_model()

    x_validation = joblib.load(
        X_VALIDATION_FILE
    )

    y_validation = pd.read_csv(
        Y_VALIDATION_FILE
    )["sentiment"]

    results = evaluate_model(
        model,
        x_validation,
        y_validation
    )

    matrix = results[
        "confusion_matrix"
    ]

    assert matrix.shape == (
        2,
        2,
    )


def test_classification_report_type():
    """
    Classification report should be a string.
    """

    model = get_trained_model()

    x_validation = joblib.load(
        X_VALIDATION_FILE
    )

    y_validation = pd.read_csv(
        Y_VALIDATION_FILE
    )["sentiment"]

    results = evaluate_model(
        model,
        x_validation,
        y_validation
    )

    assert isinstance(
        results["classification_report"],
        str
    )