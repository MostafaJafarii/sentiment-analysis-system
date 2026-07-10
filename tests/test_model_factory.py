"""
Tests for model factory.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.models.model_factory import get_models


EXPECTED_MODELS = {
    "logistic_regression",
    "naive_bayes",
    "linear_svm",
    "random_forest",
}


def test_get_models_returns_dictionary():
    """
    Model factory should return a dictionary.
    """

    models = get_models()

    assert isinstance(models, dict)


def test_expected_models_exist():
    """
    All expected models should be available.
    """

    models = get_models()

    assert set(models.keys()) == EXPECTED_MODELS


def test_logistic_regression_type():
    """
    Logistic Regression model should be correct.
    """

    models = get_models()

    assert isinstance(
        models["logistic_regression"],
        LogisticRegression,
    )


def test_naive_bayes_type():
    """
    Naive Bayes model should be correct.
    """

    models = get_models()

    assert isinstance(
        models["naive_bayes"],
        MultinomialNB,
    )


def test_linear_svm_type():
    """
    Linear SVM model should be correct.
    """

    models = get_models()

    assert isinstance(
        models["linear_svm"],
        LinearSVC,
    )


def test_random_forest_type():
    """
    Random Forest model should be correct.
    """

    models = get_models()

    assert isinstance(
        models["random_forest"],
        RandomForestClassifier,
    )