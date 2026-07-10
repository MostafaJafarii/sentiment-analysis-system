"""
Tests for preprocessing pipeline.
"""

import joblib
import pandas as pd

from src.config import (
    TFIDF_VECTORIZER_FILE,
    X_TRAIN_FILE,
    X_VALIDATION_FILE,
    X_TEST_FILE,
    Y_TRAIN_FILE,
    Y_VALIDATION_FILE,
    Y_TEST_FILE,
)


def test_processed_files_exist():
    """
    All processed files should exist.
    """

    assert TFIDF_VECTORIZER_FILE.exists()

    assert X_TRAIN_FILE.exists()
    assert X_VALIDATION_FILE.exists()
    assert X_TEST_FILE.exists()

    assert Y_TRAIN_FILE.exists()
    assert Y_VALIDATION_FILE.exists()
    assert Y_TEST_FILE.exists()


def test_vectorizer_can_be_loaded():
    """
    TF-IDF vectorizer should be loadable.
    """

    vectorizer = joblib.load(
        TFIDF_VECTORIZER_FILE
    )

    assert vectorizer is not None


def test_feature_matrix_shapes():
    """
    Feature matrices and label vectors should have
    matching sample counts.
    """

    x_train = joblib.load(
        X_TRAIN_FILE
    )

    x_validation = joblib.load(
        X_VALIDATION_FILE
    )

    x_test = joblib.load(
        X_TEST_FILE
    )

    y_train = pd.read_csv(
        Y_TRAIN_FILE
    )["sentiment"]

    y_validation = pd.read_csv(
        Y_VALIDATION_FILE
    )["sentiment"]

    y_test = pd.read_csv(
        Y_TEST_FILE
    )["sentiment"]

    assert x_train.shape[0] == len(y_train)

    assert (
        x_validation.shape[0]
        == len(y_validation)
    )

    assert x_test.shape[0] == len(y_test)


def test_feature_dimensions_are_consistent():
    """
    All datasets should have the same number
    of TF-IDF features.
    """

    x_train = joblib.load(
        X_TRAIN_FILE
    )

    x_validation = joblib.load(
        X_VALIDATION_FILE
    )

    x_test = joblib.load(
        X_TEST_FILE
    )

    assert (
        x_train.shape[1]
        == x_validation.shape[1]
    )

    assert (
        x_train.shape[1]
        == x_test.shape[1]
    )


def test_label_values():
    """
    Labels should contain only binary values.
    """

    y_train = pd.read_csv(
        Y_TRAIN_FILE
    )["sentiment"]

    y_validation = pd.read_csv(
        Y_VALIDATION_FILE
    )["sentiment"]

    y_test = pd.read_csv(
        Y_TEST_FILE
    )["sentiment"]

    assert set(y_train.unique()) == {0, 1}

    assert set(y_validation.unique()) == {0, 1}

    assert set(y_test.unique()) == {0, 1}