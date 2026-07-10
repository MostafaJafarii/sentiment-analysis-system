"""
Evaluate machine learning models.
"""

from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

from src.logger import get_logger


logger = get_logger(__name__)


def evaluate_model(
    model: BaseEstimator,
    X_validation,
    y_validation
) -> dict:
    """
    Evaluate a trained machine learning model.

    Parameters
    ----------
    model : BaseEstimator
        Trained machine learning model.

    X_validation
        Validation feature matrix.

    y_validation
        Validation labels.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """

    logger.info(
        "Evaluating %s...",
        model.__class__.__name__
    )

    predictions = model.predict(
        X_validation
    )

    results = {

        "accuracy": accuracy_score(
            y_validation,
            predictions
        ),

        "precision": precision_score(
            y_validation,
            predictions
        ),

        "recall": recall_score(
            y_validation,
            predictions
        ),

        "f1_score": f1_score(
            y_validation,
            predictions
        ),

        "classification_report":
            classification_report(
                y_validation,
                predictions,
                target_names=[
                    "negative",
                    "positive"
                ]
            ),

        "confusion_matrix":
            confusion_matrix(
                y_validation,
                predictions
            )

    }

    logger.info(
        "%s evaluation completed.",
        model.__class__.__name__
    )

    return results