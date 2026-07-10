"""
Train machine learning models.
"""

from sklearn.base import BaseEstimator

from src.logger import get_logger


logger = get_logger(__name__)


def train_model(
    model: BaseEstimator,
    X_train,
    y_train
) -> BaseEstimator:
    """
    Train a machine learning model.

    Parameters
    ----------
    model : BaseEstimator
        Machine learning model.

    X_train
        Training feature matrix.

    y_train
        Training labels.

    Returns
    -------
    BaseEstimator
        Trained machine learning model.
    """

    logger.info(
        "Training %s...",
        model.__class__.__name__
    )

    model.fit(
        X_train,
        y_train
    )

    logger.info(
        "%s training completed.",
        model.__class__.__name__
    )

    return model