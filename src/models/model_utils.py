"""
Utility functions for saving and loading machine learning models.
"""

from pathlib import Path

import joblib
from sklearn.base import BaseEstimator

from src.config import MODELS_DIRECTORY
from src.logger import get_logger


logger = get_logger(__name__)


def save_model(
    model: BaseEstimator,
    model_name: str
) -> Path:
    """
    Save a trained machine learning model.

    Parameters
    ----------
    model : BaseEstimator
        Trained machine learning model.

    model_name : str
        Model file name.

    Returns
    -------
    Path
        Path to the saved model.
    """

    MODELS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        MODELS_DIRECTORY /
        f"{model_name}.joblib"
    )

    logger.info(
        "Saving model: %s",
        model_name
    )

    joblib.dump(
        model,
        model_path
    )

    logger.info(
        "Model saved successfully: %s",
        model_path
    )

    return model_path


def load_model(
    model_name: str
) -> BaseEstimator:
    """
    Load a trained machine learning model.

    Parameters
    ----------
    model_name : str
        Model file name.

    Returns
    -------
    BaseEstimator
        Loaded machine learning model.
    """

    model_path = (
        MODELS_DIRECTORY /
        f"{model_name}.joblib"
    )

    if not model_path.exists():

        logger.error(
            "Model not found: %s",
            model_name
        )

        raise FileNotFoundError(
            f"Model '{model_name}' does not exist."
        )

    logger.info(
        "Loading model: %s",
        model_name
    )

    model = joblib.load(
        model_path
    )

    logger.info(
        "Model loaded successfully."
    )

    return model