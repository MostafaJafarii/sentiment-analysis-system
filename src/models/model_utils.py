"""
Utility functions for saving and loading machine learning models.
"""

from pathlib import Path

import joblib
from sklearn.base import BaseEstimator

from src.config import (
    MODELS_DIRECTORY,
    BEST_MODEL_FILE
)

from src.logger import get_logger

logger = get_logger(__name__)

# =============================================================================
# Supported Models
# =============================================================================

AVAILABLE_MODELS = {

    "logistic_regression": "Logistic Regression",

    "naive_bayes": "Naive Bayes",

    "linear_svm": "Support Vector Machine",

    "random_forest": "Random Forest"

}

# =============================================================================
# Save Model
# =============================================================================

def save_model(
    model: BaseEstimator,
    model_name: str
) -> Path:

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


# =============================================================================
# Load Model
# =============================================================================

def load_model(
    model_name: str
) -> BaseEstimator:

    if model_name == "best_model":

        model_path = BEST_MODEL_FILE

    else:

        validate_model_name(
            model_name
        )

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


# =============================================================================
# Validation
# =============================================================================

def validate_model_name(
    model_name: str
) -> None:

    if model_name not in AVAILABLE_MODELS:

        raise ValueError(
            f"Unsupported model: {model_name}"
        )


# =============================================================================
# Information
# =============================================================================

def get_available_models() -> dict:

    return AVAILABLE_MODELS.copy()


def get_available_model_names() -> list[str]:

    return list(
        AVAILABLE_MODELS.keys()
    )


def get_model_display_name(
    model_name: str
) -> str:

    if model_name == "best_model":

        return "Best Model"

    validate_model_name(
        model_name
    )

    return AVAILABLE_MODELS[
        model_name
    ]