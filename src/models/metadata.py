"""
Create metadata for the best machine learning model.
"""

from datetime import datetime
import json
from pathlib import Path

from src.config import MODELS_DIRECTORY
from src.logger import get_logger


logger = get_logger(__name__)


def save_best_model_info(
    model_name: str,
    metrics: dict[str, float]
) -> None:
    """
    Save metadata for the best trained model.

    Parameters
    ----------
    model_name : str
        Name of the best model.

    metrics : dict[str, float]
        Evaluation metrics of the best model.
    """

    model_information = {

        "model_name": model_name,

        "accuracy": metrics["accuracy"],

        "precision": metrics["precision"],

        "recall": metrics["recall"],

        "f1_score": metrics["f1_score"],

        "saved_at": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    }

    output_file = (
        MODELS_DIRECTORY /
        "best_model_info.json"
    )

    logger.info(
        "Saving best model metadata..."
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            model_information,
            file,
            indent=4
        )

    logger.info(
        "Best model metadata saved successfully."
    )