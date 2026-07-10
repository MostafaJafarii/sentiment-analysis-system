"""
Train, evaluate, compare and save all machine learning models.
"""

import shutil

import joblib
import pandas as pd

from src.config import (
    X_TRAIN_FILE,
    X_VALIDATION_FILE,
    Y_TRAIN_FILE,
    Y_VALIDATION_FILE,
    MODELS_DIRECTORY,
    BEST_MODEL_FILE
)
from src.logger import get_logger

from src.models.model_factory import get_models
from src.models.trainer import train_model
from src.models.evaluator import evaluate_model
from src.models.model_utils import save_model
from src.models.metadata import save_best_model_info
from src.models.report_manager import (
    save_results,
    create_evaluation_report
)


logger = get_logger(__name__)


def load_processed_data():
    """
    Load processed training and validation datasets.
    """

    logger.info(
        "Loading processed datasets..."
    )

    x_train = joblib.load(
        X_TRAIN_FILE
    )

    x_validation = joblib.load(
        X_VALIDATION_FILE
    )

    y_train = pd.read_csv(
        Y_TRAIN_FILE
    )["sentiment"]

    y_validation = pd.read_csv(
        Y_VALIDATION_FILE
    )["sentiment"]

    return (
        x_train,
        x_validation,
        y_train,
        y_validation
    )


def main() -> None:
    """
    Train, evaluate and compare all machine learning models.
    """

    (
        x_train,
        x_validation,
        y_train,
        y_validation
    ) = load_processed_data()

    models = get_models()

    results = {}

    logger.info(
        "Training %d models...",
        len(models)
    )

    for model_name, model in models.items():

        trained_model = train_model(
            model,
            x_train,
            y_train
        )

        metrics = evaluate_model(
            trained_model,
            x_validation,
            y_validation
        )

        save_model(
            trained_model,
            model_name
        )

        results[model_name] = {

            "accuracy":
                metrics["accuracy"],

            "precision":
                metrics["precision"],

            "recall":
                metrics["recall"],

            "f1_score":
                metrics["f1_score"]

        }

    save_results(
        results
    )

    best_model_name = max(

        results,

        key=lambda name:
        results[name]["f1_score"]

    )

    logger.info(
        "Best model: %s",
        best_model_name
    )

    save_best_model_info(
        best_model_name,
        results[best_model_name]
    )

    create_evaluation_report(
        results,
        best_model_name
    )

    shutil.copy2(

        MODELS_DIRECTORY /
        f"{best_model_name}.joblib",

        BEST_MODEL_FILE

    )

    logger.info(
        "Best model copied to %s",
        BEST_MODEL_FILE
    )

    logger.info(
        "Training pipeline completed successfully."
    )


if __name__ == "__main__":
    main()