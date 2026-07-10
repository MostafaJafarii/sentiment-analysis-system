"""
Manage evaluation reports for trained models.
"""

from datetime import datetime
import json
import platform

import pandas as pd

from src.config import (
    EVALUATION_REPORT_DIRECTORY
)
from src.logger import get_logger


logger = get_logger(__name__)


def save_results(
    results: dict[str, dict[str, float]]
) -> None:
    """
    Save evaluation results in CSV and JSON formats.

    Parameters
    ----------
    results : dict[str, dict[str, float]]
        Evaluation metrics for all models.
    """

    EVALUATION_REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    for model_name, metrics in results.items():

        rows.append({

            "Model": model_name,

            "Accuracy": metrics["accuracy"],

            "Precision": metrics["precision"],

            "Recall": metrics["recall"],

            "F1 Score": metrics["f1_score"]

        })

    dataframe = pd.DataFrame(rows)

    dataframe.to_csv(
        EVALUATION_REPORT_DIRECTORY /
        "model_results.csv",
        index=False
    )

    with open(
        EVALUATION_REPORT_DIRECTORY /
        "model_results.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    logger.info(
        "Evaluation results saved successfully."
    )


def create_evaluation_report(
    results: dict[str, dict[str, float]],
    best_model: str
) -> None:
    """
    Generate a readable evaluation report.

    Parameters
    ----------
    results : dict[str, dict[str, float]]
        Evaluation metrics for all models.

    best_model : str
        Name of the best model.
    """

    EVALUATION_REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path = (
        EVALUATION_REPORT_DIRECTORY /
        "evaluation_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("=" * 70 + "\n")
        file.write("Movie Review Sentiment Analysis\n")
        file.write("Evaluation Report\n")
        file.write("=" * 70 + "\n\n")

        file.write(
            f"Generated : {datetime.now()}\n"
        )

        file.write(
            f"Python    : {platform.python_version()}\n\n"
        )

        file.write("=" * 70 + "\n")
        file.write("Best Model\n")
        file.write("=" * 70 + "\n")

        metrics = results[best_model]

        file.write(
            f"Name       : {best_model}\n"
        )

        file.write(
            f"Accuracy   : {metrics['accuracy']:.4f}\n"
        )

        file.write(
            f"Precision  : {metrics['precision']:.4f}\n"
        )

        file.write(
            f"Recall     : {metrics['recall']:.4f}\n"
        )

        file.write(
            f"F1 Score   : {metrics['f1_score']:.4f}\n\n"
        )

        file.write("=" * 70 + "\n")
        file.write("All Models\n")
        file.write("=" * 70 + "\n\n")

        for model_name, metrics in results.items():

            file.write(
                f"{model_name}\n"
            )

            file.write(
                f"Accuracy : {metrics['accuracy']:.4f}\n"
            )

            file.write(
                f"Precision: {metrics['precision']:.4f}\n"
            )

            file.write(
                f"Recall   : {metrics['recall']:.4f}\n"
            )

            file.write(
                f"F1 Score : {metrics['f1_score']:.4f}\n\n"
            )

    logger.info(
        "Evaluation report generated successfully."
    )