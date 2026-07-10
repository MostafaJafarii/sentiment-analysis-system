"""
Evaluation visualization pipeline.
"""

from src.logger import get_logger

from src.visualization.model_plots import (
    generate_confusion_matrix,
    generate_model_plots,
    generate_roc_curve,
)

logger = get_logger(__name__)


def run_evaluation() -> None:
    """
    Generate all evaluation visualizations.
    """

    logger.info(
        "Starting evaluation visualization pipeline..."
    )

    generate_model_plots()

    generate_confusion_matrix()

    generate_roc_curve()

    logger.info(
        "Evaluation visualization pipeline completed successfully."
    )


if __name__ == "__main__":
    run_evaluation()