"""
Run complete Exploratory Data Analysis (EDA).
"""

import pandas as pd

from src.config import TRAIN_DATA_FILE
from src.logger import get_logger

from src.visualization.statistics import (
    add_review_statistics,
    print_summary,
)

from src.visualization.eda_plots import (
    generate_all_eda_plots,
)

logger = get_logger(__name__)


def run_eda() -> None:
    """
    Execute the complete Exploratory Data Analysis (EDA) pipeline.
    """

    logger.info(
        "Loading training dataset..."
    )

    dataset = pd.read_csv(
        TRAIN_DATA_FILE
    )

    logger.info(
        "Loaded %d reviews.",
        len(dataset)
    )

    logger.info(
        "Calculating review statistics..."
    )

    dataset = add_review_statistics(
        dataset
    )

    print_summary(
        dataset
    )

    generate_all_eda_plots(
        dataset
    )

    logger.info(
        "EDA completed successfully."
    )


if __name__ == "__main__":
    run_eda()