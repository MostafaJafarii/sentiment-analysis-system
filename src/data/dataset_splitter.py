"""
Create train, validation and test datasets.
"""

from pathlib import Path

from sklearn.model_selection import train_test_split

from src.config import (
    INTERIM_DATA_DIRECTORY,
    RANDOM_STATE,
    VALIDATION_SIZE
)
from src.data.data_loader import load_imdb_dataset
from src.logger import get_logger


logger = get_logger(__name__)


def create_dataset_split() -> None:
    """
    Create train, validation and test datasets
    and save them into the interim directory.
    """

    logger.info(
        "Loading IMDB dataset..."
    )

    train_df, test_df = load_imdb_dataset()

    logger.info(
        "Creating validation split..."
    )

    train_df, validation_df = train_test_split(
        train_df,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=train_df["sentiment"]
    )

    INTERIM_DATA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        INTERIM_DATA_DIRECTORY / "train.csv",
        index=False
    )

    validation_df.to_csv(
        INTERIM_DATA_DIRECTORY / "validation.csv",
        index=False
    )

    test_df.to_csv(
        INTERIM_DATA_DIRECTORY / "test.csv",
        index=False
    )

    logger.info(
        "Dataset split completed successfully."
    )

    logger.info(
        "Train samples: %d",
        len(train_df)
    )

    logger.info(
        "Validation samples: %d",
        len(validation_df)
    )

    logger.info(
        "Test samples: %d",
        len(test_df)
    )


if __name__ == "__main__":
    create_dataset_split()