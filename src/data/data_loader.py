"""
Load the IMDB dataset from local folders.
"""

from pathlib import Path

import pandas as pd

from src.config import IMDB_DATASET_DIRECTORY
from src.logger import get_logger


logger = get_logger(__name__)


def read_reviews(
    folder_path: Path,
    sentiment: str
) -> pd.DataFrame:
    """
    Read all review files from a sentiment folder.

    Parameters
    ----------
    folder_path : Path
        Directory containing review text files.

    sentiment : str
        Review sentiment ("positive" or "negative").

    Returns
    -------
    pd.DataFrame
        DataFrame containing reviews and labels.
    """

    logger.info(
        "Reading %s reviews from %s",
        sentiment,
        folder_path
    )

    reviews = []

    label = (
        1
        if sentiment == "positive"
        else 0
    )

    for file_path in folder_path.glob("*.txt"):

        review_text = file_path.read_text(
            encoding="utf-8"
        )

        reviews.append(
            {
                "review": review_text,
                "sentiment": label
            }
        )

    dataframe = pd.DataFrame(reviews)

    logger.info(
        "%d %s reviews loaded.",
        len(dataframe),
        sentiment
    )

    return dataframe


def load_imdb_dataset(
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the IMDB dataset.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Training and testing datasets.
    """

    logger.info(
        "Loading IMDB dataset..."
    )

    train_pos = read_reviews(
        IMDB_DATASET_DIRECTORY / "train" / "pos",
        "positive"
    )

    train_neg = read_reviews(
        IMDB_DATASET_DIRECTORY / "train" / "neg",
        "negative"
    )

    test_pos = read_reviews(
        IMDB_DATASET_DIRECTORY / "test" / "pos",
        "positive"
    )

    test_neg = read_reviews(
        IMDB_DATASET_DIRECTORY / "test" / "neg",
        "negative"
    )

    train_df = pd.concat(
        [
            train_pos,
            train_neg
        ],
        ignore_index=True
    )

    test_df = pd.concat(
        [
            test_pos,
            test_neg
        ],
        ignore_index=True
    )

    logger.info(
        "Dataset loaded successfully."
    )

    logger.info(
        "Train samples: %d",
        len(train_df)
    )

    logger.info(
        "Test samples: %d",
        len(test_df)
    )

    return (
        train_df,
        test_df
    )


if __name__ == "__main__":

    train_df, test_df = load_imdb_dataset()

    logger.info(
        "Train shape: %s",
        train_df.shape
    )

    logger.info(
        "Test shape: %s",
        test_df.shape
    )