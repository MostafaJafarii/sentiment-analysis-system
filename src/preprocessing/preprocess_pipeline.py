"""
Preprocessing pipeline.
"""

import joblib
import pandas as pd

from src.config import (
    TRAIN_DATA_FILE,
    VALIDATION_DATA_FILE,
    TEST_DATA_FILE,
    PROCESSED_DATA_DIRECTORY,
    TFIDF_VECTORIZER_FILE,
    X_TRAIN_FILE,
    X_VALIDATION_FILE,
    X_TEST_FILE,
    Y_TRAIN_FILE,
    Y_VALIDATION_FILE,
    Y_TEST_FILE
)
from src.logger import get_logger

from src.preprocessing.text_cleaner import clean_text
from src.preprocessing.feature_extractor import create_tfidf


logger = get_logger(__name__)


def clean_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Clean all reviews in a dataframe.
    """

    dataframe["review"] = (
        dataframe["review"]
        .astype(str)
        .apply(clean_text)
    )

    return dataframe


def save_labels(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> None:
    """
    Save sentiment labels.
    """

    train_df["sentiment"].to_csv(
        Y_TRAIN_FILE,
        index=False
    )

    validation_df["sentiment"].to_csv(
        Y_VALIDATION_FILE,
        index=False
    )

    test_df["sentiment"].to_csv(
        Y_TEST_FILE,
        index=False
    )


def run_preprocessing() -> None:
    """
    Execute preprocessing pipeline.
    """

    logger.info(
        "Loading datasets..."
    )

    train_df = pd.read_csv(
        TRAIN_DATA_FILE
    )

    validation_df = pd.read_csv(
        VALIDATION_DATA_FILE
    )

    test_df = pd.read_csv(
        TEST_DATA_FILE
    )

    logger.info(
        "Cleaning reviews..."
    )

    train_df = clean_dataframe(train_df)

    validation_df = clean_dataframe(
        validation_df
    )

    test_df = clean_dataframe(
        test_df
    )

    logger.info(
        "Creating TF-IDF features..."
    )

    vectorizer = create_tfidf()

    x_train = vectorizer.fit_transform(
        train_df["review"]
    )

    x_validation = vectorizer.transform(
        validation_df["review"]
    )

    x_test = vectorizer.transform(
        test_df["review"]
    )

    PROCESSED_DATA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        vectorizer,
        TFIDF_VECTORIZER_FILE
    )

    joblib.dump(
        x_train,
        X_TRAIN_FILE
    )

    joblib.dump(
        x_validation,
        X_VALIDATION_FILE
    )

    joblib.dump(
        x_test,
        X_TEST_FILE
    )

    save_labels(
        train_df,
        validation_df,
        test_df
    )

    logger.info(
        "Preprocessing completed successfully."
    )


if __name__ == "__main__":
    run_preprocessing()