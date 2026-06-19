"""
Preprocessing pipeline.
"""

from pathlib import Path

import pandas as pd
import joblib

from src.preprocessing.text_cleaner import clean_text
from src.preprocessing.feature_extractor import create_tfidf


def run_preprocessing():

    train_df = pd.read_csv(
        "data/interim/train.csv"
    )

    validation_df = pd.read_csv(
        "data/interim/validation.csv"
    )

    test_df = pd.read_csv(
        "data/interim/test.csv"
    )

    print("Cleaning train dataset...")

    train_df["review"] = (
        train_df["review"]
        .astype(str)
        .apply(clean_text)
    )

    validation_df["review"] = (
        validation_df["review"]
        .astype(str)
        .apply(clean_text)
    )

    test_df["review"] = (
        test_df["review"]
        .astype(str)
        .apply(clean_text)
    )

    vectorizer = create_tfidf()

    X_train = vectorizer.fit_transform(
        train_df["review"]
    )

    X_validation = vectorizer.transform(
        validation_df["review"]
    )

    X_test = vectorizer.transform(
        test_df["review"]
    )

    output_dir = Path(
        "data/processed"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        vectorizer,
        output_dir / "tfidf_vectorizer.joblib"
    )

    joblib.dump(
        X_train,
        output_dir / "X_train.joblib"
    )

    joblib.dump(
        X_validation,
        output_dir / "X_validation.joblib"
    )

    joblib.dump(
        X_test,
        output_dir / "X_test.joblib"
    )

    train_df["sentiment"].to_csv(
        output_dir / "y_train.csv",
        index=False
    )

    validation_df["sentiment"].to_csv(
        output_dir / "y_validation.csv",
        index=False
    )

    test_df["sentiment"].to_csv(
        output_dir / "y_test.csv",
        index=False
    )

    print("Preprocessing completed.")


if __name__ == "__main__":
    run_preprocessing()