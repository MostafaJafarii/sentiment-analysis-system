"""
Load IMDB dataset from local folders.
"""

from pathlib import Path

import pandas as pd


def read_reviews(folder_path: Path, sentiment: str) -> pd.DataFrame:
    """
    Read all reviews from a folder.
    """

    reviews = []

    for file_path in folder_path.glob("*.txt"):
        review_text = file_path.read_text(
            encoding="utf-8"
        )

        label = 1 if sentiment == "positive" else 0

        reviews.append(
            {
                "review": review_text,
                "sentiment": label
            }
        )

    return pd.DataFrame(reviews)


def load_imdb_dataset():

    dataset_root = Path("data/raw/imdb")

    train_pos = read_reviews(
        dataset_root / "train" / "pos",
        "positive"
    )

    train_neg = read_reviews(
        dataset_root / "train" / "neg",
        "negative"
    )

    test_pos = read_reviews(
        dataset_root / "test" / "pos",
        "positive"
    )

    test_neg = read_reviews(
        dataset_root / "test" / "neg",
        "negative"
    )

    train_df = pd.concat(
        [train_pos, train_neg],
        ignore_index=True
    )

    test_df = pd.concat(
        [test_pos, test_neg],
        ignore_index=True
    )

    return train_df, test_df


if __name__ == "__main__":

    train_df, test_df = load_imdb_dataset()

    print("Train Shape:")
    print(train_df.shape)

    print()

    print("Test Shape:")
    print(test_df.shape)