"""
Create train, validation and test datasets.
"""

from pathlib import Path

from sklearn.model_selection import train_test_split

from src.data.data_loader import load_imdb_dataset


def create_dataset_split():

    train_df, test_df = load_imdb_dataset()

    train_df, validation_df = train_test_split(
        train_df,
        test_size=0.20,
        random_state=42,
        stratify=train_df["sentiment"]
    )

    output_dir = Path("data/interim")

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        output_dir / "train.csv",
        index=False
    )

    validation_df.to_csv(
        output_dir / "validation.csv",
        index=False
    )

    test_df.to_csv(
        output_dir / "test.csv",
        index=False
    )

    print("Dataset split completed.")

    print("Train:", len(train_df))
    print("Validation:", len(validation_df))
    print("Test:", len(test_df))


if __name__ == "__main__":
    create_dataset_split()