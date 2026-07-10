"""
Tests for the IMDB data loader.
"""

from src.data.data_loader import load_imdb_dataset


EXPECTED_TRAIN_SIZE = 25000
EXPECTED_TEST_SIZE = 25000

EXPECTED_COLUMNS = [
    "review",
    "sentiment"
]


def test_load_imdb_dataset():
    """
    Test loading the IMDB dataset.
    """

    train_df, test_df = load_imdb_dataset()

    # Dataset size
    assert len(train_df) == EXPECTED_TRAIN_SIZE
    assert len(test_df) == EXPECTED_TEST_SIZE

    # Expected columns
    assert list(train_df.columns) == EXPECTED_COLUMNS
    assert list(test_df.columns) == EXPECTED_COLUMNS

    # Dataset must not be empty
    assert not train_df.empty
    assert not test_df.empty

    # Labels must be binary
    assert set(train_df["sentiment"].unique()) == {0, 1}
    assert set(test_df["sentiment"].unique()) == {0, 1}

    # Reviews should not be empty
    assert train_df["review"].str.len().gt(0).all()
    assert test_df["review"].str.len().gt(0).all()