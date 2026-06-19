from src.data.data_loader import load_imdb_dataset


def test_loader():

    train_df, test_df = load_imdb_dataset()

    assert len(train_df) == 25000

    assert len(test_df) == 25000

    print("Dataset loaded correctly.")


if __name__ == "__main__":
    test_loader()