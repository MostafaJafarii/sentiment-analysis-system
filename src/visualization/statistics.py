"""
Statistical analysis utilities for EDA.
"""

import pandas as pd


def add_review_statistics(
    dataset: pd.DataFrame
) -> pd.DataFrame:
    """
    Add review statistics columns.

    Parameters
    ----------
    dataset : pd.DataFrame
        Dataset containing movie reviews.

    Returns
    -------
    pd.DataFrame
        Dataset with additional statistical columns.
    """

    dataset = dataset.copy()

    dataset["word_count"] = (
        dataset["review"]
        .astype(str)
        .apply(
            lambda text: len(text.split())
        )
    )

    dataset["character_count"] = (
        dataset["review"]
        .astype(str)
        .str.len()
    )

    return dataset


def get_dataset_summary(
    dataset: pd.DataFrame
) -> dict[str, object]:
    """
    Calculate dataset summary.

    Parameters
    ----------
    dataset : pd.DataFrame
        Dataset containing movie reviews.

    Returns
    -------
    dict[str, object]
        Dataset summary.
    """

    summary = {

        "rows":
            len(dataset),

        "columns":
            len(dataset.columns),

        "positive_reviews":
            (
                dataset["sentiment"] == 1
            ).sum(),

        "negative_reviews":
            (
                dataset["sentiment"] == 0
            ).sum(),

        "missing_values":
            dataset.isnull().sum().to_dict()

    }

    return summary


def get_word_statistics(
    dataset: pd.DataFrame
) -> dict[str, object]:
    """
    Calculate word count statistics.

    Parameters
    ----------
    dataset : pd.DataFrame
        Dataset containing word counts.

    Returns
    -------
    dict[str, object]
        Word count statistics.
    """

    stats = {

        "mean":
            dataset["word_count"].mean(),

        "median":
            dataset["word_count"].median(),

        "std":
            dataset["word_count"].std(),

        "min":
            dataset["word_count"].min(),

        "max":
            dataset["word_count"].max(),

        "q1":
            dataset["word_count"].quantile(
                0.25
            ),

        "q3":
            dataset["word_count"].quantile(
                0.75
            )

    }

    return stats


def get_character_statistics(
    dataset: pd.DataFrame
) -> dict[str, object]:
    """
    Calculate character count statistics.

    Parameters
    ----------
    dataset : pd.DataFrame
        Dataset containing character counts.

    Returns
    -------
    dict[str, object]
        Character count statistics.
    """

    stats = {

        "mean":
            dataset["character_count"].mean(),

        "median":
            dataset["character_count"].median(),

        "std":
            dataset["character_count"].std(),

        "min":
            dataset["character_count"].min(),

        "max":
            dataset["character_count"].max()

    }

    return stats


def print_summary(
    dataset: pd.DataFrame
) -> None:
    """
    Print complete dataset statistics.

    Parameters
    ----------
    dataset : pd.DataFrame
        Dataset containing movie reviews.
    """

    summary = get_dataset_summary(
        dataset
    )

    words = get_word_statistics(
        dataset
    )

    chars = get_character_statistics(
        dataset
    )

    print("=" * 60)
    print("Dataset Information")
    print("=" * 60)

    print(
        f"Rows              : {summary['rows']}"
    )

    print(
        f"Columns           : {summary['columns']}"
    )

    print(
        f"Positive Reviews  : {summary['positive_reviews']}"
    )

    print(
        f"Negative Reviews  : {summary['negative_reviews']}"
    )

    print()

    print("=" * 60)
    print("Word Statistics")
    print("=" * 60)

    print(
        f"Average Words     : {words['mean']:.2f}"
    )

    print(
        f"Median Words      : {words['median']:.2f}"
    )

    print(
        f"Std Words         : {words['std']:.2f}"
    )

    print(
        f"Minimum Words     : {words['min']}"
    )

    print(
        f"Maximum Words     : {words['max']}"
    )

    print(
        f"Q1                : {words['q1']:.2f}"
    )

    print(
        f"Q3                : {words['q3']:.2f}"
    )

    print()

    print("=" * 60)
    print("Character Statistics")
    print("=" * 60)

    print(
        f"Average Characters : {chars['mean']:.2f}"
    )

    print(
        f"Median Characters  : {chars['median']:.2f}"
    )

    print(
        f"Std Characters     : {chars['std']:.2f}"
    )

    print(
        f"Minimum Characters : {chars['min']}"
    )

    print(
        f"Maximum Characters : {chars['max']}"
    )

    print()

    print("=" * 60)
    print("Missing Values")
    print("=" * 60)

    for column, value in summary[
        "missing_values"
    ].items():

        print(
            f"{column:<20}{value}"
        )