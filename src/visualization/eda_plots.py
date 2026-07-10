"""
EDA plotting functions.
"""

import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    MAX_REVIEW_LENGTH,
)

from src.logger import get_logger

from src.visualization.visualization_utils import (
    create_figure,
    apply_plot_style,
    save_plot,
    add_value_labels,
    add_mean_and_median,
)

logger = get_logger(__name__)


def remove_outliers(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove extremely long reviews for visualization only.
    """

    return dataset[
        dataset["word_count"] <= MAX_REVIEW_LENGTH
    ].copy()


def plot_sentiment_distribution(
    dataset: pd.DataFrame,
) -> None:
    """
    Plot sentiment distribution.
    """

    create_figure()

    counts = (
        dataset["sentiment"]
        .value_counts()
        .sort_index()
    )

    ax = counts.plot(
        kind="bar"
    )

    add_value_labels(ax)

    total_reviews = counts.sum()

    plt.title(
        "Sentiment Distribution",
        fontsize=14,
    )

    plt.xlabel(
        "Sentiment"
    )

    plt.ylabel(
        "Number of Reviews"
    )

    plt.xticks(
        [0, 1],
        ["Negative", "Positive"]
    )

    for index, value in enumerate(counts):

        percentage = (
            value / total_reviews
        ) * 100

        plt.text(
            index,
            value,
            f"{percentage:.1f}%",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    apply_plot_style()

    save_plot(
        "sentiment_distribution.png"
    )


def plot_average_review_length(
    dataset: pd.DataFrame,
) -> None:
    """
    Plot average review length.
    """

    create_figure()

    averages = (
        dataset
        .groupby("sentiment")["word_count"]
        .mean()
    )

    ax = averages.plot(
        kind="bar"
    )

    add_value_labels(ax)

    plt.title(
        "Average Review Length",
        fontsize=14,
    )

    plt.xlabel(
        "Sentiment"
    )

    plt.ylabel(
        "Average Number of Words"
    )

    plt.xticks(
        [0, 1],
        ["Negative", "Positive"]
    )

    apply_plot_style()

    save_plot(
        "average_review_length.png"
    )


def plot_review_length_distribution(
    dataset: pd.DataFrame,
) -> None:
    """
    Plot review length distribution.
    """

    filtered_dataset = remove_outliers(
        dataset
    )

    create_figure()

    plt.hist(
        filtered_dataset["word_count"],
        bins=40,
    )

    add_mean_and_median(
        filtered_dataset["word_count"]
    )

    plt.title(
        "Review Length Distribution",
        fontsize=14,
    )

    plt.xlabel(
        "Number of Words"
    )

    plt.ylabel(
        "Number of Reviews"
    )

    plt.text(
        0.98,
        0.95,
        f"Reviews: {len(filtered_dataset)}",
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round",
            alpha=0.20,
        ),
    )

    apply_plot_style()

    save_plot(
        "review_length_distribution.png"
    )


def plot_review_length_by_sentiment(
    dataset: pd.DataFrame,
) -> None:
    """
    Plot review length grouped by sentiment.
    """

    filtered_dataset = remove_outliers(
        dataset
    )

    create_figure()

    positive_reviews = filtered_dataset[
        filtered_dataset["sentiment"] == 1
    ]["word_count"]

    negative_reviews = filtered_dataset[
        filtered_dataset["sentiment"] == 0
    ]["word_count"]

    plt.hist(
        positive_reviews,
        bins=35,
        alpha=0.65,
        label="Positive",
    )

    plt.hist(
        negative_reviews,
        bins=35,
        alpha=0.65,
        label="Negative",
    )

    plt.title(
        "Review Length by Sentiment",
        fontsize=14,
    )

    plt.xlabel(
        "Number of Words"
    )

    plt.ylabel(
        "Number of Reviews"
    )

    plt.legend()

    apply_plot_style()

    save_plot(
        "review_length_by_sentiment.png"
    )


def plot_review_length_boxplot(
    dataset: pd.DataFrame,
) -> None:
    """
    Plot review length box plot grouped by sentiment.
    """

    filtered_dataset = remove_outliers(
        dataset
    )

    create_figure()

    filtered_dataset.boxplot(
        column="word_count",
        by="sentiment",
        grid=False,
    )

    plt.grid(
        axis="y",
        alpha=0.30,
    )

    plt.suptitle("")

    plt.title(
        "Review Length Box Plot",
        fontsize=14,
    )

    plt.xlabel(
        "Sentiment"
    )

    plt.ylabel(
        "Number of Words"
    )

    plt.xticks(
        [1, 2],
        ["Negative", "Positive"]
    )

    apply_plot_style()

    save_plot(
        "review_length_boxplot.png"
    )


def generate_all_eda_plots(
    dataset: pd.DataFrame,
) -> None:
    """
    Generate all EDA figures.
    """

    logger.info(
        "Generating EDA figures..."
    )

    plot_sentiment_distribution(
        dataset
    )

    logger.info(
        "Sentiment distribution generated."
    )

    plot_average_review_length(
        dataset
    )

    logger.info(
        "Average review length generated."
    )

    plot_review_length_distribution(
        dataset
    )

    logger.info(
        "Review length distribution generated."
    )

    plot_review_length_by_sentiment(
        dataset
    )

    logger.info(
        "Review length by sentiment generated."
    )

    plot_review_length_boxplot(
        dataset
    )

    logger.info(
        "Review length boxplot generated."
    )

    logger.info(
        "All EDA figures generated successfully."
    )