"""
Exploratory Data Analysis.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def run_eda():

    train_df = pd.read_csv(
        "data/interim/train.csv"
    )

    print(train_df.head())

    print()

    print(train_df["sentiment"].value_counts())

    Path("reports/plots").mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=train_df,
        x="sentiment"
    )

    plt.title(
        "Sentiment Distribution"
    )

    plt.savefig(
        "reports/plots/sentiment_distribution.png"
    )

    plt.show()


if __name__ == "__main__":
    run_eda()