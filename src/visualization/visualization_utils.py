"""
Utility functions for creating and saving visualizations.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

from src.config import (
    DEFAULT_FIGURE_WIDTH,
    DEFAULT_FIGURE_HEIGHT,
    DEFAULT_FIGURE_DPI,
    MATPLOTLIB_STYLE,
    EDA_FIGURES_DIRECTORY,
)


def create_output_directory(
    output_directory: Path,
) -> None:
    """
    Create output directory if it does not exist.

    Parameters
    ----------
    output_directory : Path
        Directory where figures will be stored.
    """

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def create_figure(
    width: int = DEFAULT_FIGURE_WIDTH,
    height: int = DEFAULT_FIGURE_HEIGHT,
) -> None:
    """
    Create a standard matplotlib figure.

    Parameters
    ----------
    width : int, optional
        Figure width in inches.

    height : int, optional
        Figure height in inches.
    """

    plt.style.use(MATPLOTLIB_STYLE)

    plt.figure(
        figsize=(width, height),
        dpi=DEFAULT_FIGURE_DPI,
    )


def apply_plot_style() -> None:
    """
    Apply common style to plots.
    """

    plt.grid(
        alpha=0.30,
        linestyle="--",
    )

    plt.tight_layout()


def save_plot(
    filename: str,
    output_directory: Path | None = None,
) -> None:
    """
    Save the current figure.

    Parameters
    ----------
    filename : str
        Output image filename.

    output_directory : Path | None, optional
        Target directory for the figure.
        If None, EDA_FIGURES_DIRECTORY is used.
    """

    if output_directory is None:
        output_directory = EDA_FIGURES_DIRECTORY

    create_output_directory(
        output_directory
    )

    plt.savefig(
        output_directory / filename,
        bbox_inches="tight",
    )

    plt.close()


def add_value_labels(
    ax: Axes,
) -> None:
    """
    Display values above bars.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes.
    """

    for container in ax.containers:

        ax.bar_label(
            container,
            padding=3,
        )


def add_mean_and_median(
    values: pd.Series,
) -> None:
    """
    Draw mean and median reference lines.

    Parameters
    ----------
    values : pd.Series
        Numeric values.
    """

    mean_value = values.mean()

    median_value = values.median()

    plt.axvline(
        mean_value,
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_value:.1f}",
    )

    plt.axvline(
        median_value,
        linestyle=":",
        linewidth=2,
        label=f"Median = {median_value:.1f}",
    )

    plt.legend()