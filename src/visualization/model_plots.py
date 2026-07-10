"""
Visualization functions for machine learning model evaluation.
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    roc_curve,
)

from src.config import (
    BEST_MODEL_FILE,
    MODEL_FIGURES_DIRECTORY,
    MODEL_RESULTS_FILE,
    X_TEST_FILE,
    Y_TEST_FILE,
)

from src.logger import get_logger

from src.visualization.visualization_utils import (
    apply_plot_style,
    create_figure,
    save_plot,
)

logger = get_logger(__name__)


def load_results() -> pd.DataFrame:
    """
    Load evaluation results.
    """

    return pd.read_csv(
        MODEL_RESULTS_FILE
    )


def load_test_data() -> tuple:
    """
    Load best model and test dataset.
    """

    model = joblib.load(
        BEST_MODEL_FILE
    )

    x_test = joblib.load(
        X_TEST_FILE
    )

    y_test = pd.read_csv(
        Y_TEST_FILE
    )["sentiment"]

    return model, x_test, y_test


def plot_metric(
    dataframe: pd.DataFrame,
    metric: str,
    filename: str,
) -> None:
    """
    Plot one evaluation metric.
    """

    create_figure()

    bars = plt.bar(
        dataframe["Model"],
        dataframe[metric],
    )

    plt.title(
        f"{metric} Comparison",
        fontsize=14,
    )

    plt.xlabel(
        "Machine Learning Model"
    )

    plt.ylabel(
        metric
    )

    plt.xticks(
        rotation=15
    )

    best_index = dataframe[
        metric
    ].idxmax()

    bars[
        best_index
    ].set_linewidth(2)

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            height + 0.003,
            f"{height:.3f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    apply_plot_style()

    save_plot(
        filename,
        MODEL_FIGURES_DIRECTORY,
    )


def generate_model_plots() -> None:
    """
    Generate comparison charts.
    """

    dataframe = load_results()

    metrics = [

        ("Accuracy", "accuracy_comparison.png"),

        ("Precision", "precision_comparison.png"),

        ("Recall", "recall_comparison.png"),

        ("F1 Score", "f1_score_comparison.png"),

    ]

    for metric, filename in metrics:

        logger.info(
            "Generating %s plot...",
            metric,
        )

        plot_metric(
            dataframe,
            metric,
            filename,
        )

    logger.info(
        "All model comparison plots generated."
    )


def generate_confusion_matrix() -> None:
    """
    Generate confusion matrix.
    """

    logger.info(
        "Generating confusion matrix..."
    )

    model, x_test, y_test = load_test_data()

    predictions = model.predict(
        x_test
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    create_figure(
        width=8,
        height=6,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Negative",
            "Positive",
        ],
    )

    display.plot(
        cmap="Blues",
        colorbar=False,
    )

    plt.title(
        "Confusion Matrix",
        fontsize=15,
    )

    plt.grid(False)

    save_plot(
        "confusion_matrix.png",
        MODEL_FIGURES_DIRECTORY,
    )

    logger.info(
        "Confusion matrix generated."
    )


def generate_roc_curve() -> None:
    """
    Generate ROC curve.
    """

    logger.info(
        "Generating ROC curve..."
    )

    model, x_test, y_test = load_test_data()

    if not hasattr(
        model,
        "predict_proba",
    ):

        logger.warning(
            "ROC curve skipped because model "
            "does not support predict_proba."
        )

        return

    probabilities = model.predict_proba(
        x_test
    )[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities,
    )

    roc_auc = auc(
        fpr,
        tpr,
    )

    create_figure(
        width=8,
        height=6,
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"AUC = {roc_auc:.3f}",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve"
    )

    plt.legend()

    apply_plot_style()

    save_plot(
        "roc_curve.png",
        MODEL_FIGURES_DIRECTORY,
    )

    logger.info(
        "ROC curve generated."
    )