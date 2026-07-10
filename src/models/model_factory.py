"""
Create and manage machine learning models.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.config import (
    LOGISTIC_REGRESSION_MAX_ITER,
    RANDOM_FOREST_ESTIMATORS,
    RANDOM_FOREST_N_JOBS,
    RANDOM_STATE
)


def get_models() -> dict:
    """
    Create all machine learning models used in the project.

    Returns
    -------
    dict
        Dictionary containing initialized models.
    """

    return {

        "logistic_regression": LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=LOGISTIC_REGRESSION_MAX_ITER
        ),

        "naive_bayes": MultinomialNB(),

        "linear_svm": LinearSVC(
            random_state=RANDOM_STATE
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=RANDOM_FOREST_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=RANDOM_FOREST_N_JOBS
        )

    }