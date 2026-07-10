"""
Central configuration for the Sentiment Analysis project.
"""

from pathlib import Path

# =============================================================================
# Project Root
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =============================================================================
# Random State
# =============================================================================

RANDOM_STATE = 42

VALIDATION_SIZE = 0.20

# =============================================================================
# Machine Learning Settings
# =============================================================================

LOGISTIC_REGRESSION_MAX_ITER = 1000

RANDOM_FOREST_ESTIMATORS = 200

RANDOM_FOREST_N_JOBS = -1

# =============================================================================
# TF-IDF Settings
# =============================================================================

TFIDF_MAX_FEATURES = 10000

TFIDF_NGRAM_RANGE = (1, 2)

# =============================================================================
# Data Directories
# =============================================================================

DATA_DIRECTORY = PROJECT_ROOT / "data"

RAW_DATA_DIRECTORY = DATA_DIRECTORY / "raw"

INTERIM_DATA_DIRECTORY = DATA_DIRECTORY / "interim"

PROCESSED_DATA_DIRECTORY = DATA_DIRECTORY / "processed"

# =============================================================================
# IMDB Dataset
# =============================================================================

IMDB_DATASET_DIRECTORY = RAW_DATA_DIRECTORY / "imdb"

TRAIN_DIRECTORY = IMDB_DATASET_DIRECTORY / "train"

TEST_DIRECTORY = IMDB_DATASET_DIRECTORY / "test"

# =============================================================================
# Interim Files
# =============================================================================

TRAIN_DATA_FILE = INTERIM_DATA_DIRECTORY / "train.csv"

VALIDATION_DATA_FILE = INTERIM_DATA_DIRECTORY / "validation.csv"

TEST_DATA_FILE = INTERIM_DATA_DIRECTORY / "test.csv"

# =============================================================================
# Processed Files
# =============================================================================

TFIDF_VECTORIZER_FILE = (
    PROCESSED_DATA_DIRECTORY / "tfidf_vectorizer.joblib"
)

X_TRAIN_FILE = (
    PROCESSED_DATA_DIRECTORY / "X_train.joblib"
)

X_VALIDATION_FILE = (
    PROCESSED_DATA_DIRECTORY / "X_validation.joblib"
)

X_TEST_FILE = (
    PROCESSED_DATA_DIRECTORY / "X_test.joblib"
)

Y_TRAIN_FILE = (
    PROCESSED_DATA_DIRECTORY / "y_train.csv"
)

Y_VALIDATION_FILE = (
    PROCESSED_DATA_DIRECTORY / "y_validation.csv"
)

Y_TEST_FILE = (
    PROCESSED_DATA_DIRECTORY / "y_test.csv"
)

# =============================================================================
# Models
# =============================================================================

MODELS_DIRECTORY = PROJECT_ROOT / "models"

BEST_MODEL_FILE = (
    MODELS_DIRECTORY / "best_model.joblib"
)

BEST_MODEL_INFO_FILE = (
    MODELS_DIRECTORY / "best_model_info.json"
)

# =============================================================================
# Reports
# =============================================================================

REPORTS_DIRECTORY = PROJECT_ROOT / "reports"

EVALUATION_REPORT_DIRECTORY = (
    REPORTS_DIRECTORY / "evaluation"
)

FIGURES_DIRECTORY = (
    REPORTS_DIRECTORY / "figures"
)

EDA_FIGURES_DIRECTORY = (
    FIGURES_DIRECTORY / "eda"
)

MODEL_FIGURES_DIRECTORY = (
    FIGURES_DIRECTORY / "models"
)

MODEL_RESULTS_FILE = (
    EVALUATION_REPORT_DIRECTORY / "model_results.csv"
)

MODEL_RESULTS_JSON_FILE = (
    EVALUATION_REPORT_DIRECTORY / "model_results.json"
)

EVALUATION_REPORT_FILE = (
    EVALUATION_REPORT_DIRECTORY / "evaluation_report.txt"
)

# =============================================================================
# Visualization
# =============================================================================

DEFAULT_FIGURE_WIDTH = 10

DEFAULT_FIGURE_HEIGHT = 6

DEFAULT_FIGURE_DPI = 300

MATPLOTLIB_STYLE = "ggplot"

MAX_REVIEW_LENGTH = 800

# =============================================================================
# Logs
# =============================================================================

LOG_DIRECTORY = PROJECT_ROOT / "logs"

LOG_FILE = (
    LOG_DIRECTORY / "project.log"
)