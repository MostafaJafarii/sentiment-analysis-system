"""
Unit tests for project configuration.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Add project root to Python path
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import src.config as config


# ---------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------

def test_project_root_exists():
    assert config.PROJECT_ROOT.exists()
    assert config.PROJECT_ROOT.is_dir()


# ---------------------------------------------------------------------
# Random state
# ---------------------------------------------------------------------

def test_random_state():
    assert isinstance(config.RANDOM_STATE, int)
    assert config.RANDOM_STATE == 42


def test_validation_size():
    assert 0 < config.VALIDATION_SIZE < 1


# ---------------------------------------------------------------------
# Machine Learning
# ---------------------------------------------------------------------

def test_logistic_regression_iterations():
    assert config.LOGISTIC_REGRESSION_MAX_ITER >= 100


def test_random_forest_estimators():
    assert config.RANDOM_FOREST_ESTIMATORS > 0


def test_random_forest_jobs():
    assert config.RANDOM_FOREST_N_JOBS == -1


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

def test_available_models():
    assert isinstance(config.AVAILABLE_MODELS, dict)
    assert len(config.AVAILABLE_MODELS) == 4

    assert "logistic_regression" in config.AVAILABLE_MODELS
    assert "naive_bayes" in config.AVAILABLE_MODELS
    assert "svm" in config.AVAILABLE_MODELS
    assert "random_forest" in config.AVAILABLE_MODELS


def test_default_model():
    assert config.DEFAULT_MODEL == "best_model"


# ---------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------

def test_tfidf_settings():
    assert config.TFIDF_MAX_FEATURES > 0
    assert config.TFIDF_NGRAM_RANGE == (1, 2)


# ---------------------------------------------------------------------
# API
# ---------------------------------------------------------------------

def test_api_settings():
    assert config.API_HOST == "0.0.0.0"
    assert config.API_PORT == 5000
    assert isinstance(config.API_DEBUG, bool)


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

def test_prediction_settings():
    assert config.ENABLE_CONFIDENCE_SCORE is True
    assert config.ENABLE_PREDICTION_TIME is True


# ---------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------

def test_data_directories():
    assert config.DATA_DIRECTORY.exists()
    assert config.RAW_DATA_DIRECTORY.exists()
    assert config.INTERIM_DATA_DIRECTORY.exists()
    assert config.PROCESSED_DATA_DIRECTORY.exists()


def test_model_directory():
    assert config.MODELS_DIRECTORY.exists()


def test_report_directories():
    assert config.REPORTS_DIRECTORY.exists()
    assert config.EVALUATION_REPORT_DIRECTORY.exists()
    assert config.FIGURES_DIRECTORY.exists()
    assert config.EDA_FIGURES_DIRECTORY.exists()
    assert config.MODEL_FIGURES_DIRECTORY.exists()


def test_log_directory():
    assert config.LOG_DIRECTORY.exists()


# ---------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------

def test_processed_files():
    assert config.TFIDF_VECTORIZER_FILE.suffix == ".joblib"
    assert config.X_TRAIN_FILE.suffix == ".joblib"
    assert config.X_VALIDATION_FILE.suffix == ".joblib"
    assert config.X_TEST_FILE.suffix == ".joblib"

    assert config.Y_TRAIN_FILE.suffix == ".csv"
    assert config.Y_VALIDATION_FILE.suffix == ".csv"
    assert config.Y_TEST_FILE.suffix == ".csv"


def test_model_files():
    assert config.BEST_MODEL_FILE.suffix == ".joblib"
    assert config.BEST_MODEL_INFO_FILE.suffix == ".json"


def test_report_files():
    assert config.MODEL_RESULTS_FILE.suffix == ".csv"
    assert config.MODEL_RESULTS_JSON_FILE.suffix == ".json"
    assert config.EVALUATION_REPORT_FILE.suffix == ".txt"


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------

def test_visualization_settings():
    assert config.DEFAULT_FIGURE_WIDTH > 0
    assert config.DEFAULT_FIGURE_HEIGHT > 0
    assert config.DEFAULT_FIGURE_DPI > 0

    assert isinstance(config.MATPLOTLIB_STYLE, str)
    assert config.MAX_REVIEW_LENGTH > 0


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------

def test_ui_settings():
    assert isinstance(config.UI_TITLE, str)
    assert isinstance(config.UI_SUBTITLE, str)
    assert config.UI_MAX_REVIEW_LENGTH > 0


# ---------------------------------------------------------------------
# Log file
# ---------------------------------------------------------------------

def test_log_file():
    assert config.LOG_FILE.suffix == ".log"