"""
Prediction utilities for sentiment analysis API.
"""

from time import perf_counter

import joblib

from src.config import (
    TFIDF_VECTORIZER_FILE
)

from src.preprocessing.text_cleaner import clean_text

from src.models.model_utils import (
    load_model,
    get_model_display_name
)

class SentimentPredictor:
    """
    Predict sentiment using a selected model.
    """

    def __init__(self) -> None:

        self.vectorizer = joblib.load(
            TFIDF_VECTORIZER_FILE
        )

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _prepare_features(
        self,
        review: str
    ):

        cleaned_review = clean_text(
            review
        )

        return self.vectorizer.transform(
            [cleaned_review]
        )

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------

    def predict(
        self,
        review: str,
        model_name: str = "best_model"
    ) -> dict:
        """
        Predict sentiment using the selected model.
        """

        model = load_model(
            model_name
        )

        features = self._prepare_features(
            review
        )

        start = perf_counter()

        label = int(
            model.predict(features)[0]
        )

        elapsed = perf_counter() - start

        sentiment = (
            "positive"
            if label == 1
            else "negative"
        )

        confidence = self._calculate_confidence(
            model,
            features
        )

        return {

            "label": label,

            "sentiment": sentiment,

            "confidence": round(
                confidence * 100,
                2
            ),

            "model": get_model_display_name(
                model_name
            ),

            "prediction_time": round(
                elapsed,
                4
            )

        }

    # -------------------------------------------------------------------------
    # Confidence
    # -------------------------------------------------------------------------

    def _calculate_confidence(
        self,
        model,
        features
    ) -> float:
        """
        Calculate prediction confidence.
        """

        # Models that support probabilities
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                features
            )[0]

            return float(
                probabilities.max()
            )

        # Linear SVM
        if hasattr(model, "decision_function"):

            score = float(
                model.decision_function(
                    features
                )[0]
            )

            # Convert score to pseudo probability
            return 1 / (1 + pow(2.718281828, -abs(score)))

        # Fallback
        return 1.0