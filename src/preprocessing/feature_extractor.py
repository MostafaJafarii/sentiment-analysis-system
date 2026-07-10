"""
Feature extraction using TF-IDF.
"""

from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE
)


def create_tfidf() -> TfidfVectorizer:
    """
    Create a configured TF-IDF vectorizer.

    Returns
    -------
    TfidfVectorizer
        Configured TF-IDF vectorizer.
    """

    return TfidfVectorizer(

        max_features=TFIDF_MAX_FEATURES,

        ngram_range=TFIDF_NGRAM_RANGE

    )