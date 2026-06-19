"""
Feature extraction using TF-IDF.
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf():

    return TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2)
    )