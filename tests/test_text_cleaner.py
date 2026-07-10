"""
Tests for text cleaning utilities.
"""

from src.preprocessing.text_cleaner import clean_text


def test_clean_text_lowercase():
    """
    Text should be converted to lowercase.
    """

    text = "HELLO WORLD"

    cleaned = clean_text(text)

    assert cleaned == "hello world"


def test_clean_text_remove_html():
    """
    HTML tags should be removed.
    """

    text = "<b>Hello</b> World"

    cleaned = clean_text(text)

    assert "<b>" not in cleaned
    assert "</b>" not in cleaned


def test_clean_text_remove_url():
    """
    URLs should be removed.
    """

    text = "Visit https://example.com now"

    cleaned = clean_text(text)

    assert "http" not in cleaned


def test_clean_text_remove_punctuation():
    """
    Punctuation should be removed.
    """

    text = "Amazing!!! Movie???"

    cleaned = clean_text(text)

    assert "!" not in cleaned
    assert "?" not in cleaned


def test_clean_text_stopwords_removed():
    """
    English stopwords should be removed.
    """

    text = "this is a very good movie"

    cleaned = clean_text(text)

    words = cleaned.split()

    assert "this" not in words
    assert "is" not in words
    assert "a" not in words


def test_clean_text_lemmatization():
    """
    Words should be lemmatized.
    """

    text = "cars"

    cleaned = clean_text(text)

    assert cleaned == "car"


def test_clean_text_empty_string():
    """
    Empty string should stay empty.
    """

    assert clean_text("") == ""


def test_clean_text_only_symbols():
    """
    Symbols should be removed completely.
    """

    assert clean_text("!!!@@@###") == ""


def test_clean_text_complete_sentence():
    """
    Test a realistic movie review sentence.
    """

    text = """
    I REALLY loved this movie!!!
    It was amazing.
    """

    cleaned = clean_text(text)

    assert cleaned == "really loved movie amazing"