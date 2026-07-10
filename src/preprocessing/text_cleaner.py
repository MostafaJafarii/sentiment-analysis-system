"""
Text cleaning utilities.
"""

import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# =============================================================================
# Regular Expression Patterns
# =============================================================================

HTML_PATTERN = r"<.*?>"

URL_PATTERN = r"http\S+"

NON_ALPHABET_PATTERN = r"[^a-zA-Z\s]"


# =============================================================================
# NLTK Resources
# =============================================================================

STOP_WORDS = set(
    stopwords.words("english")
)

LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Clean raw review text.

    Parameters
    ----------
    text : str
        Raw review text.

    Returns
    -------
    str
        Cleaned review text.
    """

    text = text.lower()

    text = re.sub(
        HTML_PATTERN,
        " ",
        text
    )

    text = re.sub(
        URL_PATTERN,
        " ",
        text
    )

    text = re.sub(
        NON_ALPHABET_PATTERN,
        " ",
        text
    )

    words = text.split()

    words = [

        LEMMATIZER.lemmatize(word)

        for word in words

        if word not in STOP_WORDS

    ]

    return " ".join(words)