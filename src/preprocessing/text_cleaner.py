"""
Text cleaning utilities.
"""

import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


STOP_WORDS = set(stopwords.words("english"))

LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Clean raw review text.
    """

    text = text.lower()

    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    text = re.sub(
        r"http\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
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