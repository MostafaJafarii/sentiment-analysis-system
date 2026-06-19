"""
Download required NLTK resources.
"""

import nltk


def download_resources():

    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")

    print("NLTK resources downloaded successfully.")


if __name__ == "__main__":
    download_resources()