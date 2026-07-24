"""
Download required NLTK resources for the Sentiment Analysis project.
"""

import nltk


def download_resources():
    """
    Download all required NLTK resources.
    """

    resources = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "omw-1.4",
    ]

    print("=" * 60)
    print("Downloading NLTK resources...")
    print("=" * 60)

    for resource in resources:
        print(f"Downloading: {resource}")
        nltk.download(resource, quiet=False)

    print("\nAll NLTK resources have been downloaded successfully.")
    print("=" * 60)


if __name__ == "__main__":
    download_resources()