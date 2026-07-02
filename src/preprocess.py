import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(filepath):
    """
    Load and clean dataset.
    """

    df = pd.read_csv(filepath, encoding="latin-1")

    df = df.rename(columns={
        "v1": "Category",
        "v2": "Message"
    })

    df = df.drop_duplicates(subset="Message")

    df["Label"] = (df["Category"] == "spam").astype(int)

    return df


def preprocess_data(df):
    """
    Split data and convert text into TF-IDF vectors.
    """

    X = df["Message"]
    y = df["Label"]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=3000
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        vectorizer
    )