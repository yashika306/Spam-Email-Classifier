import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
print("Script started...")
# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("data/mail_data.csv", encoding="latin-1")

# Rename columns
df = df.rename(columns={
    "v1": "Category",
    "v2": "Message"
})

# Remove duplicate messages
df = df.drop_duplicates(subset="Message")

# Create labels
df["Label"] = (df["Category"] == "spam").astype(int)

# ==========================
# Features
# ==========================

X = df["Message"]
y = df["Label"]

# ==========================
# Train Test Split
# ==========================

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# TF-IDF
# ==========================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=3000
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# ==========================
# Naive Bayes
# ==========================

nb_model = MultinomialNB()

nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)

print("\n========== Naive Bayes ==========")

print("Accuracy:",
      accuracy_score(y_test, nb_pred))

print(confusion_matrix(y_test, nb_pred))

print(classification_report(
    y_test,
    nb_pred,
    target_names=["Ham", "Spam"]
))

# ==========================
# Logistic Regression
# ==========================

log_model = LogisticRegression(
    random_state=42
)

log_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test)

print("\n========== Logistic Regression ==========")

print("Accuracy:",
      accuracy_score(y_test, log_pred))

print(confusion_matrix(y_test, log_pred))

print(classification_report(
    y_test,
    log_pred,
    target_names=["Ham", "Spam"]
))

# ==========================
# Save Models
# ==========================

joblib.dump(nb_model, "models/naive_bayes.pkl")
joblib.dump(log_model, "models/logistic_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\nModels saved successfully!")