import joblib


# ----------------------------
# Load Models
# ----------------------------

nb_model = joblib.load("models/naive_bayes.pkl")
log_model = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


# ----------------------------
# Predict using Naive Bayes
# ----------------------------

def predict_naive_bayes(message):

    vector = vectorizer.transform([message])

    prediction = nb_model.predict(vector)[0]

    probability = nb_model.predict_proba(vector)[0][1]

    return prediction, probability


# ----------------------------
# Predict using Logistic Regression
# ----------------------------

def predict_logistic(message):

    vector = vectorizer.transform([message])

    prediction = log_model.predict(vector)[0]

    probability = log_model.predict_proba(vector)[0][1]

    return prediction, probability


# ----------------------------
# Compare Both Models
# ----------------------------

def compare_models(message):

    nb_prediction, nb_probability = predict_naive_bayes(message)

    log_prediction, log_probability = predict_logistic(message)

    return {
        "Naive Bayes": {
            "Prediction": "SPAM" if nb_prediction else "HAM",
            "Probability": nb_probability
        },

        "Logistic Regression": {
            "Prediction": "SPAM" if log_prediction else "HAM",
            "Probability": log_probability
        }
    }