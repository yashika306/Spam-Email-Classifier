# 📧 Spam Email Classifier

A Machine Learning web application that classifies SMS/Email messages as **Spam** or **Ham** using **Natural Language Processing (NLP)**. This project compares the performance of **Multinomial Naive Bayes** and **Logistic Regression** models through an interactive **Streamlit** interface.

🔗 **Live Demo:** https://span-email-classifier.streamlit.app/

---

## 📖 Overview

Spam messages are one of the most common cybersecurity and communication challenges. This project uses **TF-IDF Vectorization** with two supervised machine learning algorithms to classify incoming messages and compare their predictions in real time.

Users can:

- ✉️ Enter an email or SMS
- 🤖 Compare predictions from two ML models
- 📊 View spam probability scores
- 📈 Compare model performance

---

## 🚀 Features

- 📧 Classify Email/SMS as Spam or Ham
- 🤖 Compare **Naive Bayes** and **Logistic Regression**
- 📊 Display Spam Probability
- ⚡ Real-time predictions
- 🎨 Interactive Streamlit UI
- 📈 Model comparison table
- 💾 Saved ML models using Joblib

---

## 🖥️ Live Demo

👉 https://span-email-classifier.streamlit.app/

---

## 🛠️ Tech Stack

### Programming

- Python

### Machine Learning

- Scikit-learn
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Logistic Regression

### Libraries

- Pandas
- Joblib
- Streamlit

### Concepts

- Natural Language Processing (NLP)
- Text Classification
- Feature Extraction
- Model Evaluation

---

## 📂 Project Structure

```text
Spam_Email_Classifier/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── data/
│   └── mail_data.csv
│
├── models/
│   ├── logistic_model.pkl
│   ├── naive_bayes.pkl
│   └── vectorizer.pkl
│
├── src/
    ├── preprocess.py
    ├── predict.py
    ├── evaluate.py
```

---

## ⚙️ Installation

Clone the repository.

```bash
git clone https://github.com/yashika306/Spam-Email-Classifier.git

cd Spam_Email_Classifier
```

Create a virtual environment.

```bash
python -m venv venv
```

Activate the environment.

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

Train the models.

```bash
python train_model.py
```

Launch the Streamlit application.

```bash
streamlit run app.py
```

---

## 📊 Dataset

**SMS Spam Collection Dataset**

- Source: UCI Machine Learning Repository
- Kaggle Version:
  https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

---

## 🧠 Machine Learning Workflow

```
Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Train/Test Split
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Model Training
      │
      ├──────────────┐
      ▼              ▼
Naive Bayes   Logistic Regression
      │              │
      └──────┬───────┘
             ▼
      Model Comparison
             ▼
      Streamlit Web App
```

---

## 📈 Models Used

### 1️⃣ Multinomial Naive Bayes

- Fast
- Lightweight
- Excellent for text classification

---

### 2️⃣ Logistic Regression

- Linear Classification Algorithm
- Produces probability scores
- Performs well on TF-IDF features

---

## 📋 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

---


## 🎯 Learning Outcomes

Through this project, I gained hands-on experience with:

- Data Cleaning
- NLP Preprocessing
- TF-IDF Vectorization
- Supervised Machine Learning
- Model Comparison
- Model Serialization
- Streamlit Deployment
- Git & GitHub

---

## 👩‍💻 Author

**Yashika Duthuluru**

📧 yashikaduthuluru@gmail.com

💼 LinkedIn:
https://www.linkedin.com/in/yashikaduthuluru/

🐙 GitHub:
https://github.com/yashika306

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
