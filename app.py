import streamlit as st
from src.predict import compare_models

# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="wide"
)

# -----------------------------------
# Header
# -----------------------------------

st.title("📧 Spam Email Classifier")
st.markdown(
"""
Compare predictions from two Machine Learning models:

- 🟢 **Multinomial Naive Bayes**
- 🔵 **Logistic Regression**
"""
)

st.divider()

# -----------------------------------
# Input
# -----------------------------------

message = st.text_area(
    "Enter your Email or SMS",
    placeholder="Congratulations! You've won ₹10,000. Click here to claim.",
    height=180
)

# -----------------------------------
# Predict Button
# -----------------------------------

if st.button("Predict", use_container_width=True):

    if message.strip() == "":
        st.warning("Please enter a message.")
        st.stop()

    result = compare_models(message)

    st.divider()

    col1, col2 = st.columns(2)

    # ============================================
    # Naive Bayes
    # ============================================

    with col1:

        st.subheader("🟢 Naive Bayes")

        prediction = result["Naive Bayes"]["Prediction"]
        probability = result["Naive Bayes"]["Probability"]

        if prediction == "SPAM":
            st.error("Prediction : SPAM 🚨")
        else:
            st.success("Prediction : HAM ✅")

        st.metric(
            "Spam Probability",
            f"{probability*100:.2f}%"
        )

        st.progress(float(probability))

    # ============================================
    # Logistic Regression
    # ============================================

    with col2:

        st.subheader("🔵 Logistic Regression")

        prediction = result["Logistic Regression"]["Prediction"]
        probability = result["Logistic Regression"]["Probability"]

        if prediction == "SPAM":
            st.error("Prediction : SPAM 🚨")
        else:
            st.success("Prediction : HAM ✅")

        st.metric(
            "Spam Probability",
            f"{probability*100:.2f}%"
        )

        st.progress(float(probability))

    # ============================================
    # Comparison Table
    # ============================================

    st.divider()

    st.subheader("📊 Model Comparison")

    st.table({
        "Model": [
            "Naive Bayes",
            "Logistic Regression"
        ],

        "Prediction": [
            result["Naive Bayes"]["Prediction"],
            result["Logistic Regression"]["Prediction"]
        ],

        "Spam Probability": [
            f"{result['Naive Bayes']['Probability']*100:.2f}%",
            f"{result['Logistic Regression']['Probability']*100:.2f}%"
        ]
    })

st.divider()

st.caption(
    "Built with ❤️ using Streamlit, Scikit-learn, Naive Bayes & Logistic Regression"
)