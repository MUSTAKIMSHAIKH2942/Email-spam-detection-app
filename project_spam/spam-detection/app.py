# import streamlit as st
# import joblib
# from scripts.preprocess import preprocess_text

# # Load the trained model and vectorizer
# model = joblib.load("spam_classifier.pkl")
# vectorizer = joblib.load("vectorizer.pkl")

# def predict_spam(text):
#     """
#     Predict if the input text is spam or ham.
#     """
#     # Preprocess and vectorize the input text
#     preprocessed_text = preprocess_text(text)
#     vect_text = vectorizer.transform([preprocessed_text])

#     # Predict using the trained model
#     prediction = model.predict(vect_text)[0]
#     return "Spam" if prediction == 1 else "Ham"

# # Streamlit App
# st.title("Email/SMS Spam Detection App")
# st.write("Enter a message or email content below to check if it's spam or not.")

# # Text input
# user_input = st.text_area("Enter your message:")

# # Predict button
# if st.button("Predict"):
#     if user_input.strip() == "":
#         st.warning("Please enter some text to predict.")
#     else:
#         result = predict_spam(user_input)
#         if result == "Spam":
#             st.error("The message is classified as SPAM.")
#         else:
#             st.success("The message is classified as HAM (Not Spam).")
import os
import streamlit as st
import joblib
from scripts.preprocess import preprocess_text

# Define paths to the model and vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "spam_classifier.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "vectorizer.pkl")

# Load the trained model and vectorizer
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except FileNotFoundError as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()

def predict_spam(text):
    """
    Predict if the input text is spam or ham.
    """
    # Preprocess and vectorize the input text
    preprocessed_text = preprocess_text(text)
    vect_text = vectorizer.transform([preprocessed_text])

    # Predict using the trained model
    prediction = model.predict(vect_text)[0]
    return "Spam" if prediction == 1 else "Ham"

# Streamlit App
st.title("Email/SMS Spam Detection App")
st.write("Enter a message or email content below to check if it's spam or not.")

# Text input
user_input = st.text_area("Enter your message:")

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text to predict.")
    else:
        result = predict_spam(user_input)
        if result == "Spam":
            st.error("The message is classified as SPAM.")
        else:
            st.success("The message is classified as HAM (Not Spam).")
