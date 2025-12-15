import streamlit as st
import pickle
from preprocess import clean_text
import PyPDF2

model = pickle.load(open("model/resume_classifier.pkl", "rb"))
vectorizer = pickle.load(open("model/tfidf.pkl", "rb"))

st.title("Intelligent Resume Screening System")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

if uploaded_file:
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)

    st.success(f"Predicted Job Role: {prediction[0]}")
