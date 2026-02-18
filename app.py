import streamlit as st
from backend.predict import predict_resume

# Page config
st.set_page_config(page_title="AI Resume Classifier", layout="centered")

st.title("AI Resume Classification System")
st.write("Paste resume text below to predict the job category.")

# Text input
resume_text = st.text_area("Resume Text", height=300)

if st.button("Predict Category"):
    if resume_text.strip() == "":
        st.warning("Please enter resume text.")
    else:
        result = predict_resume(resume_text)
        st.success(f"Predicted Category: {result}")
