import re
import joblib
import os

# ==============================
# Get correct path dynamically
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "resume_classifier.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

# ==============================
# Load Model & Vectorizer
# ==============================

model = joblib.load(MODEL_PATH)
tfidf = joblib.load(VECTORIZER_PATH)

# ==============================
# Cleaning Function
# ==============================

def basic_clean(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

# ==============================
# Prediction Function
# ==============================

def predict_resume(resume_text):
    cleaned_text = basic_clean(resume_text)
    vector = tfidf.transform([cleaned_text])
    prediction = model.predict(vector)
    return prediction[0]


# ==============================
# Manual Test
# ==============================

if __name__ == "__main__":
    sample = """
    Experienced software engineer skilled in Python,
    machine learning, data analysis and deep learning.
    """

    print("Predicted Category:", predict_resume(sample))
