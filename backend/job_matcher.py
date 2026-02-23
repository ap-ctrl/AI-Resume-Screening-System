import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load vectorizer
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
JOBS_PATH = os.path.join(BASE_DIR, "data", "jobs.csv")

tfidf = joblib.load(VECTORIZER_PATH)

# Load job data
jobs_df = pd.read_csv(JOBS_PATH)
jobs_df = pd.read_csv(JOBS_PATH)

# 🔥 Fix: Remove rows where Job_Description is missing
jobs_df = jobs_df.dropna(subset=["Job_Description"])

def match_jobs(resume_text, top_n=3):
    # Transform resume
    resume_vector = tfidf.transform([resume_text])

    # Transform job descriptions
    job_vectors = tfidf.transform(jobs_df["Job_Description"])

    # Compute similarity
    similarities = cosine_similarity(resume_vector, job_vectors)

    # Get top matches
    jobs_df["Similarity"] = similarities[0]

    top_jobs = jobs_df.sort_values(by="Similarity", ascending=False).head(top_n)

    #return top_jobs[["Job_Title", "Category", "Similarity"]]
    return top_jobs