import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

def extract_keywords(text):
    text = clean_text(text)
    words = text.split()
    keywords = [w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2]
    return set(keywords)

def analyze_skill_gap(resume_text, job_description):
    resume_keywords = extract_keywords(resume_text)
    job_keywords = extract_keywords(job_description)

    matched_skills = resume_keywords.intersection(job_keywords)
    missing_skills = job_keywords - resume_keywords

    return list(matched_skills), list(missing_skills)