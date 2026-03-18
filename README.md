# 🚀 AI Resume Screening & Job Matching System

🔗 **Live App:** [Click Here to Try](https://ai-resume-screeningsystem.streamlit.app/)

---

## 📌 Overview

This project is an **AI-powered Resume Screening System** that:

- 📄 Analyzes resume text
- 🎯 Matches resumes with relevant job roles
- 📊 Calculates similarity score
- 🧠 Predicts resume domain
- 🛠️ Identifies skill gaps

---

## 🧠 Features

- 🔍 **Job Matching** using TF-IDF + Cosine Similarity
- 📊 **Resume Classification** using LinearSVC
- 🧩 **Skill Gap Analysis** (matched vs missing skills)
- ⚠️ **Confidence Indicator** for weak matches
- 🌐 **Deployed Web App** using Streamlit

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Joblib

---

## 📂 Project Structure

```
resume-ai-system/
│
├── backend/
│   ├── predict.py
│   ├── job_matcher.py
│   ├── skill_gap.py
│
├── frontend/
│   └── app.py
│
├── data/
│   ├── jobs.csv
│   └── resume_dataset.csv
│
├── models/
│   ├── resume_classifier.pkl
│   └── tfidf_vectorizer.pkl
│
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

1. Resume text is converted into vectors using TF-IDF
2. Job descriptions are vectorized using the same model
3. Cosine similarity finds best job matches
4. SVM model predicts resume domain
5. Skill gap analysis compares resume vs job skills

---

## 🚀 Run Locally

```bash
git clone https://github.com/your-username/resume-ai-system.git
cd resume-ai-system
pip install -r requirements.txt
streamlit run frontend/app.py
```

---

## 🌟 Future Improvements

- 📄 Resume PDF upload support
- 🤖 BERT-based semantic matching
- 📊 Better skill extraction
- 🧠 Advanced AI recommendations

---

## 👩‍💻 Author

Developed by Ankita

---
