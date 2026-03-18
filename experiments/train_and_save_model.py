import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# ==============================
# 1️⃣ Load Dataset
# ==============================

df = pd.read_csv("data/resume_dataset.csv")
df = df[['Resume_str', 'Category']]
df.columns = ['Resume', 'Category']

# ==============================
# 2️⃣ Basic Cleaning
# ==============================

def basic_clean(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

df['Resume'] = df['Resume'].apply(basic_clean)

X = df['Resume']
y = df['Category']

# ==============================
# 3️⃣ TF-IDF Vectorization
# ==============================

tfidf = TfidfVectorizer(
    max_features=8000,
    stop_words='english',
    ngram_range=(1,2)
)

X_tfidf = tfidf.fit_transform(X)

# ==============================
# 4️⃣ Train Final Model
# ==============================

model = LinearSVC(C=1, class_weight='balanced')
model.fit(X_tfidf, y)

# ==============================
# 5️⃣ Save Model & Vectorizer
# ==============================

joblib.dump(model, "resume_classifier.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully!")
