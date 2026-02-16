#step 7
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

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

# ==============================
# 3️⃣ Define Features & Labels
# ==============================

X = df['Resume']
y = df['Category']

# ==============================
# 4️⃣ TF-IDF with Improvements
# ==============================

tfidf = TfidfVectorizer(
    max_features=8000,
    stop_words='english',
    ngram_range=(1, 2)  # unigrams + bigrams
)

X_tfidf = tfidf.fit_transform(X)

print("TF-IDF Shape:", X_tfidf.shape)

# ==============================
# 5️⃣ Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ==============================
# 6️⃣ Logistic Regression (Improved)
# ==============================

model = LogisticRegression(
    max_iter=2000,
    class_weight='balanced'
)

model.fit(X_train, y_train)

# ==============================
# 7️⃣ Prediction
# ==============================

y_pred = model.predict(X_test)

# ==============================
# 8️⃣ Evaluation
# ==============================

accuracy = accuracy_score(y_test, y_pred)

print("\nImproved Model Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
