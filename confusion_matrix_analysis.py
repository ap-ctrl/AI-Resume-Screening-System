#Step 9
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report

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
# 3️⃣ Features & Labels
# ==============================

X = df['Resume']
y = df['Category']

# ==============================
# 4️⃣ TF-IDF
# ==============================

tfidf = TfidfVectorizer(
    max_features=8000,
    stop_words='english',
    ngram_range=(1,2)
)

X_tfidf = tfidf.fit_transform(X)

# ==============================
# 5️⃣ Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42
)

# ==============================
# 6️⃣ Train LinearSVC
# ==============================

model = LinearSVC(class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ==============================
# 7️⃣ Classification Report
# ==============================

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==============================
# 8️⃣ Confusion Matrix (Raw)
# ==============================

cm = confusion_matrix(y_test, y_pred)
labels = model.classes_

print("\nConfusion Matrix (Raw Numbers):\n")
print(cm)

# ==============================
# 9️⃣ Confusion Matrix Heatmap
# ==============================

plt.figure(figsize=(14, 10))
sns.heatmap(cm,
            xticklabels=labels,
            yticklabels=labels,
            cmap="Blues",
            annot=False)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - LinearSVC")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

