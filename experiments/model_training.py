#step 6
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("data/resume_dataset.csv")

# Keep required columns
df = df[['Resume_str', 'Category']]
df.columns = ['Resume', 'Category']

# Basic cleaning
def basic_clean(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

df['Resume'] = df['Resume'].apply(basic_clean)

# Define features and labels
X = df['Resume']
y = df['Category']

# Convert text to TF-IDF
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = tfidf.fit_transform(X)

# 🔥 Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# 🔥 Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 🔥 Make Predictions
y_pred = model.predict(X_test)

# 🔥 Evaluate Model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
