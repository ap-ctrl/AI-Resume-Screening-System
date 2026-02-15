#Step 5
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv("data/resume_dataset.csv")

# Keep required columns
df = df[['Resume_str', 'Category']]
df.columns = ['Resume', 'Category']

# For now, simple cleaning (we will integrate advanced later)
#df['Resume'] = df['Resume'].str.lower()
import re

def basic_clean(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

df['Resume'] = df['Resume'].apply(basic_clean)


# Define X (features) and y (labels)
X = df['Resume']
y = df['Category']

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    max_features=5000,   # limit vocabulary size
    stop_words='english' # remove common stopwords automatically
)

# Convert text to numerical features
X_tfidf = tfidf.fit_transform(X)

# Print shape of resulting matrix
print("TF-IDF Matrix Shape:", X_tfidf.shape)

# Print first 10 feature names
#print("\nSample Feature Names:")
#print(tfidf.get_feature_names_out()[:10])
#import random

#print("\nRandom Feature Names:")
#features = tfidf.get_feature_names_out()
#print(random.sample(list(features), 20))

import numpy as np

# Get average TF-IDF score for each word
mean_tfidf = np.asarray(X_tfidf.mean(axis=0)).flatten()

# Get feature names
features = tfidf.get_feature_names_out()

# Get top 20 important words
top_indices = mean_tfidf.argsort()[-20:]
top_words = [features[i] for i in top_indices]

print("\nTop Important Words:")
print(top_words)

