import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load dataset
df = pd.read_csv("data/resume_dataset.csv")

# Select needed columns
df = df[['Resume_str', 'Category']]
df.columns = ['Resume', 'Category']

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.strip()
    return text

# Apply cleaning
df['Cleaned_Resume'] = df['Resume'].apply(clean_text)

print("Original Resume Sample:\n")
print(df['Resume'].iloc[0][:300])

print("\nCleaned Resume Sample:\n")
print(df['Cleaned_Resume'].iloc[0][:300])
