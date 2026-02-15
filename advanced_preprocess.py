#Step 4
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load dataset
df = pd.read_csv("data/resume_dataset.csv")

# Select needed columns
df = df[['Resume_str', 'Category']]
df.columns = ['Resume', 'Category']

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Cleaning + NLP function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()
    
    # Remove stopwords + Lemmatize
    cleaned_words = []
    for word in words:
        if word not in stop_words:
            lemma = lemmatizer.lemmatize(word)
            cleaned_words.append(lemma)
    
    return " ".join(cleaned_words)

# Apply preprocessing
df['Processed_Resume'] = df['Resume'].apply(preprocess_text)

# Show comparison
print("\nOriginal:\n")
print(df['Resume'].iloc[0][:300])

print("\nProcessed:\n")
print(df['Processed_Resume'].iloc[0][:300])
