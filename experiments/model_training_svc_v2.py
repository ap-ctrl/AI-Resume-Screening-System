#Step 8
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Download required resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# 1️⃣ Load Dataset
df = pd.read_csv("data/resume_dataset.csv")
df = df[['Resume_str', 'Category']]
df.columns = ['Resume', 'Category']

# 2️⃣ Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# 3️⃣ Advanced Preprocessing Function
def advanced_preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    
    cleaned_words = []
    for word in words:
        if word not in stop_words:
            lemma = lemmatizer.lemmatize(word)
            cleaned_words.append(lemma)
    
    return " ".join(cleaned_words)

df['Resume'] = df['Resume'].apply(advanced_preprocess)

# 4️⃣ Define Features & Labels
X = df['Resume']
y = df['Category']

# 5️⃣ TF-IDF
tfidf = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1,2)
)

X_tfidf = tfidf.fit_transform(X)

# 6️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42
)

# 7️⃣ LinearSVC Model
model = LinearSVC(class_weight='balanced')
model.fit(X_train, y_train)

# 8️⃣ Prediction
y_pred = model.predict(X_test)

# 9️⃣ Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("LinearSVC with Advanced Preprocessing Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
