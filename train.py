import pandas as pd
import numpy as np
import re
import nltk
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# --- LOAD DATA ---
print("--- Loading Dataset ---")

df = pd.read_csv("WELFake_Dataset.csv")

print("Loaded rows:", len(df))

# Combine title + text
df['content'] = df['title'] + " " + df['text']

# Drop missing
df = df.dropna()

# --- CLEAN TEXT ---
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(w) for w in text if w not in stop_words]
    return ' '.join(text)

print("--- Cleaning ---")
df['content'] = df['content'].apply(clean_text)

# --- SPLIT DATA ---
X = df['content']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- VECTORIZER (IMPROVED) ---
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words='english'
)

X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# --- MODEL (IMPROVED) ---
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

print("--- Training Model ---")
model.fit(X_train_vect, y_train)

# --- EVALUATE ---
y_pred = model.predict(X_test_vect)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", round(accuracy * 100, 2), "%")

# --- SAVE FILES ---
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("tfidfvect.pkl", "wb"))

print("✅ model.pkl and tfidfvect.pkl saved")
print("Label: 1 = FAKE, 0 = REAL")