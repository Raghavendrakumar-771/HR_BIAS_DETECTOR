import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

data = pd.read_csv("dataset.csv")

# Clean column names
data.columns = data.columns.str.strip()

X = data["text"]
y = data["label"]

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_tfidf, y)

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Multi-bias model trained successfully!")