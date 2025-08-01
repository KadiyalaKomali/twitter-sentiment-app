import pandas as pd
import re
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

def preprocess(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = text.lower()
    return " ".join([word for word in text.split() if word not in stop_words])

df = pd.read_csv("train.csv")[["label", "tweet"]]
df["clean"] = df["tweet"].apply(preprocess)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean"])
y = df["label"]

model = LogisticRegression()
model.fit(X, y)

# Save files
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved!")

