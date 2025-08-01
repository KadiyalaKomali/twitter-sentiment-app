import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load vectorizer and model
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove links
    text = re.sub(r'\@w+|\#','', text)  # Remove @ and #
    text = re.sub(r"[^A-Za-z\s]", "", text)  # Remove special characters
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Streamlit UI
st.set_page_config(page_title="Tweet Sentiment App", layout="centered")

st.title("🧠 Twitter Hate Speech Detector")
st.markdown("Enter a tweet below to check if it contains hate speech (racist/sexist).")

tweet = st.text_area("✍️ Type or paste a tweet here:", height=100)

if st.button("Predict Sentiment"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        processed = preprocess_text(tweet)
        vect_text = vectorizer.transform([processed])
        prediction = model.predict(vect_text)[0]

        if prediction == 1:
            st.error("⚠️ Hate Speech Detected")
        else:
            st.success("✅ Neutral Tweet")

