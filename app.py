import streamlit as st
import pandas as pd
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

TEXT_COLUMN_CANDIDATES = (
    "text",
    "tweet",
    "tweet_text",
    "full_text",
    "content",
    "message",
)


def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove links
    text = re.sub(r'\@w+|\#','', text)  # Remove @ and #
    text = re.sub(r"[^A-Za-z\s]", "", text)  # Remove special characters
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text


def find_text_column(dataframe):
    normalized = {column.lower().strip(): column for column in dataframe.columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    return None


def predict_label(text):
    processed = preprocess_text(str(text))
    vect_text = vectorizer.transform([processed])
    prediction = model.predict(vect_text)[0]
    return "Hate Speech" if prediction == 1 else "Neutral"

# Streamlit UI
st.set_page_config(page_title="Tweet Sentiment App", layout="centered")

st.title("🧠 Twitter Hate Speech Detector")
st.markdown("Enter a tweet below to check if it contains hate speech (racist/sexist).")

tweet = st.text_area("✍️ Type or paste a tweet here:", height=100)

if st.button("Predict Sentiment"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        prediction = predict_label(tweet)

        if prediction == "Hate Speech":
            st.error("⚠️ Hate Speech Detected")
        else:
            st.success("✅ Neutral Tweet")

st.subheader("Batch CSV Prediction")
uploaded_file = st.file_uploader(
    "Upload a tweet CSV",
    type=["csv"],
    help="Supports text, tweet, tweet_text, full_text, content, or message columns.",
)

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    text_column = find_text_column(batch_df)

    if text_column is None:
        st.error("No tweet text column found. Add text, tweet, tweet_text, full_text, content, or message.")
    else:
        result_df = batch_df.dropna(subset=[text_column]).copy()
        if result_df.empty:
            st.warning("The selected text column has no rows to classify.")
        else:
            result_df["predicted_label"] = [predict_label(value) for value in result_df[text_column]]
            st.dataframe(result_df, use_container_width=True)
            st.download_button(
                "Download predictions",
                result_df.to_csv(index=False).encode("utf-8"),
                file_name="tweet_hate_speech_predictions.csv",
                mime="text/csv",
            )

