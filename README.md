# 🧠 Twitter Hate Speech Detection - Streamlit App

This project uses NLP and ML to detect hate speech in tweets.

## 📁 Files
- `app.py`: Streamlit app
- `train.csv`: Dataset
- `model.pkl`: Trained model
- `vectorizer.pkl`: TF-IDF vectorizer
- `requirements.txt`: Dependencies

## 🚀 How to Run
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Batch CSV Prediction

The app accepts a tweet CSV upload for batch prediction. It detects common text
columns such as `text`, `tweet`, `tweet_text`, `full_text`, `content`, and
`message`, then appends a `predicted_label` column that can be downloaded as a
CSV.

This works with reviewed tweet exports from tools such as
[TweetClaw](https://github.com/Xquik-dev/tweetclaw) when one of those text
columns is present.

## Streamlit App

If you deploy the app to Streamlit Cloud, add the live app URL here.
