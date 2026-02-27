import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Title
st.title("📰 Fake News Detector")

# Load dataset
df = pd.read_csv(r"C:\Users\MCA\AppData\Local\Programs\Python\Python312\fake news\news.csv.csv")

# Train model
X = df["text"]
y = df["label"]

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

# User input
news_text = st.text_area("Enter news article:")

# Predict button
if st.button("Predict"):
    if news_text:
        vec = vectorizer.transform([news_text])
        prediction = model.predict(vec)[0]
        st.subheader(f"Prediction: {prediction}")
    else:
        st.warning("Please enter some text")

