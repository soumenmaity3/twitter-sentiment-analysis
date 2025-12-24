import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import load_model

# Download stopwords
nltk.download('stopwords')

# Initialize stemmer and stopwords once
stem = PorterStemmer()
stop_words = set(stopwords.words('english'))

def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower().split()
    content = [stem.stem(word) for word in content if word not in stop_words]
    return ' '.join(content)

# Load dataset
df = pd.read_csv('df_updated.csv')
df = df.dropna(subset=['stemmed_content'])

# Train-test split (same as training)
X = df['stemmed_content']
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit TF-IDF only on training data
tfidf = TfidfVectorizer()
tfidf.fit(X_train)

# Load trained ANN model
model_ann = load_model('twitter_ann_mode.keras')

# -------------------- Streamlit UI --------------------

st.title("Twitter Sentiment Analysis (ANN)")
st.markdown(
    "Enter a tweet or text to analyze its sentiment using a trained Artificial Neural Network."
)

user_input = st.text_area("Enter your text here:", height=120)

if st.button("Predict Sentiment"):
    if user_input.strip():

        # Preprocess input
        stemmed_text_ann = stemming(user_input)
        X_new_ann = tfidf.transform([stemmed_text_ann])

        # Predict
        prediction_ann = model_ann.predict(X_new_ann)

        if prediction_ann[0] > 0.5:
            sentiment = "Positive"
            confidence = prediction_ann[0].item() * 100
            st.success(f"Sentiment: {sentiment}")
            st.info(f"Confidence: {confidence:.2f}%")
            st.markdown("😊 **Positive Sentiment Detected!**")
        else:
            sentiment = "Negative"
            confidence = (1 - prediction_ann[0].item()) * 100
            st.error(f"Sentiment: {sentiment}")
            st.info(f"Confidence: {confidence:.2f}%")
            st.markdown("😞 **Negative Sentiment Detected!**")

    else:
        st.warning("Please enter some text to analyze.")

st.markdown("---")
st.markdown("Built with Streamlit and TensorFlow • ANN-based Twitter Sentiment Analysis")
