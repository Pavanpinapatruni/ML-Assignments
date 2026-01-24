import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

import re

word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

# Load the pre-trained model with Relu activation
model = load_model('simple_rnn_imdb.h5')

# Helper function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_review(review, maxlen=500):
    # Simple preprocessing: lowercase and split by spaces
    # Better preprocessing: handle punctuation and lowercase
    # Remove punctuation and convert to lowercase
    review = re.sub(r'[^a-zA-Z\s]', '', review.lower())
    words = review.split()
    encoded_review = []
    for word in words:
        index = word_index.get(word, 2)  # 2 is the index for 'unknown' words
        encoded_review.append(index + 3)  # Offset by 3 for special tokens
    padded_review = sequence.pad_sequences([encoded_review], maxlen=maxlen)
    return padded_review

### Step3: Predict function
def predict_review(review):
    
    preprocessed_review = preprocess_review(review)
    prediction = model.predict(preprocessed_review)
    sentiment = 'Positive' if prediction[0][0] >= 0.5 else 'Negative'

    return sentiment, prediction[0][0]

## Streamlit app
import streamlit as st
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (Positive/Negative)")

# User Input
user_input = st.text_area("Movie Review", "Type your review here...")

if st.button("Predict Sentiment"):
    sentiment, confidence = predict_review(user_input)
    st.write(f"Predicted Sentiment: **{sentiment}**")
    st.write(f"Confidence Score: **{confidence:.4f}**")
else:
    st.write("Please enter a review and click 'Predict Sentiment'")