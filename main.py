import numpy as np 
import tensorflow as tf
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

# load the IMDB dataset 
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with Relu activation
model = load_model('simple_rnn.h5')

# Step 2: Helper function
# Function to decode reviews
def decode_review(encoded_reviews):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_reviews])

# Function to preproces user input
def preproces_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Step 4: Streamlit app
st.title("IMDB Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment:")

## User input
user_input = st.text_area("Movie review")

if st.button('Classify'):
    preprocess_input = preproces_text(user_input)

    # Make prediction
    prediction = model.predict(preprocess_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display prediction
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction score: {prediction[0][0]}")

else :
    st.write("Please enter a movie review to predict its sentiment.")