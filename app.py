# Import necessary libraries
import streamlit as st
import transformers
import torch
from transformers import pipeline

# Set up the Streamlit app
st.title("Emotion Detection with Transformers")

# Create a text input widget
user_input = st.text_area("Enter your text:")


# Define a function for sentiment analysis using transformers
@st.cache_data
def load_model():
    return pipeline("sentiment-analysis")


# Load the sentiment analysis model
sentiment_analyzer = load_model()

# Create a button to analyze the emotion
if st.button("Analyze Emotion"):
    if user_input:
        # Perform sentiment analysis on user input
        result = sentiment_analyzer(user_input)

        # Display the result
        emotion = result[0]['label']
        st.write(f"Emotion: {emotion}")
    else:
        st.warning("Please enter some text to analyze.")
