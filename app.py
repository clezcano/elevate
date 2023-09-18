import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Set up Streamlit
st.title("Emotion Detection with Transformers")

# Text input
user_input = st.text_area("Enter your text:")

# Load the model and tokenizer using st.cache_data
@st.cache_data()
def load_model_and_tokenizer():
    model_name = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return sentiment_analyzer

sentiment_analyzer = load_model_and_tokenizer()

# Function to analyze emotion
def analyze_emotion(text):
    if text.strip() == "":
        return "Please enter some text to analyze."
    
    result = sentiment_analyzer(text)
    emotion = result[0]["label"]
    confidence = result[0]["score"]
    return f"Emotion: {emotion.capitalize()} (Confidence: {confidence:.2f})"

# Analyze button
if st.button("Analyze Emotion"):
    result = analyze_emotion(user_input)
    st.write(result)

