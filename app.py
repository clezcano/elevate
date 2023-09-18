import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Set up Streamlit
st.title("Emotion Detection with Transformers")

# Text input
user_input = st.text_area("Enter your text:")

# Function to load model and tokenizer using @st.cache_data
@st.cache_data()
def load_model_and_tokenizer():
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# Function to analyze emotion
def analyze_emotion(text):
    if text.strip() == "":
        return "Please enter some text to analyze."
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    # Emotion labels (customize as needed)
    emotion_labels = ["anger", "joy", "sadness", "neutral"]
    emotion = emotion_labels[predicted_class]
    
    return f"Emotion: {emotion.capitalize()}"

# Analyze button
if st.button("Analyze Emotion"):
    result = analyze_emotion(user_input)
    st.write(result)
