import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import sentencepiece

# Set up Streamlit
st.title("Emotion Detection with Transformers")

# Text input
user_input = st.text_area("Enter your text:")


# Function to load model and tokenizer using @st.cache_data
@st.cache_data
def load_model_and_tokenizer():
    model_name = "mrm8488/t5-base-finetuned-emotion"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name, use_fast=False)
    return tokenizer, model


tokenizer, model = load_model_and_tokenizer()


# Function to analyze emotion
def analyze_emotion(text):
    if text.strip() == "":
        return "Please enter some text to analyze."

    input_ids = tokenizer.encode(text + '</s>', return_tensors='pt')

    output = model.generate(input_ids=input_ids,
                            max_length=2)

    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output]
    label = dec[0]

    return f"Emotion: {label.capitalize()}"


# Analyze button
if st.button("Analyze Emotion"):
    result = analyze_emotion(user_input)
    st.write(result)
