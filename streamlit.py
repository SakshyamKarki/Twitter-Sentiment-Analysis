import streamlit as st
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

st.title("Sentiment Analysis System")

# =========================
# 1. Load Models
# =========================

# Classical ML
lr_model = joblib.load("models/linear_regression.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# RNN & LSTM
rnn_model = load_model("models/rnn_model.h5")
lstm_model = load_model("models/lstm_model.h5")

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# BERT (CPU)
device = torch.device("cpu")
bert_tokenizer = DistilBertTokenizer.from_pretrained("models/distilbert_model")
bert_model = DistilBertForSequenceClassification.from_pretrained("models/distilbert_model")
bert_model.to(device)
bert_model.eval()

# Labels
labels = ['Negative', 'Neutral', 'Positive']

# =========================
# 2. Prediction Functions
# =========================
def predict_lr(text):
    X = vectorizer.transform([text])
    return labels[lr_model.predict(X)[0]]

def predict_rnn(text, maxlen=100):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=maxlen)
    pred = np.argmax(rnn_model.predict(pad, verbose=0))
    return labels[pred]

def predict_lstm(text, maxlen=100):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=maxlen)
    pred = np.argmax(lstm_model.predict(pad, verbose=0))
    return labels[pred]

def predict_bert(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return labels[pred]

# =========================
# 3. Streamlit UI
# =========================
text_input = st.text_area("Enter Text Here:")

if st.button("Predict Sentiment"):
    if text_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Predicting..."):
            st.write("Logistic Regression:", predict_lr(text_input))
            st.write("RNN:", predict_rnn(text_input))
            st.write("LSTM:", predict_lstm(text_input))
            st.write("BERT:", predict_bert(text_input))