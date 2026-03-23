Sentiment Analysis System
Project Overview

This project implements a Sentiment Analysis System that classifies text (tweets, reviews, or social media posts) into three categories: Positive, Neutral, or Negative.

It demonstrates a full NLP pipeline using multiple approaches:

Classical Machine Learning – Logistic Regression with TF-IDF features
Deep Learning – RNN and LSTM models using pre-trained GloVe embeddings
Transformer-based Model – DistilBERT (Hugging Face Transformers)

The system also includes a Streamlit UI for real-time sentiment predictions.

Project Structure
sentiment_analysis/
├─ data/
│  ├─ twitter_training.csv
│  ├─ twitter_validation.csv
│  ├─ clean_train.csv
│  ├─ clean_val.csv
│  ├─ glove.6B.100d.txt.csv
├─ models/
│  ├─ linear_regression.pkl
│  ├─ tfidf_vectorizer.pkl
│  ├─ rnn_model.h5
│  ├─ lstm_model.h5
│  ├─ tokenizer.pkl
│  └─ distilbert_model/ (pretrained DistilBERT fine-tuned)
├─ notebooks/
│  ├─ linear_regression.ipynb   # Notebook for Linear Regression training
│  ├─ preprocessing.ipynb       # Text cleaning & tokenizer scripts
│  ├─ rnn_lstm.ipynb            # Notebook for RNN/LSTM model training
│  ├─ bert.ipynb                # Notebook for BERT training
├─ app.py                       # Streamlit app
├─ README.md
├─ requirement.txt
└─ .gitignore

Installation Instructions

Clone the repository
git clone [(https://github.com/SakshyamKarki/Twitter-Sentiment-Analysis.git)]
cd sentiment_analysis
Create virtual environment
conda create -n sentiment_env python=3.10 -y
conda activate sentiment_env

Install dependencies
pip install -r requirements.txt
# Or manually:
pip install tensorflow torch transformers scikit-learn pandas numpy matplotlib streamlit joblib

Run Streamlit app
streamlit run app.py

Data Description
Training dataset: twitter_training.csv
Validation dataset: twitter_validation.csv

Each CSV contains columns:
Column Name	Description
tweet_id	Unique tweet identifier
entity	Tweet subject / entity (optional)
sentiment	Sentiment label: Positive/Neutral/Negative
tweet_content	Raw text of the tweet

Text Preprocessing
Convert text to lowercase
Remove special characters and numbers
Tokenization
Stopword removal, optional stemming/lemmatization
Sequence padding for deep learning models

Models
Model Type	Implementation Details
Logistic Regression	TF-IDF vectorizer, classical ML, baseline model
RNN	Embedding layer with GloVe 100d, SimpleRNN(64), Dense(3)
LSTM	Embedding layer with GloVe 100d, LSTM(64), Dense(3)
DistilBERT	Fine-tuned on training dataset, Hugging Face Transformers

Evaluation Metrics
Accuracy
Precision, Recall, F1-score
Confusion Matrix for error analysis

Example of expected evaluation:

Model	Accuracy	Precision	Recall	F1-score
LR	    0.82	    0.81	    0.80	0.81
RNN	    0.85	    0.84	    0.84	0.84
LSTM	0.87	    0.86	    0.86	0.86
BERT	0.91	    0.90	    0.91	0.91

Usage Instructions
Open the Streamlit app (app.py)
Enter any text in the input box
Click Predict Sentiment
Predictions from all models will be displayed

Git Ignore

Add to .gitignore to avoid committing large files:

# Models
*.h5
*.pkl
distilbert_model/
glove*.txt

# Python cache
__pycache__/
*.pyc

# Jupyter
.ipynb_checkpoints/

# System files
.DS_Store

References
Hugging Face Transformers
GloVe: Global Vectors for Word Representation
TensorFlow & Keras documentation
Scikit-learn documentation

Future Work / Enhancements
Add real-time streaming Twitter sentiment analysis
Use BERT variants (RoBERTa, XLNet) for higher accuracy
Integrate vector databases (FAISS, ChromaDB) for semantic search
Deploy as a web app using Streamlit Sharing, Heroku, or AWS