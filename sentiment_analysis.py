import streamlit as st
import pickle
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

st.title("Aentiment Analysis of Financial data")
st.write("This app uses a Naive Bayes Classifier to predict sentiment from text.")

with open('model.pkl', 'rb') as file:
    senti_model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()  # Lowercase the text
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and special characters
    tokens = word_tokenize(text)  # Tokenize using punkt (correct one)
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatize and remove stopwords
    return ' '.join(cleaned)

user_text = st.text_input("Enter your text:")

predict_btn = st.button("Predict")

if predict_btn:
    preprocessed_text = preprocess(user_text)
    vectorized_text = vectorizer. transform( [preprocessed_text])
    prediction = senti_model.predict(vectorized_text)
    st.write("Prediction:", prediction[0])

