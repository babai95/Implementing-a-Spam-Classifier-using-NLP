# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 00:40:36 2020

@author: sayagupt
"""
import nltk
nltk.download()
import numpy as np
import pickle
import pandas as pd
import streamlit as st
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

pickle_in = open("model.pkl","rb")
classifier=pickle.load(pickle_in)

if __name__ == "__main__":
    st.title("Spam Classification")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Get to know whether message is Spam or not </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    text = st.text_input("Input your text here")
    
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()

    words = nltk.word_tokenize(text)
    for j, word in enumerate(words):
        if word in set(stopwords.words('english')):
            words.remove(word)
        else:
            words[j] = lemmatizer.lemmatize(word)
    text = " ".join(words)
    
    text = [text]
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    cv = TfidfVectorizer()
    X = cv.fit_transform(text).toarray()
    
    str = "Predict whether message is spam or not"
    if st.button("Predict"):
        y = classifier.predict(vect)
        if(y == 0):
            str = "The message is not a spam"
        else:
            str = "Beware!! Its a spam message!"
        
    st.success(f'{str}')
    
        