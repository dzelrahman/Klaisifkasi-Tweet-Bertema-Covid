import streamlit as st
import pickle
import pandas as pd
import nltk

model = pickle.load(open("model.pkl", "rb"))

st.cache()

def data_pro(text):
    df = {"text":[text]}
    df = pd.DataFrame(df)

tokens = []
def num_token(df):
    for line in df.text:
        token = nltk.word_tokenize(line)
        tokens.append(len(token))
        df['No_Token'] = pd.Series(tokens)

def split(text):
    return text

character = []
def char(df):
    for x in df.text.iloc[0:]:
        chara = split(x)
        character.append(len(chara))
        df['No_Characters'] = character

def predict(df):
    prediction = model.predict(df)
    return prediction

st.title("Aplikasi pendeteksi teks yang berhubungan dengan Covid")

text = st.text_input("Text")
df = {"text":[text]}
df = pd.DataFrame(df)
num_token(df)
char(df)
df = df.drop(["text"], axis=1)

if st.button("Classify"):
    output = predict(df=df)

    if output == 1:
        st.markdown("Tweet Anda berhubungan dengan Covid")
    else:
        st.markdown("Tweet Anda tidak berhubungan dengan Covid")
