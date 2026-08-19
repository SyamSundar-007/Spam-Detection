import pickle as pkl
import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

model = pkl.load(open('model.pkl', 'rb') )
tfidf_vector = pkl.load( open('tfidf.pkl', 'rb'))
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def text_preprocessing(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = word_tokenize(text)
    text=  [words for  words in text  if words not in stop_words]
    text=  [lemmatizer.lemmatize(words) for words in text]
    return " ".join(text)

def predict_spam(msg):
    cleand_txt = [text_preprocessing(msg)]
    X = tfidf_vector.transform(cleand_txt )
    # y_predict =  model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    threshold = 0.2
    y_pred_new = (y_prob >= threshold).astype(int)
    return y_pred_new

st.header("Spam Detection Classifier")
st.write("You Think it's a spam 🫤 let's find out.")
message = st.text_input(
    label="Please put your here to check the spam: ",
    placeholder="Type something detailed...",
    # height=100 # Optional: height of the text box in pixels
)

import streamlit as st

st.markdown("""
<style>
div.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 12px;
    border: none;
    background: linear-gradient(90deg, #6a5acd, #8a2be2);
    color: white;
    font-size: 18px;
    font-weight: 600;
    letter-spacing: 0.5px;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0px 5px 15px rgba(106, 90, 205, 0.35);
}

div.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0px 8px 20px rgba(138, 43, 226, 0.45);
}

div.stButton > button:active {
    transform: scale(0.98);
}
</style>
""", unsafe_allow_html=True)


if st.button("🚀 Detect Spam"):
    if message != "":
        prediction = predict_spam(message)
        if prediction == 1:
            output =  "a Spam"
            st.error(f'Our Prediction: This is {output} Message')
        else:
            output = "Not a Spam"
            st.success(f'Our Prediction: This is {output} Message' )

