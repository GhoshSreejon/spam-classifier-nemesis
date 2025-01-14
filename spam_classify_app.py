import streamlit as st
import pickle
import string
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# from jupyter ntbk
def transform_txt(txt):
    txt = txt.lower()  # to lowercase
    txt = nltk.word_tokenize(txt)  # extracts words from a text and converts to a list

    y = []  # empty list
    for i in txt:
        if i.isalnum():  # checks if the str is alphanumeric then appends, else skip
            y.append(i)

    txt = y[:]  # cloning the list
    y.clear()  # emptying y var

    for i in txt:  # stopwords and punctuations removed
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    txt = y[:]
    y.clear()

    for i in txt:  # stemming
        y.append(ps.stem(i))

    return " ".join(y)  # return as a string



tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email / SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_txt(input_sms)

    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. predict
    result = model.predict(vector_input)[0]

    # 4. display
    if (result==1):
        st.header("Spam")
    else:
        st.header("Not Spam")
