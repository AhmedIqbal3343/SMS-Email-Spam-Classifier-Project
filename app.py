import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

# Explicitly download required NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')

# Initialize PorterStemmer
ps = PorterStemmer()

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    
    # Tokenization
    text = word_tokenize(text)

    # Remove special characters and punctuation
    y = [i for i in text if i.isalnum()]

    # Remove stopwords
    y = [i for i in y if i not in stopwords.words('english')]

    # Apply stemming
    y = [ps.stem(i) for i in y]

    return ' '.join(y)

# Load vectorizer and model
try:
    tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    model = pickle.load(open('bnb_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: Model or vectorizer file not found! Please check the file paths.")

# Streamlit UI
st.set_page_config(page_title='SMS/Email Spam Classifier', page_icon='📩', layout='centered')

# Stylish header
st.markdown("""
    <h2 style="text-align:center;">📩 SMS/Email Spam Classifier</h2>
""", unsafe_allow_html=True)

# User input
sms = st.text_area("", placeholder="Type your message here...", height=150)

# Predict button
if st.button("🔍 Predict"):
    if sms.strip():
        transformed_sms = transform_text(sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")
    else:
        st.warning("Please enter a message to classify.")
