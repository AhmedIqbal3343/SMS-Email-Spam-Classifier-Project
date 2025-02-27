import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download("stopwords")

# Initialize PorterStemmer
ps = PorterStemmer()


# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]

    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    y = [ps.stem(i) for i in y]

    return ' '.join(y)


# Load vectorizer and model
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
model = pickle.load(open('bnb_model.pkl', 'rb'))

# Streamlit UI
st.set_page_config(page_title='SMS/Email Spam Classifier', page_icon='📩', layout='centered')

# Apply full background color properly
st.markdown("""
    <style>
        .stApp {
            background-color: #1E1E1E;
        }
    </style>
""", unsafe_allow_html=True)

# Stylish header
st.markdown("""
    <div style="background-color:#444;padding:20px;margin-bottom:20px;border-radius:10px;">
        <h2 style="color:white;text-align:center;">📩 SMS/Email Spam Classifier</h2>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <h3 style="color:#E0E0E0; text-align:center;">💬 Enter your message below to check if it's spam or not:</h3>
""", unsafe_allow_html=True)

# User input
sms = st.text_area("", placeholder="Type your message here...", height=150)

# Predict button
if st.button("🔍 Predict", help="Click to classify your message"):
    if sms.strip() != "":
        transformed_sms = transform_text(sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.markdown("""
                <div style="background-color:#D32F2F;padding:15px;margin-top:20px;border-radius:10px;">
                    <h3 style="color:white;text-align:center;">🚨 Spam Message</h3>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="background-color:#388E3C;padding:15px;margin-top:20px;border-radius:10px;">
                    <h3 style="color:white;text-align:center;">✅ Not Spam</h3>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Please enter a message to classify.")
