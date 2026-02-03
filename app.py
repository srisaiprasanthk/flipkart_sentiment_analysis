import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ===============================
# Page Config
# ===============================
st.set_page_config(page_title="Flipkart Sentiment Analyzer")

# ===============================
# Download NLTK (only once)
# ===============================
@st.cache_resource
def load_nltk():
    nltk.download("stopwords")
    nltk.download("wordnet")

load_nltk()

# ===============================
# Load Model + Vectorizer (CACHED)
# ===============================
@st.cache_resource
def load_model():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ===============================
# Text Preprocessing
# ===============================
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)

    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)

# ===============================
# Streamlit UI
# ===============================
st.title("🛒 Flipkart Product Review Sentiment Analysis")
st.write("Enter a review to detect whether it is Positive or Negative.")

review = st.text_area("Enter Review Text")

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review")

    else:
        with st.spinner("Analyzing sentiment..."):
            cleaned = clean_text(review)
            vec = vectorizer.transform([cleaned])
            prediction = model.predict(vec)[0]

        if prediction == 1:
            st.success("✅ Positive Review 😊")
        else:
            st.error("❌ Negative Review 😞")
