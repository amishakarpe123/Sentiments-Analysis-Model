import streamlit as st
import pickle
import numpy as np
import requests
from streamlit_lottie import st_lottie

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Sentiment Analysis AI",
    page_icon="🔮",
    layout="centered"
)

# --- CUSTOM CSS FOR BETTER AESTHETICS ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .title-text {
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle-text {
        color: #4B5563;
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_style_html=True)

# --- HELPER FUNCTION FOR LOTTIE ANIMATIONS ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# Load animation (A cool, floating brain/AI animation)
lottie_ai = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_m9unqem6.json")

# --- APP HEADER ---
st.markdown("<h1 class='title-text'>🔮 Sentiment Analysis Dashboard</h1>", unsafe_style_html=True)
st.markdown("<p class='subtitle-text'>Enter your text below to analyze its emotional tone using Machine Learning.</p>", unsafe_style_html=True)

if lottie_ai:
    st_lottie(lottie_ai, height=200, key="coding")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    # Loading your uploaded nlp_model.pkl
    with open("nlp_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- USER INPUT SECTION ---
st.markdown("### 📝 Input Text")
user_text = st.text_area(
    "What's on your mind?", 
    placeholder="Type something positive or negative here...",
    height=150
)

# --- PREDICTION LOGIC ---
if st.button("🚀 Analyze Sentiment", use_container_width=True):
    if user_text.strip() == "":
        st.warning("Please enter some text before analyzing!")
    else:
        with st.spinner("Analyzing text patterns... Please wait..."):
            try:
                # NOTE: If your model is a standalone MultinomialNB, it expects vectorized input.
                # If it's an sklearn Pipeline (Vectorizer + Model), it accepts raw text directly.
                
                # Attempting prediction assuming it's a Pipeline or handles text:
                if hasattr(model, "predict"):
                    # Wrapping in a list because NLP models expect an iterable of documents
                    prediction = model.predict([user_text])[0]
                    
                    st.balloons() # Fun animation celebration
                    st.markdown("---")
                    st.markdown("### 📊 Analysis Result")
                    
                    if prediction == "positive" or prediction == 1:
                        st.success(f"### 🎉 Positive Sentiment Detected!")
                        st.markdown("The text leans toward a cheerful, confident, or constructive tone.")
                    else:
                        st.error(f"### ⚠️ Negative Sentiment Detected!")
                        st.markdown("The text leans toward a critical, frustrated, or concerned tone.")
                        
            except Exception as e:
                st.error("Prediction failed.")
                st.info(
                    "💡 **Developer Note:** This model (MultinomialNB) might require a pre-processing "
                    "vectorizer step (like `CountVectorizer`). If you have a `vectorizer.pkl` file, "
                    "transform the text using `vectorizer.transform([user_text])` before passing it to the model."
                )
                # Printing the technical error to logs for debugging
                print(f"Prediction error: {e}")
