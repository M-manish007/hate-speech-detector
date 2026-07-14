import os
import pickle
import streamlit as st
from src.train import TextPreprocessor  # Reusing your exact cleaning pipeline

# Configure the web page
st.set_page_config(
    page_title="Hate Speech Detection Engine",
    page_icon="🛡️",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    """Loads and caches the model and vectorizer using robust relative-to-absolute paths."""
    # Get the directory where app.py is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build absolute paths to the pickled models
    model_path = os.path.join(base_dir, "models", "baseline_model.pkl")
    vectorizer_path = os.path.join(base_dir, "models", "tfidf_vectorizer.pkl")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
        
    return model, vectorizer

# App Header UI
st.title("🛡️ Hate Speech & Content Moderation Engine")
st.markdown("""
This production-ready NLP pipeline analyzes text inputs to identify hate speech, offensive language, or clean content. 
*Built using LinearSVC and TF-IDF.*
""")

try:
    model, vectorizer = load_artifacts()
    preprocessor = TextPreprocessor()
except Exception as e:
    st.error(f"Failed to load model artifacts: {e}")
    st.stop()

# User Input area
user_input = st.text_area("Enter text or a tweet to evaluate:", height=100, placeholder="Type something here...")

if st.button("Run Moderation Check", type="primary"):
    if user_input.strip():
        # 1. Clean the input text using your custom NLTK pipeline
        cleaned_text = preprocessor.clean_text(user_input)
        
        # 2. Vectorize
        vectorized_text = vectorizer.transform([cleaned_text])
        
        # 3. Predict
        prediction = model.predict(vectorized_text)[0]
        
        # 4. Display Results beautifully
        st.subheader("Verdict")
        if prediction == 0:
            st.error("⚠️ Hate Speech Detected")
        elif prediction == 1:
            st.warning("👀 Offensive Language Detected")
        else:
            st.success("✅ Clean / Neutral Text")
            
        with st.expander("See Pipeline Details"):
            st.write(f"**Processed Tokens:** `{cleaned_text if cleaned_text else '[Empty after cleaning]'}`")
    else:
        st.info("Please enter some text to test.")