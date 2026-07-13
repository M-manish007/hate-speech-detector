import os
import re
import pickle
import pandas as pd

# NLP and Machine Learning tools
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# Ensure all required NLTK assets are downloaded locally
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading required NLTK resources...")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')

class TextPreprocessor:
    """Handles text cleaning and normalization for the NLP pipeline."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text: str) -> str:
        """Applies regex and NLTK transformations to clean raw tweets/text."""
        if not isinstance(text, str):
            return ""
        
        # 1. Lowercase
        text = text.lower()
        
        # 2. Remove URLs, mentions, and hashtags
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#", "", text)
        
        # 3. Tokenize
        tokens = word_tokenize(text)
        
        # 4. Filter out punctuation and stop words
        cleaned_tokens = [
            word for word in tokens 
            if word.isalnum() and word not in self.stop_words
        ]
        
        return " ".join(cleaned_tokens)


def load_and_prepare_data(data_path: str) -> pd.DataFrame:
    """Loads dataset and applies text cleaning across the text column."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}. Please check your path.")
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Map Davidson dataset columns
    if 'tweet' in df.columns:
        df = df.rename(columns={'tweet': 'text'})
    if 'class' in df.columns:
        df = df.rename(columns={'class': 'label'})
        
    print("Running text preprocessing pipeline...")
    preprocessor = TextPreprocessor()
    df['clean_text'] = df['text'].apply(preprocessor.clean_text)
    
    df = df[df['clean_text'].str.strip() != ""]
    return df


def train_pipeline(data_path: str, models_dir: str) -> None:
    """Executes the full training, evaluation, and serialization lifecycle."""
    df = load_and_prepare_data(data_path)
    
    X = df['clean_text']
    y = df['label']
    
    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"Train set size: {len(X_train)} | Test set size: {len(X_test)}")
    
    # Text Vectorization using TF-IDF
    print("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Model Training
    print("Training LinearSVC baseline model...")
    model = LinearSVC(class_weight='balanced', random_state=42, dual=False)
    model.fit(X_train_vec, y_train)
    
    # Comprehensive Evaluation
    print("\n=== Model Performance Evaluation ===")
    y_pred = model.predict(X_test_vec)
    print(classification_report(y_test, y_pred))
    
    # Artifact Serialization
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'baseline_model.pkl')
    vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
    
    print(f"Saving artifacts to '{models_dir}' directory...")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print("Pipeline execution complete!")


if __name__ == "__main__":
    DATA_FILE_PATH = os.path.join("data", "raw", "labeled_data.csv")
    MODELS_OUTPUT_DIR = "models"
    
    train_pipeline(DATA_FILE_PATH, MODELS_OUTPUT_DIR)