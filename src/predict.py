import os
import pickle
from train import TextPreprocessor  # Reusing your exact cleaning pipeline

def load_inference_artifacts(models_dir: str):
    """Loads the trained model and vectorizer from disk."""
    model_path = os.path.join(models_dir, 'baseline_model.pkl')
    vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Model artifacts missing. Please run train.py first.")
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
        
    return model, vectorizer

def main():
    MODELS_DIR = "models"
    
    # Class mapping from the Davidson dataset structure
    class_labels = {
        0: "⚠️ Hate Speech Detected",
        1: "👀 Offensive Language Detected",
        2: "✅ Clean / Neutral Text"
    }
    
    print("Loading content moderation engine baseline...")
    try:
        model, vectorizer = load_inference_artifacts(MODELS_DIR)
        preprocessor = TextPreprocessor()
        print("Engine ready! Type 'exit' or 'quit' to stop.\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    while True:
        user_input = input("Enter text to evaluate >>> ")
        if user_input.strip().lower() in ['exit', 'quit']:
            print("Shutting down engine. Goodbye!")
            break
            
        if not user_input.strip():
            continue
            
        # 1. Apply the exact same cleaning used during training
        cleaned_text = preprocessor.clean_text(user_input)
        
        # 2. Transform text using the saved TF-IDF vectorizer
        vectorized_text = vectorizer.transform([cleaned_text])
        
        # 3. Predict class
        prediction = model.predict(vectorized_text)[0]
        result_label = class_labels.get(prediction, "Unknown Class")
        
        # Output results to terminal
        print("-" * 50)
        print(f"Processed Text: \"{cleaned_text}\"")
        print(f"Moderation Verdict: {result_label}")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()