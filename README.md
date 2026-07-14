# 🛡️ Hate Speech & Content Moderation Engine

A production-ready NLP machine learning pipeline designed to detect offensive language, hate speech, or clean content in real-time. This application is built using a machine learning classifier, feature-engineered TF-IDF pipelines, and served instantly via a lightweight Streamlit web application.

🚀 **[Live Demo: Run the Web App Here](https://hate-speech-detector-mb6avssvpcpkjomr6nnfmt.streamlit.app/)**

---

## 📌 Project Overview
Online platforms process millions of user-generated posts daily. Moderating this text at scale is critical to preventing harassment and maintaining platform safety. This project implements an end-to-end NLP text classification model capable of sorting raw user inputs into three distinct categories:
1. **Hate Speech** ⚠️
2. **Offensive Language** 👀
3. **Clean / Neutral Text** ✅

---

## 🛠️ Features & Pipeline
* **Custom Text Preprocessor:** Implements tokenization, lowercase conversion, and punctuation removal using NLTK to streamline inputs.
* **Vectorization:** Converts clean text into statistical numerical features using a `TfidfVectorizer` pipeline.
* **Classification Model:** Employs a trained machine learning model optimized to balance precision and recall on highly imbalanced text datasets.
* **Interactive UI:** A highly intuitive Streamlit web application allows users to paste raw strings or tweets and view instant, color-coded moderation verdicts along with details about the processed tokens.

---

## 📂 Repository Structure
```text
├── .devcontainer/         # Dev container configurations for cloud environments
├── data/
│   └── raw/               # Location of raw datasets used for training
├── models/
│   ├── baseline_model.pkl # Trained classification model file (binary payload)
│   └── tfidf_vectorizer.pkl  # Trained TF-IDF vectorizer parameters
├── src/
│   ├── __init__.py
│   └── train.py           # Preprocessing pipeline and training scripts
├── app.py                 # Streamlit web application front-end
├── requirements.txt       # Dependencies for running and deploying the app
└── README.md              # Project documentation
