import os
import re
import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity    
from sklearn.metrics import confusion_matrix
import tkinter as tk
from tkinter import scrolledtext, messagebox

# --- Stemmer Bahasa Indonesia
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
except:
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()

# --- NLTK Setup
for pkg in ["punkt", "punkt_tab", "stopwords"]:
    nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Load Stopword
try:
    stop_id = set(stopwords.words("indonesian"))
except LookupError:
    stop_id = set()

extra_stop = {"yang", "dan", "di", "ke", "dari", "pada", "untuk", "adalah", "dengan", "atau", "sebagai"}
stop_id = stop_id.union(extra_stop)

# --- Dataset Path
DATA_PATH = r"C:\Users\mzaid\OneDrive\Documents\TugasU\ChatbotILKOM\translated_computer_science_dataset.csv"

# --- Preprocessing Debug
def clean_text_debug(text):
    if not isinstance(text, str):
        text = str(text)
    original = text
    text = text.lower()
    text = re.sub(r"[^0-9a-zA-Z\s]", " ", text)
    tokens = word_tokenize(text)
    filtered = [t for t in tokens if t not in stop_id and len(t) > 1]
    stemmed = [stemmer.stem(t) for t in filtered]
    return {
        "original": original,
        "lowercase": text,
        "tokens": tokens,
        "filtered": filtered,
        "stemmed": stemmed,
        "clean_text": " ".join(stemmed)
    }

# --- Clean Text untuk TF-IDF
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"[^0-9a-zA-Z\s]", " ", text)
    tokens = word_tokenize(text)
    return " ".join([t for t in tokens if t not in stop_id and len(t) > 1])

# --- Load Dataset
def load_dataset(path=DATA_PATH):
    df = pd.read_csv(path)
    return df[["input_id", "output_id"]].dropna().reset_index(drop=True)

# --- Chatbot Core Class
class Chatbot:
    def __init__(self, df, threshold=0.25):
        self.df = df.copy()
        self.df["clean_input"] = self.df["input_id"].apply(clean_text)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["clean_input"])
        self.threshold = threshold

    def get_response(self, user_text):
        debug = clean_text_debug(user_text)
        vec = self.vectorizer.transform([debug["clean_text"]])
        sims = cosine_similarity(vec, self.tfidf_matrix).flatten()
        best_idx = np.argmax(sims)
        best_score = sims[best_idx]
        if best_score < self.threshold:
            return None, "Tidak relevan", best_score, debug
        return self.df.loc[best_idx, "input_id"], self.df.loc[best_idx, "output_id"], best_score, debug

# --- EVALUASI MODEL
def evaluate_model(bot, test_df):
    y_true, y_pred = [], []
    for i, row in test_df.iterrows():
        _, _, score, _ = bot.get_response(row["input_id"])
        y_pred.append(1 if score >= bot.threshold else 0)
        y_true.append(1)

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    TP, FN = cm[0][0], cm[0][1]
    FP, TN = cm[1][0], cm[1][1]

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "precision": precision, "recall": recall, "f1_score": f1}

# --- CREATE TABLE PREPROCESSING ---
def create_preprocessing_table(df, n=5):
    rows = []
    for i in range(n):
        debug = clean_text_debug(df.loc[i, "input_id"])
        rows.append({
            "No": i+1,
            "Teks Asli": debug["original"],
            "Lowercase": debug["lowercase"],
            "Tokenisasi": debug["tokens"],
            "Stopword Removed": debug["filtered"],
            "Stemming": debug["stemmed"],
            "Plain Text": debug["clean_text"]
        })
    tabel = pd.DataFrame(rows)
    tabel.to_csv("tabel_preprocessing.csv", index=False)
    return tabel

# --- MAIN PROGRAM
if __name__ == "__main__":
    df = load_dataset()
    bot = Chatbot(df, threshold=0.25)

    # TABEL PREPROCESSING
    tbl = create_preprocessing_table(df, n=5)
    print(tbl.head())
    
    # EVALUASI MODEL
    eval_result = evaluate_model(bot, df.sample(30))  # 30 sample testing
    print("📊 Hasil Evaluasi:", eval_result)
