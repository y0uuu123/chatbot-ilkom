import os
import re
import numpy as np
import pandas as pd
import nltk
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
#   NLTK SETUP
# ===============================
for pkg in ["punkt", "stopwords"]:
    nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


# ===============================
#   LOAD DATASET
# ===============================
DATA_PATH = r"D:\STK\ChatbotILKOM\translated_computer_science_dataset.csv"

def load_dataset(path=DATA_PATH):
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    return df[["input_id", "output_id"]].dropna().reset_index(drop=True)


# ===============================
#   STOPWORDS
# ===============================
stop_id = set(stopwords.words("indonesian"))
extra_stop = {
    "yang", "dan", "di", "ke", "dari", "pada", "untuk",
    "adalah", "dengan", "atau", "sebagai"
}
stop_id.update(extra_stop)

stemmer = PorterStemmer()


# ===============================
#   PREPROCESSING
# ===============================
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|\S+@\S+", " ", text)
    text = re.sub(r"[^0-9a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens_clean = [t for t in tokens if t not in stop_id and len(t) > 1]

    tokens_stem = [stemmer.stem(t) for t in tokens_clean]

    return " ".join(tokens_stem)


# ===============================
#   CHATBOT
# ===============================
class Chatbot:
    def __init__(self, df, threshold=0.25):
        self.df = df.copy()

        # cleaning seluruh input dataset
        self.df["clean_input"] = self.df["input_id"].apply(clean_text)

        # TF-IDF unigram + bigram
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["clean_input"])

        self.threshold = threshold

    def get_response(self, user_text):

        clean_user = clean_text(user_text)
        vec = self.vectorizer.transform([clean_user])
        sims = cosine_similarity(vec, self.tfidf_matrix).flatten()

        best_idx = np.argmax(sims)
        best_score = sims[best_idx]

        # ==========================
        # DETEKSI OUT-OF-CONTEXT
        # ==========================
        if best_score < 0.60:
            return None, "⚠️ Pertanyaan di luar konteks ilmu komputer.", best_score

        # kalau similarity masih kurang
        if best_score < self.threshold:
            return None, "❌ Tidak ditemukan jawaban relevan.", best_score

        best_q = self.df.loc[best_idx, "input_id"]
        best_a = self.df.loc[best_idx, "output_id"]

        return best_q, best_a, best_score


# ===============================
#   FLASK APP
# ===============================
app = Flask(__name__)

df = load_dataset()
bot = Chatbot(df)


@app.route("/", methods=["GET", "POST"])
def home():
    question = None
    answer = None
    similarity = None

    if request.method == "POST":
        user_input = request.form.get("question")

        if user_input.strip():
            result = bot.get_response(user_input)
            question, answer, similarity = result

    return render_template(
        "index.html",
        question=question,
        answer=answer,
        similarity=similarity
    )


if __name__ == "__main__":
    app.run(debug=True)
