import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk

# ===============================
# DOWNLOAD NLTK
# ===============================
for pkg in ["punkt", "stopwords"]:
    nltk.download(pkg, quiet=True)

stop_id = set(stopwords.words("indonesian"))
extra_stop = {"yang","dan","di","ke","dari","pada","untuk","adalah","dengan","atau","sebagai"}
stop_id.update(extra_stop)

stemmer = PorterStemmer()

# ===============================
# LOAD DATA
# ===============================
DATASET = r"C:\Users\mzaid\OneDrive\Documents\TugasU\ChatbotILKOM\translated_computer_science_dataset.csv"
df = pd.read_csv(DATASET, engine="python", on_bad_lines="skip")
df = df[["input_id", "output_id"]].dropna().reset_index(drop=True)

print("\n==============================")
print("📌 PENGUMPULAN DATA (TABEL 1)")
print("==============================")
print(df.head(3).to_string(), "\n")

# ===============================
# PREPROCESSING FUNCTION
# ===============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^0-9a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_id and len(t) > 1]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

# ===============================
# PREPROCESSING SHOWCASE
# ===============================
sample = "Apa saja cara berbeda untuk memasukkan data ke dalam program?"

print("\n==============================")
print("📌 PENGOLAHAN DATA (TABEL 2)")
print("==============================")
print("Teks asli:", sample)
print("Lowercase:", sample.lower())

tokens = word_tokenize(sample.lower())
print("Tokenisasi:", tokens)

stop_removed = [t for t in tokens if t not in stop_id]
print("Stopword removed:", stop_removed)

stemmed = [stemmer.stem(t) for t in stop_removed]
print("Stemming:", stemmed)

clean = clean_text(sample)
print("Plain Text:", clean, "\n")

# ===============================
# TF-IDF
# ===============================
df["clean_input"] = df["input_id"].apply(clean_text)
vectorizer = TfidfVectorizer(ngram_range=(1,2))
tfidf_matrix = vectorizer.fit_transform(df["clean_input"])

print("\n==============================")
print("📌 PEMBOBOTAN TF-IDF")
print("==============================")
print("Jumlah dokumen :", tfidf_matrix.shape[0])
print("Jumlah fitur   :", tfidf_matrix.shape[1])

query_vec = vectorizer.transform([clean])
nz = query_vec.nonzero()[1]
top_weights = sorted([(vectorizer.get_feature_names_out()[i], query_vec[0, i]) for i in nz],
                     key=lambda x: x[1], reverse=True)[:10]

print("\nTop 10 bobot TF-IDF query:")
for w, s in top_weights:
    print(f"- {w} : {s:.5f}")

# ===============================
# COSINE SIMILARITY
# ===============================
sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
top10_idx = sim.argsort()[-10:][::-1]

print("\n==============================")
print("📌 PERHITUNGAN COSINE SIMILARITY (TABEL 3)")
print("==============================")
for i in top10_idx:
    print(f"- {df.loc[i,'input_id'][:50]}... | Score: {sim[i]:.3f}")

# ===============================
# EVALUASI (PRECISION / RECALL / F1)
# ===============================
relevant = set()
query_terms = set(clean.split())

for i, row in df.iterrows():
    tokens = set(row["clean_input"].split())
    if not query_terms.isdisjoint(tokens):
        relevant.add(i)

retrieved = set(top10_idx)
tp = len(relevant.intersection(retrieved))
fp = len(retrieved - relevant)
fn = len(relevant - retrieved)
tn = len(df) - (tp + fp + fn)

precision = tp / (tp + fp) if tp + fp > 0 else 0
recall = tp / (tp + fn) if tp + fn > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

print("\n==============================")
print("📌 EVALUASI SISTEM (TABEL 4)")
print("==============================")
print("TP :", tp)
print("FP :", fp)
print("FN :", fn)
print("TN :", tn)
print("Precision :", precision)
print("Recall    :", recall)
print("F1-Score  :", f1)

print("\n==============================")
print("📌 CONFUSION MATRIX (TABEL 5)")
print("==============================")
print("                 Pred_Relevan   Pred_Tidak")
print(f"Aktual Relevan      {tp:5d}         {fn:5d}")
print(f"Aktual Tidak        {fp:5d}         {tn:5d}")

print("\n🎉 Semua hasil BAB III berhasil ditampilkan di terminal!\n")
