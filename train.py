import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ── 1. Load dataset ──────────────────────────────────────────────
from sklearn.datasets import load_files
import urllib.request
import tarfile
import os

# Download the IMDB dataset if not already present
if not os.path.exists("aclImdb"):
    print("Downloading IMDB dataset (~84MB), this takes about 30 seconds...")
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    urllib.request.urlretrieve(url, "aclImdb_v1.tar.gz")
    print("Extracting...")
    with tarfile.open("aclImdb_v1.tar.gz", "r:gz") as tar:
        tar.extractall(".")
    print("Done!")

# Load the data
print("Loading reviews...")
train_data = load_files("aclImdb/train", categories=["pos", "neg"], encoding="utf-8")
df = pd.DataFrame({"text": train_data.data, "label": ["positive" if t == 1 else "negative" for t in train_data.target]})
print(f"Dataset loaded: {len(df)} samples")
print(df["label"].value_counts())

# ── 2. Preprocessing ─────────────────────────────────────────────
def preprocess(text):
    text = text.lower()                          # lowercase everything
    text = re.sub(r"[^a-z\s]", "", text)        # remove punctuation & numbers
    text = re.sub(r"\s+", " ", text).strip()    # collapse extra whitespace
    return text

df["clean_text"] = df["text"].apply(preprocess)
print("\nSample after preprocessing:")
print(df[["text", "clean_text"]].head(3))

# ── 3. Train / test split ─────────────────────────────────────────
# 80% of data trains the model, 20% tests how well it learned
# random_state=42 means we get the same split every time (reproducible)
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"],
    test_size=0.2,
    random_state=42
)
print(f"\nTraining samples: {len(X_train)}, Test samples: {len(X_test)}")

# ── 4. TF-IDF Vectorization ───────────────────────────────────────
# Converts text into a matrix of numbers the model can learn from
# max_features limits vocabulary size (important for large datasets)
vectorizer = TfidfVectorizer(max_features=500, stop_words="english")

# IMPORTANT: fit_transform on TRAIN only, transform on TEST
# If you fit on test data too, the model "cheats" by seeing test info
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"\nVectorized shape: {X_train_vec.shape}")
print("(rows = samples, columns = unique words in vocabulary)")

# ── 5. Train the model ────────────────────────────────────────────
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
print("\nModel trained!")

# ── 6. Evaluate ───────────────────────────────────────────────────
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2%}")
print("\nDetailed report:")
print(classification_report(y_test, y_pred))

# ── 7. Save model & vectorizer ────────────────────────────────────
# pickle serializes Python objects to disk so Flask can reload them
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel saved to model.pkl")
print("Vectorizer saved to vectorizer.pkl")

# ── 8. Quick sanity check ─────────────────────────────────────────
def predict_sentiment(text):
    clean = preprocess(text)
    vec = vectorizer.transform([clean])
    label = model.predict(vec)[0]
    confidence = model.predict_proba(vec).max()
    return label, confidence

test_sentences = [
    "This is the best thing I have ever bought",
    "Absolutely terrible, I want a refund",
    "The product is okay I guess",
]

print("\nSanity check predictions:")
for sentence in test_sentences:
    label, confidence = predict_sentiment(sentence)
    print(f"  '{sentence}'")
    print(f"   → {label} ({confidence:.2%} confidence)\n")