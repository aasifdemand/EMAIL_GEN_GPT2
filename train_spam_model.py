import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack, csr_matrix
import joblib

from feature_extractor import extract_email_features

# === Correct Paths ===
DATA_PATH = os.path.join("data", "spam_assassin.csv")
MODEL_PATH = os.path.join("models", "spam_model_advanced.pkl")
VECTORIZER_PATH = os.path.join("models", "vectorizer_advanced.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")

os.makedirs("models", exist_ok=True)

# === Load Dataset ===
print("📂 Loading dataset...")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Dataset not found at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
text_col = "text"
label_col = "target"

if text_col not in df.columns or label_col not in df.columns:
    raise KeyError(f"❌ Columns '{text_col}' or '{label_col}' missing in dataset!")

df = df.dropna(subset=[text_col, label_col])
df[text_col] = df[text_col].astype(str)
print(f"📊 Loaded {len(df)} valid samples")

# === Feature Extraction ===
print("🔍 Extracting engineered features...")
engineered_features = []
for i, text in enumerate(df[text_col]):
    feats = extract_email_features(text)
    engineered_features.append(feats)
    if (i + 1) % 200 == 0:
        print(f"🔍 Processed {i + 1}/{len(df)} samples...")

engineered_df = pd.DataFrame(engineered_features)
print(f"✅ Extracted {engineered_df.shape[1]} engineered features")

# === Split Dataset ===
X_text = df[text_col]
y = df[label_col]

X_train_text, X_test_text, y_train, y_test, X_train_extra, X_test_extra = train_test_split(
    X_text, y, engineered_df, test_size=0.2, random_state=42, stratify=y
)

# === Vectorization (TF-IDF) ===
vectorizer = TfidfVectorizer(
    stop_words="english",
    lowercase=True,
    max_features=8000,
    ngram_range=(1, 2)
)
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

# === Scale Engineered Features ===
scaler = MinMaxScaler()
X_train_extra_scaled = scaler.fit_transform(X_train_extra)
X_test_extra_scaled = scaler.transform(X_test_extra)

# === Combine TF-IDF + Engineered Features ===
X_train_combined = hstack([X_train_tfidf, csr_matrix(X_train_extra_scaled)]).tocsr()
X_test_combined = hstack([X_test_tfidf, csr_matrix(X_test_extra_scaled)]).tocsr()

# === Train Model ===
print("🤖 Training MultinomialNB model...")
model = MultinomialNB()
model.fit(X_train_combined, y_train) # type: ignore

# === Evaluate Model ===
y_pred = model.predict(X_test_combined) # type: ignore
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc * 100:.2f}%")
print(classification_report(y_test, y_pred, digits=3))

# === Save Model Artifacts ===
joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)
joblib.dump(scaler, SCALER_PATH)

print(f"💾 Model saved to {MODEL_PATH}")
print(f"💾 Vectorizer saved to {VECTORIZER_PATH}")
print(f"💾 Scaler saved to {SCALER_PATH}")
print("🎉 Training complete!")
