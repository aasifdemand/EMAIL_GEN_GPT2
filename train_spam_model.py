import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.naive_bayes import MultinomialNB # type: ignore
from sklearn.metrics import accuracy_score, classification_report # type: ignore
import joblib # type: ignore
import os

# === 1. Load dataset ===
data_path = "data/spam_assassin.csv"
data = pd.read_csv(data_path)

# ✅ Match your dataset columns
text_column = "text"
label_column = "target"

# Drop rows with missing text
data = data.dropna(subset=[text_column])

X = data[text_column]
y = data[label_column]

# === 2. Split data ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 3. Text vectorization ===
vectorizer = TfidfVectorizer(
    stop_words='english',
    lowercase=True,
    max_features=5000
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# === 4. Train model ===
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# === 5. Evaluate ===
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully! Accuracy: {acc:.2f}")
print(classification_report(y_test, y_pred))

# === 6. Save model and vectorizer ===
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/spam_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("💾 Model and vectorizer saved in /models/")
