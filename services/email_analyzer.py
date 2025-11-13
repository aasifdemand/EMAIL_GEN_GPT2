import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from feature_extractor import batch_extract


MODEL_PATH = os.path.join("models", "spam_model_advanced.pkl")
VECTORIZER_PATH = os.path.join("models", "vectorizer_advanced.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
scaler = joblib.load(SCALER_PATH)


def analyze_email(email_text: str, subject: str = "") -> dict:
    try:
        full_text = f"{subject}\n\n{email_text}".strip()

        X_tfidf = vectorizer.transform([full_text])
        df = pd.DataFrame({"text": [full_text]})
        engineered_df = batch_extract(df)
        engineered_scaled = scaler.transform(engineered_df)
        X_combined = hstack([csr_matrix(X_tfidf), csr_matrix(engineered_scaled)])

        spam_prob = float(model.predict_proba(X_combined)[0][1])
        label = "spam" if spam_prob > 0.5 else "ham"

        # Basic metrics
        word_count = len(email_text.split())
        sentence_count = len([s for s in email_text.split(".") if s.strip()])
        paragraph_count = len([p for p in email_text.split("\n") if p.strip()])
        question_count = email_text.count("?")
        uppercase_count = sum(1 for w in email_text.split() if w.isupper())
        link_count = email_text.lower().count("http")
        subject_len = len(subject)

        # Dynamic detections
        spam_words = [w for w in ["promotion", "free", "buy now", "discount", "risk free", "offer", "click here", "urgent"] if w in email_text.lower()]
        personalization_tags = [tag for tag in ["{{first_name}}", "{name}", "[name]", "{company}"] if tag in email_text]
        cta_present = any(word in email_text.lower() for word in ["book a call", "schedule", "reply", "learn more", "visit"])
        greeting_present = any(word in email_text.lower() for word in ["hi", "hello", "dear"])

        # Health score calculation
        health_score = max(0, 100 - int(spam_prob * 60) - len(spam_words) * 10)
        health_status = (
            "Excellent - Ready to send" if health_score >= 80
            else "Good - Minor improvements suggested" if health_score >= 60
            else "Needs Work - Review content"
        )

        # Build dynamic positives
        positives = []
        if personalization_tags:
            positives.append("Includes personalization tags — good for engagement.")
        if greeting_present:
            positives.append("Friendly greeting detected.")
        if cta_present:
            positives.append("Has a clear call-to-action — great for conversions.")
        if 50 <= word_count <= 200:
            positives.append("Balanced content length.")
        if len(positives) == 0:
            positives.append("Clean structure with minimal spam signs.")

        # Dynamic best practices
        best_practices = []
        if spam_words:
            best_practices.append({
                "title": "Avoid Spam Trigger Words",
                "description": "Replace words like " + ", ".join(spam_words) + " with neutral terms."
            })
        if not personalization_tags:
            best_practices.append({
                "title": "Add Personalization",
                "description": "Include dynamic tags like {first_name} or {company} to improve deliverability."
            })
        if uppercase_count > 3:
            best_practices.append({
                "title": "Reduce Excessive Uppercase",
                "description": "Avoid shouting in subject or body; it triggers spam filters."
            })
        if not best_practices:
            best_practices.append({
                "title": "Maintain Current Quality",
                "description": "No major issues found — continue using similar tone and structure."
            })

        # Final unified output
        return {
            "template_analytics": {
                "email_health_score": health_score,
                "status": health_status,
                "metrics": {
                    "subject": {"value": subject_len, "label": "SUBJECT", "status": "Optimal"},
                    "words": {"value": word_count, "label": "WORDS", "status": "Optimal"},
                    "sentences": {"value": sentence_count, "label": "SENTENCES", "status": "Optimal"},
                    "paragraphs": {"value": paragraph_count, "label": "PARAGRAPHS", "status": "Optimal"},
                    "links": {"value": link_count, "label": "LINKS", "status": "Optimal"},
                    "questions": {"value": question_count, "label": "QUESTIONS", "status": "Optimal"},
                    "uppercase": {"value": uppercase_count, "label": "UPPERCASE", "status": "Optimal"},
                    "spam_words": {"value": len(spam_words), "label": "SPAM WORDS", "status": "Review" if spam_words else "Optimal"},
                    "personal_tags": {"value": len(personalization_tags), "label": "PERSONAL TAGS", "status": "Optimal" if personalization_tags else "Missing"},
                },
            },
            "detailed_analysis": {
                "summary": {
                    "critical": len(spam_words),
                    "warning": 1 if uppercase_count > 3 else 0,
                    "suggestion": 1 if not personalization_tags else 0,
                    "passed": 1,
                },
                "issues": [
                    {
                        "title": "Spam Words Detected",
                        "priority": "HIGH",
                        "found": spam_words,
                        "recommendation": "Avoid common spam triggers like these."
                    } if spam_words else {},
                    {
                        "title": "Too Much Uppercase",
                        "priority": "MEDIUM",
                        "recommendation": "Limit uppercase usage to proper nouns or emphasis only."
                    } if uppercase_count > 3 else {},
                    {
                        "title": "Missing Personalization",
                        "priority": "LOW",
                        "recommendation": "Add dynamic name/company tags for better trust."
                    } if not personalization_tags else {}
                ],
            },
            "positive_aspects": {
                "found": positives,
                "recommendation": "Keep improving engagement while avoiding spam language."
            },
            "best_practices": best_practices,
            "ml_prediction": {
                "spam_probability": spam_prob,
                "label": label,
                "message": f"Email classified as {label.upper()} (spam score: {spam_prob:.2f})"
            },
        }

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return {"error": str(e), "message": "Error analyzing email."}
