import joblib # type: ignore
import os
import re
import numpy as np # type: ignore
from typing import Dict, List, Any

# Load the saved model and vectorizer
MODEL_PATH = os.path.join("models", "spam_model.pkl")
VECTORIZER_PATH = os.path.join("models", "vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Comprehensive spam trigger words
SPAM_TRIGGER_WORDS = [
    "promotion", "free", "winner", "urgent", "discount", "buy now", 
    "limited time", "act now", "click here", "offer", "cash", "prize",
    "risk-free", "guaranteed", "special promotion", "limited offer"
]

NEGATIVE_TONE_WORDS = [
    "problem", "issue", "failed", "wrong", "error", "broken", 
    "complaint", "trouble", "difficulty", "can't", "won't",
    "sorry", "apologize", "mistake", "fix", "repair", "spam folder"
]

def analyze_email(text: str, subject: str = "") -> Dict[str, Any]:
    """Analyze email content with comprehensive metrics matching the UI design"""
    
    # Preserve original text for display, but clean for analysis
    original_text = text
    cleaned_text = clean_text_for_analysis(text)
    
    # Basic text analysis WITH PROPER QUANTIZATION
    words = cleaned_text.split()
    word_count = len(words)
    
    # Sentence counting with proper quantization
    sentences = [s.strip() for s in re.split(r'[.!?]+', original_text) if s.strip()]
    sentence_count = len(sentences)
    
    # Paragraph counting from line breaks (preserve structure)
    paragraphs = [p.strip() for p in original_text.split('\n') if p.strip()]
    paragraph_count = len(paragraphs)
    
    line_count = len(original_text.split('\n'))
    question_count = len(re.findall(r'\?', original_text))
    
    # Link detection
    link_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    link_count = len(re.findall(link_pattern, original_text))
    
    # Uppercase words (all caps)
    uppercase_count = len(re.findall(r'\b[A-Z]{2,}\b', original_text))
    
    # Personalization tags detection (handles underscores and spaces)
    personal_tags = re.findall(r'\[[^\]]+\]', original_text)
    
    # Content analysis
    text_lower = original_text.lower()
    spam_words_found = [word for word in SPAM_TRIGGER_WORDS if word in text_lower]
    negative_words_found = [word for word in NEGATIVE_TONE_WORDS if word in text_lower]
    
    # Readability calculation with proper quantization
    avg_sentence_length = word_count / max(sentence_count, 1)
    readability_score = max(0, min(100, 100 - (avg_sentence_length - 10) * 2))
    
    # Read time estimation
    read_time_min = max(1, round(word_count / 200))
    
    # AI/ML spam prediction
    X = vectorizer.transform([cleaned_text])
    spam_probability = float(model.predict_proba(X)[0][1])
    
    # Calculate health score USING THE ACCURATE FORMULA
    health_score = calculate_accurate_health_score(
        spam_probability, 
        len(spam_words_found),
        uppercase_count
    )
    
    # FIXED: Calculate subject length correctly
    subject_length = len(subject) if subject else len(original_text.split('\n')[0].strip()) if original_text.split('\n') else 0
    
    # Generate analysis results matching the UI structure
    return {
        "template_analytics": {
            "email_health_score": health_score,
            "status": get_health_status(health_score),
            "metrics": {
                "subject": {"value": subject_length, "status": get_subject_status(subject_length), "label": "SUBJECT"},
                "words": {"value": word_count, "status": "Optimal" if 50 <= word_count <= 150 else "Review", "label": "WORDS"},
                "sentences": {"value": sentence_count, "status": "Optimal" if sentence_count >= 3 else "Review", "label": "SENTENCES"},
                "paragraphs": {"value": paragraph_count, "status": "Optimal" if paragraph_count >= 2 else "Review", "label": "PARAGRAPHS"},
                "lines": {"value": line_count, "status": "Optimal", "label": "LINES"},
                "read_time": {"value": f"{read_time_min} min", "status": "Optimal", "label": "READ TIME"},
                "links": {"value": link_count, "status": "Optimal", "label": "LINKS"},
                "questions": {"value": question_count, "status": "Optimal", "label": "QUESTIONS"},
                "spam_words": {"value": len(spam_words_found), "status": "Review" if spam_words_found else "Optimal", "label": "SPAM WORDS"},
                "personal_tags": {"value": len(personal_tags), "status": "Review" if personal_tags else "Optimal", "label": "PERSONAL TAGS"},
                "uppercase": {"value": uppercase_count, "status": "Optimal" if uppercase_count <= 3 else "Review", "label": "UPPERCASE"},
                "readability": {"value": int(readability_score), "status": "Optimal" if readability_score >= 60 else "Review", "label": "READABILITY"}
            }
        },
        "detailed_analysis": {
            "critical_issues": get_critical_issues(spam_probability, spam_words_found),
            "warnings": get_warnings(negative_words_found, personal_tags),
            "suggestions": get_suggestions(original_text, word_count, question_count, paragraph_count, subject_length),
            "passed_checks": get_passed_checks(original_text, word_count, uppercase_count, paragraph_count, link_count, subject_length)
        },
        "warmup_strategies": get_warmup_strategies(),
        "positive_aspects": get_positive_aspects(original_text, personal_tags, sentence_count, word_count, paragraph_count, subject_length)
    }

def get_subject_status(subject_length: int) -> str:
    """Get subject line status based on length"""
    if 20 <= subject_length <= 60:
        return "Optimal"
    elif subject_length < 20:
        return "Too Short"
    else:
        return "Too Long"

def clean_text_for_analysis(text: str) -> str:
    """Clean text for ML analysis while preserving structure for metrics"""
    # Replace multiple spaces with single space, but preserve line breaks for paragraph counting
    text = re.sub(r'[ \t]+', ' ', text)  # Only spaces and tabs, not newlines
    return text.strip()

def calculate_accurate_health_score(spam_prob: float, spam_word_count: int, uppercase_count: int) -> int:
    """Calculate health score using the accurate formula from the simple version"""
    base_score = 100 - int(spam_prob * 50) - spam_word_count * 5 - uppercase_count
    return max(min(base_score, 100), 0)

def get_health_status(score: int) -> str:
    """Get health status based on score"""
    if score >= 80:
        return "Excellent - Your template is ready for sending!"
    elif score >= 60:
        return "Good - Minor improvements recommended"
    else:
        return "Needs Work - Review content carefully"

def get_critical_issues(spam_prob: float, spam_words: List[str]) -> List[Dict]:
    """Get critical issues that need urgent attention"""
    issues = []
    
    if spam_prob > 0.5:
        issues.append({
            "priority": "HIGH PRIORITY",
            "category": "Spam Risk",
            "found": f"High spam probability ({spam_prob:.1%})",
            "recommendation": "Content may be flagged as spam. Reduce promotional language and avoid spam triggers."
        })
    
    if spam_words:
        issues.append({
            "priority": "HIGH PRIORITY", 
            "category": "Spam Trigger Words",
            "found": ", ".join(spam_words),
            "recommendation": f"Replace {len(spam_words)} spam-triggering words with professional alternatives"
        })
    
    return issues

def get_warnings(negative_words: List[str], personal_tags: List[str]) -> List[Dict]:
    """Get warnings and recommended improvements"""
    warnings = []
    
    if negative_words:
        warnings.append({
            "priority": "LOW PRIORITY",
            "category": "Negative Tone Detected", 
            "found": "Email content may sound negative or problematic",
            "recommendation": "Rephrase to maintain a positive, solution-oriented tone while addressing issues"
        })
    
    if personal_tags:
        warnings.append({
            "priority": "MEDIUM PRIORITY",
            "category": "Unfilled Personalization Tags",
            "found": f"{len(personal_tags)} personalization tags found",
            "recommendation": "Replace all [placeholder] tags with actual recipient data before sending to improve engagement"
        })
    
    return warnings

def get_suggestions(text: str, word_count: int, question_count: int, paragraph_count: int, subject_length: int) -> List[Dict]:
    """Get optional enhancement suggestions"""
    suggestions = []
    
    if word_count < 50:
        suggestions.append({
            "priority": "SUGGESTION",
            "category": "Content Length",
            "found": "Email content is quite short",
            "recommendation": "Consider adding more valuable content to improve engagement"
        })
    
    if question_count < 1:
        suggestions.append({
            "priority": "SUGGESTION", 
            "category": "Engagement",
            "found": "No engaging questions detected",
            "recommendation": "Add questions to encourage recipient interaction"
        })
    
    if paragraph_count < 3:
        suggestions.append({
            "priority": "SUGGESTION",
            "category": "Formatting",
            "found": "Could use better paragraph structure",
            "recommendation": "Add more line breaks to improve readability"
        })
    
    if subject_length > 60:
        suggestions.append({
            "priority": "SUGGESTION",
            "category": "Subject Line",
            "found": "Subject line is too long",
            "recommendation": "Shorten subject line to under 60 characters for better open rates"
        })
    elif subject_length < 20:
        suggestions.append({
            "priority": "SUGGESTION",
            "category": "Subject Line", 
            "found": "Subject line is too short",
            "recommendation": "Make subject line more descriptive (20-60 characters recommended)"
        })
    
    return suggestions

def get_passed_checks(text: str, word_count: int, uppercase_count: int, paragraph_count: int, link_count: int, subject_length: int) -> List[str]:
    """Get list of checks that passed"""
    passed = []
    
    if 30 <= word_count <= 200:
        passed.append("Optimal word count")
    
    if uppercase_count <= 3:
        passed.append("Good uppercase usage")
    
    if paragraph_count >= 2:
        passed.append("Good paragraph structure")
    
    if len(re.findall(r'\?', text)) > 0:
        passed.append("Engaging questions present")
    
    if link_count == 0:
        passed.append("No suspicious links")
    
    if 20 <= subject_length <= 60:
        passed.append("Good subject line length")
    
    return passed if passed else ["Basic structure OK"]

def get_positive_aspects(text: str, personal_tags: List[str], sentence_count: int, word_count: int, paragraph_count: int, subject_length: int) -> Dict:
    """Get positive aspects found in the email"""
    positive_feedback = []
    
    if personal_tags:
        positive_feedback.append("personalization tags")
    
    if len(re.findall(r'\?', text)) > 0:
        positive_feedback.append("engaging questions")
    
    if sentence_count >= 3:
        positive_feedback.append("good sentence structure")
    
    if paragraph_count >= 3:
        positive_feedback.append("excellent paragraph formatting")
    elif paragraph_count >= 2:
        positive_feedback.append("good paragraph structure")
    
    if 50 <= word_count <= 150:
        positive_feedback.append("optimal content length")
    
    if 20 <= subject_length <= 60:
        positive_feedback.append("optimal subject length")
    
    return {
        "priority": "NONE PRIORITY", 
        "category": "Positive Feedback",
        "found": f"Well done on: {', '.join(positive_feedback)}" if positive_feedback else "No specific positive aspects identified",
        "recommendation": "Continue maintaining these good practices in your email templates" if positive_feedback else "Focus on adding personalization and engagement elements"
    }

def get_warmup_strategies() -> List[Dict]:
    """Get email warmup strategies and best practices"""
    return [
        {
            "title": "Gradual Warmup Process",
            "description": "Start with 5-10 emails daily, gradually increasing volume over 4-8 weeks. Monitor engagement metrics closely and adjust based on performance."
        },
        {
            "title": "Authentication Setup", 
            "description": "Configure SPF, DKIM, and DMARC records properly. This builds trust with email providers and improves deliverability rates significantly."
        },
        {
            "title": "List Hygiene & Engagement",
            "description": "Regularly clean your email list. Remove inactive subscribers, validate email addresses, and monitor open/click rates to maintain list quality."
        },
        {
            "title": "Content Optimization",
            "description": "Personalize content, avoid spam triggers, maintain a clean professional tone, and ensure mobile responsiveness for better engagement."
        }
    ]