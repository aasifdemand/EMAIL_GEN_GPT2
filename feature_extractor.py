import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


def flesch_reading_ease(text: str) -> float:
    """
    Compute Flesch Reading Ease score manually.
    Formula: 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
    Higher score = easier to read.
    """
    text = text.lower()
    words = re.findall(r'\w+', text)
    if not words:
        return 0.0

    sentences = max(1, text.count('.') + text.count('!') + text.count('?'))
    vowels = "aeiouy"
    syllables = 0
    for word in words:
        word_syllables = 0
        word = word.lower().strip()
        if word:
            if word[0] in vowels:
                word_syllables += 1
            for i in range(1, len(word)):
                if word[i] in vowels and word[i - 1] not in vowels:
                    word_syllables += 1
            if word.endswith("e"):
                word_syllables = max(1, word_syllables - 1)
        syllables += max(1, word_syllables)

    word_count = len(words)
    return round(206.835 - 1.015 * (word_count / sentences) - 84.6 * (syllables / word_count), 2)


def extract_email_features(text: str) -> dict:
    """Extract lexical, structural, and readability-based features from email text."""
    soup = BeautifulSoup(text, "html.parser")
    visible_text = soup.get_text(separator=" ", strip=True)

    features = {}
    features["char_count"] = len(visible_text)
    features["word_count"] = len(visible_text.split())
    features["sentence_count"] = visible_text.count('.') + visible_text.count('!') + visible_text.count('?')
    features["avg_word_len"] = np.mean([len(w) for w in visible_text.split()]) if visible_text.split() else 0.0
    features["avg_sentence_len"] = features["word_count"] / features["sentence_count"] if features["sentence_count"] > 0 else 0.0

    # Symbolic / stylistic features
    features["exclamation_count"] = visible_text.count('!')
    features["question_count"] = visible_text.count('?')
    features["uppercase_ratio"] = sum(1 for c in visible_text if c.isupper()) / max(1, len(visible_text))
    features["num_ratio"] = sum(1 for c in visible_text if c.isdigit()) / max(1, len(visible_text))
    features["special_char_ratio"] = len(re.findall(r'[^a-zA-Z0-9\s]', visible_text)) / max(1, len(visible_text))

    # Structural features
    features["link_count"] = len(re.findall(r'http[s]?://', text))
    features["html_tag_count"] = len(soup.find_all())
    features["email_mention_count"] = len(re.findall(r'[\w\.-]+@[\w\.-]+', text))
    features["currency_symbol_count"] = len(re.findall(r'[$€£₹]', visible_text))

    # Keyword and semantic features
    spammy_words = ["free", "win", "offer", "buy", "click", "subscribe", "bonus", "cash", "urgent", "limited", "prize"]
    text_lower = visible_text.lower()
    features["spammy_word_density"] = sum(text_lower.count(w) for w in spammy_words) / max(1, features["word_count"])

    # Readability (safe custom function)
    features["flesch_reading"] = flesch_reading_ease(visible_text)

    # Ratios
    features["punctuation_density"] = (features["exclamation_count"] + features["question_count"]) / max(1, features["char_count"])
    features["avg_word_freq"] = np.log1p(features["word_count"]) / max(1, features["sentence_count"])

    # HTML presence
    features["contains_html"] = 1 if len(soup.find_all()) > 0 else 0
    features["has_title_tag"] = 1 if soup.title else 0
    features["has_links"] = 1 if features["link_count"] > 0 else 0

    return features


def batch_extract(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Batch process DataFrame text column for feature extraction."""
    all_features = []
    total = len(df)

    for idx, text in enumerate(df[text_col].fillna("").astype(str)):
        if (idx + 1) % 200 == 0:
            print(f"🔍 Processed {idx + 1}/{total} samples...")
        feats = extract_email_features(text)
        all_features.append(feats)

    feature_df = pd.DataFrame(all_features)
    return feature_df.fillna(0.0)
