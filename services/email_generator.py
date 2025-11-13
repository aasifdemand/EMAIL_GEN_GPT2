import os
import requests
import random
import joblib
import re
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/Aasif1234/email_gpt2"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# Load spam detection model
MODEL_PATH = os.path.join("models", "spam_model.pkl")
VECTORIZER_PATH = os.path.join("models", "vectorizer.pkl")

try:
    spam_model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✅ Spam detection model loaded successfully")
except Exception as e:
    print(f"❌ Error loading spam model: {e}")
    spam_model = None
    vectorizer = None

# Spam trigger words for prevention
SPAM_TRIGGER_WORDS = {
    "free": ["complimentary", "no cost", "at no charge"],
    "winner": ["selected", "chosen", "eligible"],
    "urgent": ["important", "priority", "time-sensitive"],
    "buy now": ["learn more", "explore", "discover"],
    "discount": ["special offer", "opportunity", "value"],
    "cash": ["reward", "benefit", "advantage"],
    "prize": ["recognition", "achievement", "honor"],
    "guaranteed": ["assured", "reliable", "dependable"],
    "act now": ["get started", "begin today", "take action"],
    "limited time": ["available now", "current opportunity", "present chance"]
}

# ----------------- HELPERS -----------------
def _safe_field(value: str, default: str):
    """Clean up undefined, none, or empty values."""
    if not value or str(value).strip().lower() in ["undefined", "none", "null", ""]:
        return default
    return value.strip()

def _random_tagline():
    return random.choice([
        "Business Development Team",
        "Growth Specialist", 
        "Client Success Manager",
        "Marketing Executive",
        "Strategic Outreach Team",
    ])

def _random_variation():
    """Introduce small random style variation for diversity."""
    return random.choice([
        "short and professional",
        "concise yet friendly", 
        "warm and engaging",
        "polished and respectful",
        "approachable and courteous",
    ])

def _is_spammy_content(text: str) -> bool:
    """Check if content is spammy using the ML model"""
    if spam_model is None or vectorizer is None:
        # Fallback: basic spam word check
        text_lower = text.lower()
        spam_words_found = [word for word in SPAM_TRIGGER_WORDS.keys() if word in text_lower]
        return len(spam_words_found) > 2
    
    try:
        # Use ML model for spam detection
        X = vectorizer.transform([text])
        spam_probability = spam_model.predict_proba(X)[0][1]
        return spam_probability > 0.6  # Threshold for spam detection
    except:
        return False

def _create_anti_spam_prompt(base_prompt: str) -> str:
    """Add anti-spam instructions to the prompt"""
    anti_spam_instructions = (
        "\n\nIMPORTANT: Avoid spam trigger words like 'free', 'winner', 'urgent', 'buy now', 'discount', "
        "'cash', 'prize', 'guaranteed', 'act now', 'limited time'. Use professional business language "
        "that sounds natural and trustworthy. Focus on value and collaboration, not sales pressure."
    )
    return base_prompt + anti_spam_instructions

def _enhance_content(text: str) -> str:
    """Replace any spammy words that might have slipped through"""
    enhanced_text = text
    for spam_word, alternatives in SPAM_TRIGGER_WORDS.items():
        if spam_word in enhanced_text.lower():
            replacement = random.choice(alternatives)
            enhanced_text = re.sub(
                re.escape(spam_word), 
                replacement, 
                enhanced_text, 
                flags=re.IGNORECASE
            )
    return enhanced_text

def _improve_tone(text: str) -> str:
    """Improve overall email tone"""
    improvements = [
        (r'\b(immediately|right away)\b', 'promptly'),
        (r'\b(cheap|low cost)\b', 'cost-effective'),
        (r'\b(huge|massive)\b', 'significant'),
        (r'\b(amazing|incredible)\b', 'valuable'),
        (r'\b(never\s+before|unique)\b', 'distinctive'),
    ]
    
    improved_text = text
    for pattern, replacement in improvements:
        improved_text = re.sub(pattern, replacement, improved_text, flags=re.IGNORECASE)
    
    return improved_text

async def _generate_text(prompt: str, max_new_tokens: int = 85) -> str:
    """Generate text using Hugging Face Inference API with anti-spam prompt"""
    # Add anti-spam instructions to prevent spammy generation
    safe_prompt = _create_anti_spam_prompt(prompt)
    
    payload = {
        "inputs": safe_prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.7,  # Lower temperature for more focused output
            "top_p": 0.85,
            "do_sample": True,
            "return_full_text": False,
            "repetition_penalty": 1.2
        },
        "options": {
            "wait_for_model": True,
            "use_cache": True
        }
    }
    
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0]['generated_text'].strip()
                # Quick enhancement to catch any remaining spammy words
                return _enhance_content(generated_text)
        
        return await _fallback_generation(prompt)
        
    except Exception as e:
        print(f"API Error: {e}")
        return await _fallback_generation(prompt)

async def _fallback_generation(prompt: str) -> str:
    """Smart fallback when model is unavailable - guaranteed spam-free"""
    if "subject" in prompt.lower():
        subjects = [
            "Following up on our conversation",
            "Quick update regarding your inquiry",
            "Opportunity for collaboration", 
            "Connecting as discussed",
            "Follow-up on recent discussion"
        ]
        return random.choice(subjects)
    else:
        professional_responses = [
            "I hope this message finds you well. I wanted to follow up on our recent conversation and explore potential opportunities for collaboration.",
            "Thank you for your time recently. I've been considering our discussion and wanted to share some thoughts on how we might work together.",
            "I appreciate the opportunity to connect with you. Following our conversation, I wanted to propose some next steps for our potential collaboration.",
            "It was great speaking with you. I've been reflecting on our discussion and believe there are meaningful opportunities we could explore together."
        ]
        return random.choice(professional_responses)

async def _generate_subject(sender_email, receiver_email, industry, purpose):
    """Generate realistic short subject line with anti-spam protection"""
    sender_prefix = sender_email.split("@")[0].capitalize()
    receiver_prefix = receiver_email.split("@")[0].capitalize()

    examples = [
        f"{receiver_prefix}, let's discuss {purpose}",
        f"Quick idea for {industry}",
        f"Exploring {purpose} in {industry}",
        f"{sender_prefix} here – opportunity in {industry}",
        f"Connecting about {purpose}",
    ]

    base_prompt = (
        f"Write a short and natural business email subject (under 8 words).\n"
        f"From: {sender_email}\nTo: {receiver_email}\n"
        f"Industry: {industry}\nPurpose: {purpose}\n"
        f"Examples:\n" + "\n".join(f"- {s}" for s in examples) + "\nSubject:"
    )

    subject = await _generate_text(base_prompt, max_new_tokens=12)
    subject = subject.split("\n")[0].strip().rstrip(".!?")

    if not subject or len(subject) < 5:
        subject = random.choice(examples)

    subject += f" – {random.choice(['Note', 'Intro', 'Insight'])}"
    
    # Final spam check and enhancement
    return _enhance_content(subject)

# ----------------- EMAIL GENERATOR -----------------
async def generate_email(
    sender_email: str,
    receiver_email: str,
    industry: str,
    target_role: str, 
    tone: str = "professional",
    purpose: str = "collaboration"
):
    """Generate email with built-in spam prevention"""
    sender_email = _safe_field(sender_email, "team@example.com")
    receiver_email = _safe_field(receiver_email, "client@example.com")
    industry = _safe_field(industry, "your field")
    target_role = _safe_field(target_role, "your team")
    tone = _safe_field(tone, "professional")
    purpose = _safe_field(purpose, "collaboration")

    variation = _random_variation()
    
    # Generate subject with anti-spam protection
    subject = await _generate_subject(sender_email, receiver_email, industry, purpose)

    # Create prompt with anti-spam instructions built-in
    prompt = (
        f"Write a {variation} {tone} outreach email (max 5 sentences) for {purpose}.\n"
        f"Use the format:\n"
        f"1. Greeting\n2. One-line purpose intro\n3. 2–3 sentence main paragraph about benefits or ideas\n"
        f"4. Call to action line\n5. Closing.\n\n"
        f"From: {sender_email}\nTo: {receiver_email}\n"
        f"Industry: {industry}\nTarget Role: {target_role}\n\nEmail:\n"
    )

    # Generate body with anti-spam protection
    body = await _generate_text(prompt, max_new_tokens=80)
    body = " ".join(line.strip() for line in body.splitlines() if line.strip())
    body = body.replace("undefined", "").replace("  ", " ")

    if not body.lower().startswith(("hi", "hello")):
        name = receiver_email.split("@")[0].capitalize()
        body = f"Hi {name}, {body}"

    # Final tone improvement
    body = _improve_tone(body)

    signature = f"<br><br>Best regards,<br>{sender_email}<br>{_random_tagline()}"

    body_html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>"
        "body {font-family: Arial, sans-serif; line-height:1.6; color:#333; "
        "max-width:600px; margin:0 auto; padding:20px;}"
        "p {margin-bottom:15px; text-align:justify;}"
        ".signature {margin-top:20px; font-weight:bold;}"
        "</style></head><body>"
        f"<p>{body}</p>"
        f"<div class='signature'>{signature}</div>"
        "</body></html>"
    )
    
    return {
        "subject": subject, 
        "body_html": body_html
    }

# ----------------- REPLY GENERATOR -----------------
async def generate_reply(
    original_email: str,
    replier_email: str, 
    original_sender_email: str,
    tone: str = "professional"
):
    """Generate reply with built-in spam prevention"""
    replier_email = _safe_field(replier_email, "our-team@example.com")
    original_sender_email = _safe_field(original_sender_email, "client@example.com")
    original_email = _safe_field(original_email, "your previous email")

    prompt = (
        f"Write a short, {tone} reply (3–4 sentences) to this email.\n"
        f"Be polite, acknowledge the message, show appreciation, and suggest next steps.\n\n"
        f"Original email:\n{original_email}\n\nReply:\n"
    )

    reply_text = await _generate_text(prompt, max_new_tokens=70)
    reply_text = " ".join(line.strip() for line in reply_text.splitlines() if line.strip())
    reply_text = reply_text.replace("undefined", "")

    signature = f"<br><br>Best regards,<br>{replier_email}<br>{_random_tagline()}"

    reply_html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>"
        "body {font-family: Arial, sans-serif; line-height:1.6; color:#333; "
        "max-width:600px; margin:0 auto; padding:20px;}"
        "p {margin-bottom:15px; text-align:justify;}"
        ".signature {margin-top:20px; font-weight:bold;}"
        "</style></head><body>"
        f"<p>{reply_text}</p>"
        f"<div class='signature'>{signature}</div>"
        "</body></html>"
    )

    subject = f"Re: {original_sender_email.split('@')[0].capitalize()} – Follow-up"

    return {
        "subject": subject, 
        "body_html": reply_html
    }