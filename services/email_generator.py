import os
import random
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dotenv import load_dotenv


load_dotenv()
hf_token=os.getenv("HUGGINGFACE_TOKEN")

try:
    MODEL_NAME = "Aasif1234/email_gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME,token=hf_token)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME,token=hf_token)
    print("✅ Successfully loaded fine-tuned model from Hugging Face!")
except Exception as e:
    MODEL_NAME = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

model.eval()


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


async def _generate_text(prompt: str, max_new_tokens: int = 85) -> str:
    """Generate text using fine-tuned GPT-2."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.9,
            top_p=0.92,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return text.strip()


async def _generate_subject(sender_email, receiver_email, industry, purpose):
    """Generate realistic short subject line."""
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
    return subject


# ----------------- EMAIL GENERATOR -----------------
async def generate_email(
    sender_email: str,
    receiver_email: str,
    industry: str,
    target_role: str,
    tone: str = "professional",
    purpose: str = "collaboration"
):
    sender_email = _safe_field(sender_email, "team@example.com")
    receiver_email = _safe_field(receiver_email, "client@example.com")
    industry = _safe_field(industry, "your field")
    target_role = _safe_field(target_role, "your team")
    tone = _safe_field(tone, "professional")
    purpose = _safe_field(purpose, "collaboration")

    variation = _random_variation()
    subject = await _generate_subject(sender_email, receiver_email, industry, purpose)

    prompt = (
        f"Write a {variation} {tone} outreach email (max 5 sentences) for {purpose}.\n"
        f"Use the format:\n"
        f"1. Greeting\n2. One-line purpose intro\n3. 2–3 sentence main paragraph about benefits or ideas\n"
        f"4. Call to action line\n5. Closing.\n\n"
        f"From: {sender_email}\nTo: {receiver_email}\n"
        f"Industry: {industry}\nTarget Role: {target_role}\n\nEmail:\n"
    )

    body = await _generate_text(prompt, max_new_tokens=80)
    body = " ".join(line.strip() for line in body.splitlines() if line.strip())
    body = body.replace("undefined", "").replace("  ", " ")

    if not body.lower().startswith(("hi", "hello")):
        name = receiver_email.split("@")[0].capitalize()
        body = f"Hi {name}, {body}"

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

    return {"subject": subject, "body_html": body_html}


# ----------------- REPLY GENERATOR -----------------
async def generate_reply(
    original_email: str,
    replier_email: str,
    original_sender_email: str,
    tone: str = "professional"
):
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
    return {"subject": subject, "body_html": reply_html}
