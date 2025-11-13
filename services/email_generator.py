import os
import requests
import random
import joblib
import re
import hashlib
from dotenv import load_dotenv
from datetime import datetime

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

# Template patterns with ONLY BR tags - NO \n
TEMPLATE_PATTERNS = [
    {
        "name": "problem_empathy",
        "structure": "{greeting}<br><br>I hope things are going smoothly for you.<br><br>{challenge_context} and I am sure you must be putting in your best efforts to manage it effectively. {value_proposition}<br><br>{call_to_action}<br><br>I hope you have an enjoyable and productive day.<br><br>{closing}<br>{sender_name}"
    },
    {
        "name": "collaboration_discussion", 
        "structure": "{greeting}<br><br>I hope this message finds you well.<br><br>{industry_context} and I believe there could be some interesting opportunities worth exploring together. {specific_insight}<br><br>{invitation}<br><br>Wishing you all the best.<br><br>{closing}<br>{sender_name}"
    },
    {
        "name": "value_exchange",
        "structure": "{greeting}<br><br>Hope you're having a productive week.<br><br>{field_observation} particularly regarding {target_focus}. {perspective_share}<br><br>{connection_request}<br><br>Best regards,<br><br>{sender_name}"
    }
]

# Natural components for dynamic generation
GREETINGS = ["Hello {name},", "Hi {name},", "Dear {name},"]
CLOSINGS = ["Best regards,", "Sincerely,", "Kind regards,", "Warm regards,"]

# ----------------- DYNAMIC TEMPLATE-BASED GENERATION -----------------
def _get_sender_name(email: str) -> str:
    """Extract sender name from email"""
    if email and "@" in email:
        name_part = email.split("@")[0]
        name = re.sub(r'[0-9_\-\.]+', ' ', name_part)
        name = name.strip().title()
        if name and len(name) > 1:
            return name
    return "Team"

def _generate_unique_id(sender_email: str, receiver_email: str, industry: str, purpose: str) -> str:
    """Generate unique ID for variation"""
    base_string = f"{sender_email}{receiver_email}{industry}{purpose}{datetime.now().strftime('%f')}"
    return hashlib.md5(base_string.encode()).hexdigest()[:12]

def _safe_field(value: str, default: str) -> str:
    """Clean up undefined values"""
    if not value or str(value).strip().lower() in ["undefined", "none", "null", ""]:
        return default
    return value.strip()

def _remove_all_newlines(text: str) -> str:
    """Remove ALL newline characters completely"""
    return text.replace('\n', ' ').replace('\r', ' ').strip()

async def _generate_dynamic_component(prompt: str, max_tokens: int = 40) -> str:
    """Generate dynamic content for template components - NO NEWLINES"""
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": 0.8,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False,
            "repetition_penalty": 1.3,
        },
        "options": {
            "wait_for_model": True,
            "use_cache": False
        }
    }
    
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated = result[0]['generated_text'].strip()
                # REMOVE ALL NEWLINES from generated content
                return _remove_all_newlines(generated)
        
        return await _fallback_component(prompt)
        
    except Exception as e:
        print(f"API Error: {e}")
        return await _fallback_component(prompt)

async def _fallback_component(prompt: str) -> str:
    """Fallback components - NO NEWLINES"""
    if "challenge" in prompt.lower():
        return "Managing current industry challenges requires careful attention"
    elif "value" in prompt.lower():
        return "I wanted to share some perspectives that might be helpful"
    elif "insight" in prompt.lower():
        return "There are some interesting developments worth considering"
    elif "invitation" in prompt.lower():
        return "Would you have some time to discuss this further"
    else:
        return "I appreciate the opportunity to connect"

async def _generate_dynamic_email(template_pattern: dict, receiver_name: str, sender_name: str, industry: str, target_role: str, purpose: str, unique_id: str) -> str:
    """Generate dynamic email using template structure with ONLY BR tags"""
    local_random = random.Random(int(unique_id, 16))
    
    # Generate dynamic components
    greeting = local_random.choice(GREETINGS).format(name=receiver_name)
    closing = local_random.choice(CLOSINGS)
    
    # Generate dynamic content for each template slot
    challenge_prompt = f"Write a brief sentence about challenges in {industry} for {target_role}:"
    challenge_context = await _generate_dynamic_component(challenge_prompt, 25)
    
    value_prompt = f"Write a helpful perspective about {industry} and {target_role}:"
    value_proposition = await _generate_dynamic_component(value_prompt, 30)
    
    action_prompt = f"Write a polite invitation to discuss {purpose} in {industry}:"
    call_to_action = await _generate_dynamic_component(action_prompt, 20)
    
    # Use template structure with dynamic content and ONLY BR tags
    email_body = template_pattern["structure"].format(
        greeting=greeting,
        challenge_context=challenge_context,
        value_proposition=value_proposition,
        call_to_action=call_to_action,
        closing=closing,
        sender_name=sender_name,
        industry_context=f"Our discussion about {industry} has been on my mind",
        specific_insight="I've been considering how recent developments might align with your approach",
        invitation="Would you be available for a conversation in the coming week",
        field_observation=f"I've been following developments in {industry}",
        target_focus=target_role,
        perspective_share="I have some thoughts that might be relevant to your work",
        connection_request="Might you have some time to connect and discuss this further"
    )
    
    return email_body

async def _generate_dynamic_subject(industry: str, target_role: str, purpose: str, unique_id: str) -> str:
    """Generate dynamic subject line - NO NEWLINES"""
    subject_prompts = [
        f"Write a professional email subject about {purpose} in {industry}:",
        f"Create a business subject line for discussing {industry} with {target_role}:",
        f"Generate a professional subject for {purpose} regarding {target_role}:"
    ]
    
    local_random = random.Random(int(unique_id, 16))
    prompt = local_random.choice(subject_prompts)
    
    subject = await _generate_dynamic_component(prompt, 15)
    subject = subject.split('.')[0].strip()  # Take first sentence only
    
    # Clean up subject - REMOVE ALL NEWLINES
    subject = _remove_all_newlines(subject)
    subject = re.sub(r'[^a-zA-Z0-9\s\-]', '', subject)
    if not subject or len(subject) < 5:
        subject = f"Following up on {industry} discussion"
    
    return subject

def _create_html_email(body_content: str) -> str:
    """Create HTML email with proper structure"""
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
        }}
    </style>
</head>
<body>
    {body_content}
</body>
</html>"""

# ----------------- DYNAMIC TEMPLATE-BASED GENERATOR -----------------
async def generate_email(
    sender_email: str,
    receiver_email: str,
    industry: str,
    target_role: str, 
    tone: str = "professional",
    purpose: str = "collaboration"
):
    """Generate dynamic emails using template patterns as guidance"""
    unique_id = _generate_unique_id(sender_email, receiver_email, industry, purpose)
    
    sender_email = _safe_field(sender_email, "team@example.com")
    receiver_email = _safe_field(receiver_email, "client@example.com")
    industry = _safe_field(industry, "your field")
    target_role = _safe_field(target_role, "your role")
    purpose = _safe_field(purpose, "collaboration")

    # Get names
    receiver_name = receiver_email.split("@")[0].capitalize()
    sender_name = _get_sender_name(sender_email)
    
    # Select template pattern
    local_random = random.Random(int(unique_id, 16))
    template_pattern = local_random.choice(TEMPLATE_PATTERNS)
    
    # Generate dynamic content using template structure
    subject = await _generate_dynamic_subject(industry, target_role, purpose, unique_id)
    body_with_br = await _generate_dynamic_email(template_pattern, receiver_name, sender_name, industry, target_role, purpose, unique_id)
    
    # Create final HTML email
    body_html = _create_html_email(body_with_br)
    
    return {
        "subject": subject, 
        "body_html": body_html
    }

# ----------------- DYNAMIC REPLY GENERATOR -----------------
async def generate_reply(
    original_email: str,
    replier_email: str, 
    original_sender_email: str,
    tone: str = "professional"
):
    """Generate dynamic replies with ONLY BR tags"""
    unique_id = _generate_unique_id(replier_email, original_sender_email, "reply", tone)
    
    replier_email = _safe_field(replier_email, "our-team@example.com")
    original_sender_email = _safe_field(original_sender_email, "client@example.com")
    original_email = _safe_field(original_email, "your previous email")

    # Get names
    receiver_name = original_sender_email.split("@")[0].capitalize()
    sender_name = _get_sender_name(replier_email)
    
    # Generate dynamic reply
    reply_prompt = f"Write a professional reply acknowledging a message and suggesting follow-up:"
    reply_content = await _generate_dynamic_component(reply_prompt, 50)
    
    greeting = random.choice(GREETINGS).format(name=receiver_name)
    closing = random.choice(CLOSINGS)
    
    # Use ONLY BR tags - NO NEWLINES
    reply_with_br = f"{greeting}<br><br>{reply_content}<br><br>{closing}<br>{sender_name}"
    
    reply_html = _create_html_email(reply_with_br)
    subject = "Re: Your message"

    return {
        "subject": subject, 
        "body_html": reply_html
    }