from fastapi import FastAPI, Query
from services.email_analyzer import analyze_email
from services.email_generator import generate_email, generate_reply
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

app = FastAPI(title="Email Warmup")

# --- CORS setup ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:5173", "http://localhost:5000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Analyzer request model ---
class EmailRequest(BaseModel):
    subject: str = ""
    body: str

# ----------------- ANALYZER -----------------
@app.post("/analyze")
def analyze(email: EmailRequest):
    text = f"{email.subject}\n\n{email.body}"
    result = analyze_email(text, email.subject)
    return {"subject": email.subject, "result": result}

# ----------------- GENERATE EMAIL -----------------
# In your main.py, update the endpoints to handle the enhanced responses:

@app.get("/generate-email")
async def api_generate_email(
    sender_email: str,
    receiver_email: str,
    industry: str = Query("", description="Optional industry name"),
    target_role: str = Query("", description="Optional target role"),
    tone: str = Query("professional", description="Email tone"),
    purpose: str = Query("follow_up", description="Purpose of email")
):
    """Generate a professional email with quality analysis."""
    result = await generate_email(
        sender_email=sender_email,
        receiver_email=receiver_email,
        industry=industry,
        target_role=target_role,
        tone=tone,
        purpose=purpose
    )
    return result

@app.post("/generate-reply")
async def api_generate_reply(
    original_email: str,
    replier_email: str,
    original_sender_email: str,
    tone: str = Query("professional", description="Tone of reply")
):
    """Generate a professional reply with quality analysis."""
    result = await generate_reply(
        original_email=original_email,
        replier_email=replier_email,
        original_sender_email=original_sender_email,
        tone=tone
    )
    return result



if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)