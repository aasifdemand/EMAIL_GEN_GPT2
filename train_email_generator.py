import os
import pandas as pd
import torch
import joblib
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import re

# === TRAINING CONFIG ===
NUM_EMAILS = 100
MAX_LENGTH = 256
BATCH_SIZE = 4
NUM_EPOCHS = 3
LEARNING_RATE = 3e-5

# === 1. EMAIL CLEANING FUNCTIONS ===
def clean_spammy_patterns(text):
    """Restructure spammy email patterns into professional ones"""
    
    # Remove made-up statistics
    text = re.sub(r'\b\d+% (?:better|increase|reduce|improve)\b', 'improved', text)
    text = re.sub(r'reduce by \d+%', 'improve', text)
    text = re.sub(r'\b\d+% better results\b', 'better outcomes', text)
    
    # Remove generic personalization
    text = re.sub(r'Your (?:team\'s|recent) .* caught my attention', 'I was impressed by your work', text)
    text = re.sub(r'Your .* is quite innovative', 'Your approach is interesting', text)
    text = re.sub(r'Saw your work in', 'I came across your work in', text)
    
    # Remove salesy phrases
    text = re.sub(r'time-consuming and frustrating', 'challenging', text)
    text = re.sub(r'personalized demo', 'discussion', text)
    text = re.sub(r'schedule.*minutes', 'find some time', text)
    text = re.sub(r'quick question about', 'wanted to discuss', text)
    
    # Remove template markers and references
    text = re.sub(r'Ref: [a-f0-9]+', '', text)
    text = re.sub(r'TrueInc|ClearSystems', '[Company]', text)
    
    # Fix subject lines
    text = re.sub(r'^Helping.*companies with', 'Exploring opportunities in', text)
    text = re.sub(r'^Way to improve', 'Opportunity to enhance', text)
    text = re.sub(r'^An approach that\'s working', 'A strategy that has worked', text)
    text = re.sub(r'^Simplifying.*for', 'Streamlining processes for', text)
    
    return text.strip()

def restructure_email(subject, body):
    """Restructure the entire email to be non-spammy"""
    
    # Clean subject
    clean_subject = clean_spammy_patterns(subject)
    
    # Clean body
    clean_body = clean_spammy_patterns(body)
    
    # Restructure the email format
    lines = clean_body.split('\n')
    restructured_lines = []
    
    for line in lines:
        # Remove repetitive signatures
        if any(marker in line for marker in ['Cheers,', 'Talk soon,', 'Best,', 'Ref:']):
            continue
        
        # Clean up the content
        clean_line = line.strip()
        if clean_line:
            restructured_lines.append(clean_line)
    
    # Rebuild the email with better structure
    if restructured_lines:
        # Ensure it starts with a proper greeting
        if not restructured_lines[0].lower().startswith(('hi', 'hello', 'dear')):
            restructured_lines.insert(0, "Hello,")
        
        # Ensure it has a proper closing
        if not any(line.lower().startswith(('best', 'regards', 'sincerely')) for line in restructured_lines[-2:]):
            restructured_lines.append("\nBest regards,\n[Your Name]")
    
    clean_body = '\n'.join(restructured_lines)
    
    return clean_subject, clean_body

# === 2. Load and RESTRUCTURE dataset ===
CSV_PATH = "data/email_samples.csv"
assert os.path.exists(CSV_PATH), f"❌ Dataset not found: {CSV_PATH}"

df = pd.read_csv(CSV_PATH).head(NUM_EMAILS)
print(f"📂 Loaded dataset with {len(df)} records")

print("🔄 Restructuring spammy emails into professional ones...")
restructured_samples = []

for _, row in df.iterrows():
    try:
        original_subject = str(row.get('subject', ''))
        original_body = str(row.get('body', ''))
        
        # Restructure the email to remove spammy patterns
        clean_subject, clean_body = restructure_email(original_subject, original_body)
        
        restructured_samples.append((clean_subject, clean_body))
        
        # Show some examples of transformation
        if len(restructured_samples) <= 3:
            print(f"📧 Example {len(restructured_samples)}:")
            print(f"   BEFORE: {original_subject[:60]}...")
            print(f"   AFTER:  {clean_subject[:60]}...")
            print()
            
    except Exception as e:
        print(f"⚠️ Error restructuring row: {e}")
        # Use original if restructuring fails
        restructured_samples.append((str(row.get('subject', '')), str(row.get('body', ''))))

print(f"✅ Restructured {len(restructured_samples)} emails")

# === 3. Tokenizer setup ===
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def encode_batch(subject, body, max_length=MAX_LENGTH):
    text = f"Subject: {subject}\n\nBody: {body}"
    return tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )

# === 4. Model setup ===
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.eos_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Using device: {device}")
model.to(device) # type: ignore

# === 5. Prepare DataLoader ===
print("🔨 Encoding restructured training data...")
encodings = [encode_batch(subject, body) for subject, body in restructured_samples]
inputs = torch.cat([e["input_ids"] for e in encodings])
masks = torch.cat([e["attention_mask"] for e in encodings])

dataset = TensorDataset(inputs, masks)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"📦 DataLoader prepared: {len(loader)} batches")

# === 6. Training ===
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

model.train()
total_steps = len(loader) * NUM_EPOCHS

print(f"🚀 Training on {len(restructured_samples)} RESTRUCTURED emails for {NUM_EPOCHS} epochs...")
progress_bar = tqdm(total=total_steps, desc="Training Progress")

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    for batch_idx, batch in enumerate(loader):
        input_ids, attention_mask = [b.to(device) for b in batch]
        labels = input_ids.clone()

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        
        progress_bar.update(1)
        progress_bar.set_postfix({
            'epoch': epoch + 1,
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{(epoch_loss / (batch_idx + 1)):.4f}'
        })

    avg_epoch_loss = epoch_loss / len(loader)
    print(f"📊 Epoch {epoch + 1} completed - Average Loss: {avg_epoch_loss:.4f}")

progress_bar.close()

# === 7. Save model ===
SAVE_DIR = os.path.join("models", "comprehensive_email_generator")
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"🎉 Model trained on RESTRUCTURED data saved at: {SAVE_DIR}")
print(f"📈 Final average loss: {avg_epoch_loss:.4f}")

