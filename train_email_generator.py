import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# === TRAINING CONFIG ===
NUM_EMAILS = 3000         # Train on 3000 diverse emails from comprehensive dataset
MAX_LENGTH = 256          # Increased for longer, more realistic emails
BATCH_SIZE = 4            # Better batch size for training
NUM_EPOCHS = 2            # More epochs for better learning
LEARNING_RATE = 3e-5      # Optimal learning rate for fine-tuning

# === 1. Load dataset ===
CSV_PATH = "data/email_samples.csv"  # 🚀 USE THE COMPREHENSIVE DATASET
assert os.path.exists(CSV_PATH), f"❌ Dataset not found: {CSV_PATH}"

df = pd.read_csv(CSV_PATH).head(NUM_EMAILS)
print(f"📂 Loaded dataset with columns: {list(df.columns)}")
print(f"📊 Dataset shape: {df.shape}")

# Use the comprehensive dataset structure
if "subject" in df.columns and "body" in df.columns:
    df = df[["subject", "body"]].dropna().reset_index(drop=True)
    print(f"✅ Using subject/body columns from comprehensive dataset")
else:
    raise ValueError("❌ Comprehensive dataset missing subject/body columns.")

print(f"✅ Loaded {len(df)} high-quality training samples")

# === 2. Tokenizer setup ===
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def encode_batch(subject, body, max_length=MAX_LENGTH):
    # Format: "Subject: [subject]\n\nBody: [body]"
    text = f"Subject: {subject}\n\nBody: {body}"
    return tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )

# === 3. Model setup ===
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.eos_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Using device: {device}")
model.to(device) # type: ignore

# === 4. Prepare DataLoader ===
print("🔨 Encoding training data...")
encodings = [encode_batch(row.subject, row.body) for _, row in df.iterrows()]
inputs = torch.cat([e["input_ids"] for e in encodings])
masks = torch.cat([e["attention_mask"] for e in encodings])

dataset = TensorDataset(inputs, masks)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"📦 DataLoader prepared: {len(loader)} batches")

# === 5. Optimizer ===
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

# === 6. Training loop ===
model.train()
total_steps = len(loader) * NUM_EPOCHS

print(f"🚀 Training on {len(df)} comprehensive emails for {NUM_EPOCHS} epochs...")
progress_bar = tqdm(total=total_steps, desc="Training Progress")

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    for batch_idx, batch in enumerate(loader):
        input_ids, attention_mask = [b.to(device) for b in batch]
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

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

print(f"🎉 Model trained on comprehensive data saved at: {SAVE_DIR}")
print(f"📈 Final average loss: {avg_epoch_loss:.4f}")

# === 8. Quick test ===
print("\n🧪 Testing the trained model...")
model.eval()

test_prompt = "Subject: Following up on our conversation\n\nBody: Hi there, I wanted to follow up"
input_ids = tokenizer.encode(test_prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=150,
        num_return_sequences=1,
        temperature=0.8,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"📧 Sample generated email:\n{generated_text}")