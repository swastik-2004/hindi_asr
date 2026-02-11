import torch
from transformers import WhisperProcessor,WhisperForConditionalGeneration

MODEL_NAME = "openai/whisper-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] Loading Whisper model on {DEVICE}...")

processor=WhisperProcessor.from_pretrained(
    MODEL_NAME,
    language="Hindi",
    task="transcribe"
)

model=WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

print("[INFO] Model loaded successfully!")