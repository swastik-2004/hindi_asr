import torch
from datasets import load_from_disk
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import random

MODEL_NAME = "openai/whisper-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset = load_from_disk("data/fleurs_hi_processed")

processor = WhisperProcessor.from_pretrained(
    MODEL_NAME,
    language="Hindi",
    task="transcribe"
)

model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# pick 5 random samples from test set
samples = random.sample(list(dataset["test"]), 5)

for i, sample in enumerate(samples, 1):
    audio = sample["audio"]["array"]
    sr = sample["audio"]["sampling_rate"]

    inputs = processor(
        audio,
        sampling_rate=sr,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        predicted_ids = model.generate(inputs.input_features)

    prediction = processor.tokenizer.decode(
        predicted_ids[0],
        skip_special_tokens=True
    )

    print(f"\n--- Sample {i} ---")
    print("Ground Truth :", sample["text"])
    print("Prediction   :", prediction)
