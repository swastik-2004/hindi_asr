import os 
import uuid
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException

from api.model import processor, model, DEVICE
from api.utils import load_audio

app=FastAPI(title="Hindi ASR API", version="1.0")

UPLOAD_DIR="temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Hindi ASR API"}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".wav", ".mp3")):
        raise HTTPException(status_code=400, detail="Only .wav and .mp3 files are supported")
    
    temp_filename=f"{uuid.uuid4()}_{file.filename}"
    temp_filepath=os.path.join(UPLOAD_DIR, temp_filename)

    with open(temp_filepath, "wb") as f:
        f.write(await file.read())

    try:
        audio, sr = load_audio(temp_filepath)
        inputs = processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            predicted_ids = model.generate(inputs.input_features)

        transcription = processor.tokenizer.decode(
            predicted_ids[0],
            skip_special_tokens=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

        return {"transcription": transcription}