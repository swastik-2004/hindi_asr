# 🎙️ Hindi ASR API (Whisper + FastAPI)

A lightweight Automatic Speech Recognition (ASR) API for Hindi speech using OpenAI Whisper and FastAPI.

This project demonstrates:

* Speech-to-text using Whisper
* FastAPI backend deployment
* Model loading and inference pipeline
* Clean project structuring
* Zero-shot evaluation before fine-tuning

---

## 🚀 Project Overview

This repository implements a Hindi ASR system using:

* 🤗 HuggingFace Transformers
* 🧠 OpenAI Whisper (small)
* ⚡ FastAPI for serving inference
* 🎧 Google FLEURS (Hindi) for testing

The goal of this project was to:

* Understand multilingual ASR systems
* Practice dataset preparation with HuggingFace Datasets
* Build an inference API
* Prepare for real-world ML deployment workflows

---

## 🏗️ Project Structure

```
hindi_asr/
│
├── app/
│   ├── main.py          # FastAPI entrypoint
│   ├── model.py         # Model + processor loading
│
├── src/
│   ├── prepare_fleurs.py
│   ├── zeroshot_inference.py
│
├── requirements.txt
└── README.md
```

---

## 📦 Installation

Create a fresh environment:

```bash
conda create -n hindi_asr python=3.10
conda activate hindi_asr
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the API

```bash
uvicorn app.main:app --reload
```

Visit:

```
http://127.0.0.1:8000/docs
```

Upload a `.wav` file to test transcription.

---

## 🧠 Model Details

* Model: `openai/whisper-small`
* Task: Hindi transcription
* Device: CPU (GPU supported if available)

---

## 📊 Zero-Shot Results (Sample)

Example predictions from Hindi speech:

```
Ground Truth:
कुछ अणुओं में अस्थिर केंद्रक होता है...

Prediction:
कुछ अणुओं में अस्थिर केंद्रक होता है...
```

Model performs reasonably well in zero-shot Hindi without fine-tuning.

---

## ⚠️ Challenges Faced

During development, several practical ML engineering challenges were encountered:

* HuggingFace dataset version conflicts
* PyArrow & NumPy compatibility issues
* Windows symlink warnings
* Dependency resolution conflicts
* Docker image size concerns (~1GB+ model)
* Slow CPU inference for Whisper-small

These were resolved through:

* Version pinning
* Environment isolation
* Proper dependency management

---

## 🛠️ Future Improvements

* Fine-tune on larger Hindi dataset
* Add batching support
* Add streaming transcription
* Implement async processing
* Add load balancing
* Deploy on cloud (Render / Railway / Azure)
* Add logging & monitoring

---

## 📚 Learning Outcomes

This project helped in understanding:

* HuggingFace Datasets pipeline
* Whisper architecture
* FastAPI async handling
* Model serving fundamentals
* Environment hygiene & dependency management
* Real-world debugging

---

## 🧹 Cleanup Note

This repository intentionally does NOT include:

* Docker configuration
* Model weights
* Virtual environments

Everything can be rebuilt cleanly using `requirements.txt`.

---

## 👨‍💻 Author

Swastik Dasgupta
AIML Undergraduate
Interested in ML Systems, Deployment, and Applied AI

---

If you found this project useful, feel free to ⭐ the repo.
