from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "berturk_campaign_classifier"

if not MODEL_DIR.exists():
    raise FileNotFoundError(
        f"BERTurk modeli bulunamadı: {MODEL_DIR}\n"
        "Önce `py train_berturk.py` ile modeli eğit."
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

app = FastAPI(
    title="Katılım Bankacılığı BERTurk NLP API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CampaignRequest(BaseModel):
    metin: str

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "berturk-campaign-classifier",
        "device": str(device),
        "model": "dbmdz/bert-base-turkish-cased"
    }

@app.post("/classify")
def classify(request: CampaignRequest):
    text = request.metin.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Metin boş olamaz.")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    scores = []
    for idx, score in enumerate(probabilities.tolist()):
        label = model.config.id2label[idx]
        scores.append({
            "kategori": label,
            "skor": float(score)
        })

    scores.sort(key=lambda item: item["skor"], reverse=True)

    return {
        "kategori": scores[0]["kategori"],
        "guven": scores[0]["skor"],
        "tum_skorlar": scores,
        "model": "BERTurk",
        "model_name": "dbmdz/bert-base-turkish-cased"
    }