from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "berturk_campaign_classifier"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    pairs = []
    for idx, score in enumerate(probs.tolist()):
        label = model.config.id2label[idx]
        pairs.append((label, score))

    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs

if __name__ == "__main__":
    text = input("Kampanya metni: ").strip()
    scores = predict(text)

    print("\nTahmin:", scores[0][0])
    print(f"Güven: %{scores[0][1] * 100:.2f}")
    print("\nTüm sınıflar:")
    for label, score in scores:
        print(f"{label:22s} %{score * 100:.2f}")