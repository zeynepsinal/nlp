from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL_DIR = ROOT / "models" / "berturk_campaign_classifier"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "dbmdz/bert-base-turkish-cased"
SEED = 42
MAX_LENGTH = 256

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

train_df = pd.read_csv(DATA / "train.csv")
val_df = pd.read_csv(DATA / "validation.csv")
test_df = pd.read_csv(DATA / "test_real.csv")

labels = sorted(train_df["kategori"].unique().tolist())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

print("Etiketler:", label2id)
print("CUDA kullanılabilir:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("UYARI: Eğitim CPU üzerinde çalışacak ve daha yavaş olabilir.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class CampaignDataset(torch.utils.data.Dataset):
    def __init__(self, frame):
        self.texts = frame["metin"].astype(str).tolist()
        self.labels = [label2id[x] for x in frame["kategori"].tolist()]
        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(value[idx])
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_ds = CampaignDataset(train_df)
val_ds = CampaignDataset(val_df)
test_ds = CampaignDataset(test_df)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    label2id=label2id,
    id2label=id2label,
)

def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=-1)

    precision, recall, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_macro,
        "precision_macro": precision,
        "recall_macro": recall,
    }

args = TrainingArguments(
    output_dir=str(ROOT / "models" / "berturk_checkpoints"),
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    seed=SEED,
    data_seed=SEED,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print("\n=== BERTurk EĞİTİMİ BAŞLIYOR ===")
trainer.train()

print("\n=== SADECE GERÇEK TEST SETİ SONUCU ===")
test_metrics = trainer.evaluate(test_ds)

pred_output = trainer.predict(test_ds)
pred_ids = np.argmax(pred_output.predictions, axis=-1)
true_ids = pred_output.label_ids

from sklearn.metrics import classification_report, confusion_matrix

true_labels = [id2label[int(i)] for i in true_ids]
pred_labels = [id2label[int(i)] for i in pred_ids]

report_text = classification_report(
    true_labels,
    pred_labels,
    labels=labels,
    digits=4,
    zero_division=0,
)
report_dict = classification_report(
    true_labels,
    pred_labels,
    labels=labels,
    output_dict=True,
    zero_division=0,
)
cm = confusion_matrix(true_labels, pred_labels, labels=labels).tolist()

accuracy = accuracy_score(true_labels, pred_labels)
macro_f1 = f1_score(true_labels, pred_labels, average="macro")

print(f"Accuracy : {accuracy:.4f}")
print(f"Macro F1 : {macro_f1:.4f}")
print("\nClassification Report:")
print(report_text)

trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

result = {
    "model": MODEL_NAME,
    "test_source": "real_only",
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "trainer_test_metrics": {
        k: float(v) for k, v in test_metrics.items()
        if isinstance(v, (int, float))
    },
    "labels": labels,
    "classification_report": report_dict,
    "confusion_matrix": cm,
}

(RESULTS / "berturk_metrics.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"\nModel kaydedildi: {MODEL_DIR}")
print(f"Sonuçlar: {RESULTS / 'berturk_metrics.json'}")