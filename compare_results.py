from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

tfidf_path = RESULTS / "tfidf_metrics.json"
bert_path = RESULTS / "berturk_metrics.json"

if not tfidf_path.exists() or not bert_path.exists():
    raise FileNotFoundError(
        "Önce sırasıyla train_tfidf_same_split.py ve train_berturk.py çalıştır."
    )

tfidf = json.loads(tfidf_path.read_text(encoding="utf-8"))
bert = json.loads(bert_path.read_text(encoding="utf-8"))

print("\n=== AYNI GERÇEK TEST SETİNDE MODEL KARŞILAŞTIRMASI ===")
print(f"{'Model':38s} {'Accuracy':>10s} {'Macro F1':>10s}")
print("-" * 62)
print(f"{'TF-IDF + Logistic Regression':38s} {tfidf['accuracy']:10.4f} {tfidf['macro_f1']:10.4f}")
print(f"{'BERTurk':38s} {bert['accuracy']:10.4f} {bert['macro_f1']:10.4f}")

delta_acc = bert["accuracy"] - tfidf["accuracy"]
delta_f1 = bert["macro_f1"] - tfidf["macro_f1"]

print("\nBERTurk farkı:")
print(f"Accuracy : {delta_acc:+.4f}")
print(f"Macro F1 : {delta_f1:+.4f}")