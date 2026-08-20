from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test_real.csv")

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )),
])

model.fit(train["metin"], train["kategori"])
pred = model.predict(test["metin"])

metrics = {
    "model": "TF-IDF + Logistic Regression",
    "test_source": "real_only",
    "accuracy": float(accuracy_score(test["kategori"], pred)),
    "macro_f1": float(f1_score(test["kategori"], pred, average="macro")),
    "classification_report": classification_report(
        test["kategori"], pred, output_dict=True, zero_division=0
    ),
}

joblib.dump(model, MODELS / "tfidf_augmented.joblib")
(RESULTS / "tfidf_metrics.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\n=== TF-IDF BASELINE / AYNI TEST SETİ ===")
print(f"Accuracy : {metrics['accuracy']:.4f}")
print(f"Macro F1 : {metrics['macro_f1']:.4f}")
print("\n")
print(classification_report(test["kategori"], pred, digits=4, zero_division=0))