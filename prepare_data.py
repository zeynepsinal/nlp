from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

REAL_PATH = DATA_DIR / "kampanyalar.csv"
SYN_PATH = DATA_DIR / "synthetic.csv"

OUT_TRAIN = DATA_DIR / "train.csv"
OUT_VAL = DATA_DIR / "validation.csv"
OUT_TEST = DATA_DIR / "test_real.csv"
OUT_INFO = DATA_DIR / "split_info.json"

SEED = 42
VALID_CLASSES = {
    "konut_finansmani",
    "tasit_finansmani",
    "ihtiyac_finansmani",
    "kredi_karti",
    "katilma_hesabi",
    "yatirim",
    "diger",
}

def normalize(df, default_source):
    required = {"metin", "kategori"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Eksik sütunlar: {missing}")

    df = df.copy()
    df["metin"] = df["metin"].astype(str).str.strip()
    df["kategori"] = df["kategori"].astype(str).str.strip()

    if "kaynak" not in df.columns:
        df["kaynak"] = default_source
    else:
        df["kaynak"] = df["kaynak"].fillna(default_source).astype(str).str.lower().str.strip()

    if "zorluk" not in df.columns:
        df["zorluk"] = "belirsiz"

    df = df[df["metin"].ne("")]
    df = df[df["kategori"].isin(VALID_CLASSES)]
    return df

if not REAL_PATH.exists():
    raise FileNotFoundError(
        "\nGerçek veri dosyası bulunamadı.\n"
        "Mevcut 374 örnekli dosyanı şu isimle kopyala:\n"
        f"{REAL_PATH}\n"
    )

real_all = normalize(pd.read_csv(REAL_PATH), "real")
syn_file = normalize(pd.read_csv(SYN_PATH), "synthetic")

# Eğer ana CSV içinde kaynak sütunu varsa, sentetik diye işaretlenmiş satırları
# gerçek test havuzuna sokmuyoruz.
real_pool = real_all[real_all["kaynak"] != "synthetic"].copy()
embedded_syn = real_all[real_all["kaynak"] == "synthetic"].copy()

synthetic_pool = pd.concat([embedded_syn, syn_file], ignore_index=True)
synthetic_pool["kaynak"] = "synthetic"

# Aynı metin iki tarafta varsa gerçek olanı koru, sentetik kopyayı çıkar.
real_pool = real_pool.drop_duplicates(subset=["metin"], keep="first")
synthetic_pool = synthetic_pool[~synthetic_pool["metin"].isin(set(real_pool["metin"]))]
synthetic_pool = synthetic_pool.drop_duplicates(subset=["metin"], keep="first")

counts = real_pool["kategori"].value_counts()
too_small = counts[counts < 5]
if not too_small.empty:
    raise ValueError(
        "Gerçek veride bazı sınıflar stratified split için çok küçük:\n"
        + too_small.to_string()
    )

# Gerçek verinin %20'si dokunulmamış test seti.
real_trainval, test_real = train_test_split(
    real_pool,
    test_size=0.20,
    random_state=SEED,
    stratify=real_pool["kategori"],
)

# Kalan gerçek verinin %15'i validation olacak.
# (toplam gerçek verinin yaklaşık %12'si)
real_train, val_real = train_test_split(
    real_trainval,
    test_size=0.15,
    random_state=SEED,
    stratify=real_trainval["kategori"],
)

# Sentetik veri SADECE train'e eklenir.
train = pd.concat([real_train, synthetic_pool], ignore_index=True)
train = train.sample(frac=1, random_state=SEED).reset_index(drop=True)

for df in (train, val_real, test_real):
    df["metin"] = df["metin"].astype(str)
    df["kategori"] = df["kategori"].astype(str)

train.to_csv(OUT_TRAIN, index=False, encoding="utf-8-sig")
val_real.to_csv(OUT_VAL, index=False, encoding="utf-8-sig")
test_real.to_csv(OUT_TEST, index=False, encoding="utf-8-sig")

info = {
    "seed": SEED,
    "train_total": len(train),
    "train_real": int((train["kaynak"] != "synthetic").sum()),
    "train_synthetic": int((train["kaynak"] == "synthetic").sum()),
    "validation_real": len(val_real),
    "test_real": len(test_real),
    "train_class_counts": train["kategori"].value_counts().to_dict(),
    "validation_class_counts": val_real["kategori"].value_counts().to_dict(),
    "test_class_counts": test_real["kategori"].value_counts().to_dict(),
}

OUT_INFO.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n=== VERİ AYRIMI TAMAMLANDI ===")
print(f"Train toplam     : {len(train)}")
print(f"  Gerçek         : {(train['kaynak'] != 'synthetic').sum()}")
print(f"  Sentetik       : {(train['kaynak'] == 'synthetic').sum()}")
print(f"Validation gerçek: {len(val_real)}")
print(f"Test gerçek      : {len(test_real)}")

print("\nTest sınıf dağılımı:")
print(test_real["kategori"].value_counts())

print("\nÖNEMLİ: Sentetik veri validation/test setine eklenmedi.")