# BERTurk Kampanya Sınıflandırma Deneyi

Amaç:
TF-IDF + Logistic Regression baseline ile BERTurk modelini AYNI gerçek test setinde karşılaştırmak.

Model:
`dbmdz/bert-base-turkish-cased`

## 0. Gerçek CSV'yi ekle

Mevcut 374 örnekli ana veri setini şu dosya adıyla koy:

```text
data/kampanyalar.csv
```

Paketin içinde `data/synthetic.csv` zaten bulunuyor.

Ana CSV'nin minimum sütunları:

```text
metin,kategori
```

Eğer `kaynak` sütunun varsa script bunu kullanır.
`kaynak=synthetic` olan satırlar gerçek test setine alınmaz.

## 1. Kütüphaneler

Ana klasörde:

```powershell
py -m pip install -r requirements_bert.txt
```

## 2. Veriyi ayır

```powershell
py prepare_data.py
```

Bu adım:
- gerçek veriden train / validation / test ayırır
- sentetik veriyi yalnızca train'e ekler
- test setini sadece gerçek veriden oluşturur

## 3. Aynı split'te TF-IDF baseline

```powershell
py train_tfidf_same_split.py
```

## 4. BERTurk eğit

```powershell
py train_berturk.py
```

İlk çalıştırmada BERTurk modeli internetten indirilir.
Model dosyaları yüzlerce MB olabilir.

GPU varsa otomatik kullanılır.
CPU'da da çalışır ancak daha yavaş olabilir.

## 5. Karşılaştır

```powershell
py compare_results.py
```

Böylece iki model AYNI gerçek test setinde karşılaştırılır.

## 6. BERTurk'u manuel test et

```powershell
py predict_berturk.py
```

## Dosyalar

```text
data/
  kampanyalar.csv      <- senin gerçek/ana verin
  synthetic.csv        <- 140 sentetik örnek
  train.csv
  validation.csv
  test_real.csv

models/
  tfidf_augmented.joblib
  berturk_campaign_classifier/

results/
  tfidf_metrics.json
  berturk_metrics.json
```

## Neden test sadece gerçek?

Sentetik metinleri test setine koyarsak modelin gerçek banka metinlerinde ne kadar iyi
çalıştığını güvenilir biçimde ölçemeyiz. Sentetik augmentation yalnızca eğitim
performansını desteklemek için kullanılır.