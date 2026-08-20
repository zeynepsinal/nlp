# Katılım Bankacılığı NLP Analiz Sistemi

Bu proje, katılım bankacılığı kampanya metinlerini Türkçe doğal dil işleme yöntemleriyle analiz etmek amacıyla geliştirilmiştir.

Sistem şu anda:

* **BERTurk** ile kampanya kategorisini sınıflandırır.
* **Regex tabanlı bilgi çıkarımı** ile oran, tutar, vade, tarih ve benzeri alanları çıkarır.
* **Katılım bankacılığı ontolojisi** ile terminolojiyi normalize eder.
* **FastAPI** üzerinden modeli servis eder.
* **HTML/CSS/JavaScript arayüzü** üzerinden analiz sonuçlarını gösterir.
* TF-IDF + Logistic Regression baseline modeli ile BERTurk sonuçlarını karşılaştırır.

## NLP Pipeline

```text
Kampanya Metni
      ↓
BERTurk
      ↓
Kategori Sınıflandırması
      ↓
Regex Tabanlı Bilgi Çıkarımı
      ↓
Katılım Bankacılığı Ontolojisi
      ↓
Yapılandırılmış JSON
      ↓
Web Arayüzü
```

Desteklenen kategoriler:

```text
konut_finansmani
tasit_finansmani
ihtiyac_finansmani
kredi_karti
katilma_hesabi
yatirim
diger
```

## Model Sonuçları

| Model                        |   Accuracy |   Macro F1 |
| ---------------------------- | ---------: | ---------: |
| TF-IDF + Logistic Regression |     0.8333 |     0.8214 |
| **BERTurk**                  | **0.8889** | **0.8857** |

BERTurk, baseline modele göre Accuracy’de **+0.0556**, Macro F1’da **+0.0643** iyileşme sağlamıştır.

## Git LFS

Model dosyaları büyük olduğu için projede **Git LFS** kullanılmaktadır.

Özellikle:

```text
*.safetensors
*.pt
*.pth
```

dosyaları LFS ile takip edilir.

Projeyi klonlamadan önce:

```bash
git lfs install
```

Ardından:

```bash
git clone https://github.com/zeynepsinal/nlp.git
cd nlp
git lfs pull
```

LFS dosyalarını kontrol etmek için:

```bash
git lfs ls-files
```

## Kurulum

Gerekli paketleri kurun:

```powershell
py -m pip install -r requirements_bert.txt
```

Pip bulunamazsa:

```powershell
py -m ensurepip --upgrade
py -m pip install --upgrade pip
```

## Uygulamayı Çalıştırma

Önce BERTurk API’yi başlatın:

```powershell
py -m uvicorn api_berturk:app --reload --port 8000
```

Başarılı olduğunda:

```text
Application startup complete.
Uvicorn running on http://127.0.0.1:8000
```

görülür.

API kontrolü:

```text
http://127.0.0.1:8000
```

API dokümantasyonu:

```text
http://127.0.0.1:8000/docs
```

Ardından:

```text
index_berturk.html
```

dosyasını açın.

Sonuç ekranında:

```text
Kategori Kaynağı:
Transformer — BERTurk
```

görünüyorsa model başarıyla çalışmaktadır.

## Modeli Yeniden Eğitme

```powershell
py prepare_data.py
py train_tfidf_same_split.py
py train_berturk.py
```

Model karşılaştırması:

```powershell
py compare_results.py
```

Terminalden tahmin:

```powershell
py predict_berturk.py
```

## Roadmap

Tamamlananlar:

* [x] TF-IDF + Logistic Regression baseline
* [x] BERTurk sınıflandırması
* [x] Regex bilgi çıkarımı
* [x] Katılım bankacılığı ontolojisi
* [x] FastAPI
* [x] Web arayüzü
* [x] Model karşılaştırması
* [x] Git LFS

Planlananlar:

* [ ] Banka sitelerinden canlı kampanya verisi çekme
* [ ] Scraper / crawler
* [ ] Canlı verinin NLP pipeline’ına otomatik aktarılması
* [ ] NER tabanlı bilgi çıkarımı
* [ ] Veritabanı
* [ ] Banka kampanyalarının karşılaştırılması
* [ ] Kullanıcı önceliklerine göre avantaj skoru
* [ ] Chatbot
* [ ] Otomatik veri güncelleme
* [ ] Docker deployment

## Repository

```text
https://github.com/zeynepsinal/nlp
```

> Proje aktif olarak geliştirilmektedir. Canlı veri çekme, karşılaştırma, NER ve chatbot modülleri sonraki aşamalarda eklenecektir.
> ::: 
