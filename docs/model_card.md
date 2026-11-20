---
language: tr
pipeline_tag: text-classification
base_model: dbmdz/bert-base-turkish-cased
datasets:
- stefan-it/offenseval2020_tr
tags:
- offensive-language
- content-moderation
- turkish
- quantization
---

# UC4 — Türkçe Sohbet Moderasyonu: eğitilmiş model dosyaları

SWE015 (Introduction to Large Language Models) ödevi, Use Case 4 için eğitilmiş
model dosyaları. Kod, kurulum ve rapor GitHub deposundadır:
**https://github.com/dilaragultekin/uc4-turkce-sohbet-moderasyonu**

Bu depoyu elle indirmeniz gerekmez: uygulama (`python app.py`) eksik dosyaları
ilk açılışta buradan, sabitlenmiş bir commit'ten indirir ve her dosyanın SHA-256
özetini GitHub'daki `models_manifest.json` ile karşılaştırır.

## İçerik

| Klasör | Model | Uygulamada |
|---|---|---|
| `tfidf/` | TF-IDF (kelime 1–2 + karakter 2–5 n-gram) + Lojistik Regresyon | evet |
| `nn/` | Kendi Word2Vec'imiz (yalnızca train) + 1D CNN | evet |
| `bertturk/` | BERTurk ince ayar, FP32 | evet |
| `bertturk_int8/` | BERTurk, dinamik INT8 quantization (`nn.Linear`) — **varsayılan model** | evet |
| `w2v/` | Word2Vec vektörleri (CNN'in eğitimi için) | hayır, arşiv |

Kural tabanlı model (sözlük) burada değil, GitHub deposundadır (`src/lexicon/`).

## Eğitim ve değerlendirme

- Veri: OffensEval-TR 2020 (`stefan-it/offenseval2020_tr`, commit `f34679c`):
  30 000 train / 1 756 dev / 3 528 resmi test.
- Bütün seçimler (hiperparametre, mimari, karar eşiği) **dev** setinde yapıldı;
  test seti yalnızca final değerlendirmede bir kez kullanıldı.
- Her klasördeki `meta.json` dev'de seçilen eşiği ve eğitim ayrıntılarını içerir.

Resmi test seti, CPU (Intel i7-13650HX, 4 thread), tek mesaj uçtan uca gecikme:

| Model | Macro-F1 | Boyut | Ortalama / p95 (ms) |
|---|---|---|---|
| TF-IDF + LR | 0.7697 | 6.9 MB | 1.21 / 2.18 |
| Word2Vec + CNN | 0.7147 | 25.8 MB | 0.37 / 0.61 |
| BERTurk FP32 | 0.8114 | 442.5 MB | 22.15 / 40.11 |
| BERTurk INT8 | 0.8177 | 186.0 MB | 16.40 / 22.27 |

## Sınırlılıklar

- Etiketler Twitter verisinden gelir; oyun sohbetinin argosu ve yazım biçimi farklı olabilir.
- Olumsuzlama ve ironiyi kaçırabilir (ör. "Sen salak değilsin" → saldırgan).
- Bir kişiyi sansürlemek veya cezalandırmak için tek başına kullanılmamalıdır; insan denetimi gerekir.

## Güvenlik notu

`bertturk_int8/model_int8.pt` (torch.load) ve `tfidf/model.joblib` pickle tabanlıdır.
Yalnızca SHA-256 özeti GitHub'daki `models_manifest.json` ile eşleşen dosyaları yükleyin;
uygulama bunu kendisi yapar.

## Atıf ve lisans

- Temel model: [`dbmdz/bert-base-turkish-cased`](https://huggingface.co/dbmdz/bert-base-turkish-cased) (MIT).
- Veri: Çöltekin, Ç. (2020). *A Corpus of Turkish Offensive Language on Social Media.*
  LREC 2020, 6174–6184. CC-BY-2.0.
