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

# UC4 — Turkish Chat Moderation: trained model files

Trained model files for the SWE015 (Introduction to Large Language Models) homework,
Use Case 4. The code, setup steps and report are in the GitHub repository:
**https://github.com/dilaragultekin/uc4-turkce-sohbet-moderasyonu**

You do not need to download this repository by hand: the app (`python app.py`) downloads
the missing files from here on its first start, from a pinned commit, and compares every
file's SHA-256 checksum with `models_manifest.json` in the GitHub repository.

## Contents

| Folder | Model | Used by the app |
|---|---|---|
| `tfidf/` | TF-IDF (word 1–2 + character 2–5 n-grams) + Logistic Regression | yes |
| `nn/` | Our own Word2Vec (train only) + 1D CNN | yes |
| `bertturk/` | BERTurk fine-tuned, FP32 | yes |
| `bertturk_int8/` | BERTurk, dynamic INT8 quantization (`nn.Linear`) — **default model** | yes |
| `w2v/` | Word2Vec vectors (for training the CNN) | no, archive |

The rule-based model (lexicon) is not here; it is in the GitHub repository (`src/lexicon/`).

## Training and evaluation

- Data: OffensEval-TR 2020 (`stefan-it/offenseval2020_tr`, commit `f34679c`):
  30,000 train / 1,756 dev / 3,528 official test.
- All choices (hyperparameters, architecture, decision threshold) were made on the **dev** set;
  the test set was used only once, in the final evaluation.
- The `meta.json` in each folder holds the threshold chosen on dev and the training details.

Official test set, CPU (Intel i7-13650HX, 4 threads), single-message end-to-end latency:

| Model | Macro-F1 | Size | Mean / p95 (ms) |
|---|---|---|---|
| TF-IDF + LR | 0.7697 | 6.9 MB | 1.21 / 2.18 |
| Word2Vec + CNN | 0.7147 | 25.8 MB | 0.37 / 0.61 |
| BERTurk FP32 | 0.8114 | 442.5 MB | 22.15 / 40.11 |
| BERTurk INT8 | 0.8177 | 186.0 MB | 16.40 / 22.27 |

## Limitations

- The labels come from Twitter data; game-chat slang and spelling can differ.
- It can miss negation and irony (e.g. "Sen salak değilsin" — "you are not stupid" → offensive).
- It should not be used on its own to censor or punish a person; human review is needed.

## Security note

`bertturk_int8/model_int8.pt` (torch.load) and `tfidf/model.joblib` are pickle-based.
Only load files whose SHA-256 checksum matches `models_manifest.json` in the GitHub repository;
the app does this itself.

## Attribution and license

- Base model: [`dbmdz/bert-base-turkish-cased`](https://huggingface.co/dbmdz/bert-base-turkish-cased) (MIT).
- Data: Çöltekin, Ç. (2020). *A Corpus of Turkish Offensive Language on Social Media.*
  LREC 2020, 6174–6184. CC-BY-2.0.
