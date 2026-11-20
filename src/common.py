"""Ortak yardımcılar: yollar, sabitler, veri yükleme, metrikler, eşik seçimi.

Bütün betikler ve uygulama bu modülü kullanır; böylece veri yolu, etiket
kodlaması ve metrik hesabı tek bir yerde tanımlıdır (R3: aynı test seti,
R6: diyagram kodla eşleşir).
"""
from __future__ import annotations

import csv
import json
import os
import random
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ── Yollar ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
LEXICON_DIR = ROOT / "src" / "lexicon"

# ── Veri kaynağı (sabitlenmiş sürüm = tekrarlanabilirlik) ──────────────────
# stefan-it/offenseval2020_tr: resmi OffensEval-TR 2020 eğitim setini
# 30 000 train + 1 756 dev olarak böler; resmi test seti (3 528) değişmez.
DATASET_REPO = "stefan-it/offenseval2020_tr"
DATASET_REVISION = "f34679caea00504400fc70421adb853765c2a0f7"
SPLITS = ("train", "dev", "test")

# Dış sözlük kaynağı (CC-BY-SA-4.0), sabit commit
EXT_LEXICON_URL = (
    "https://raw.githubusercontent.com/ooguz/turkce-kufur-karaliste/"
    "56962ae66c3160871e1b9a4ba70f94c90e5a64e5/karaliste.txt"
)

# Transformer
BERT_NAME = "dbmdz/bert-base-turkish-cased"
BERT_REVISION = "b6e1de16c983e0f2c70664591ea3f22810072608"
BERT_MAX_LEN = 128

# Gecikme bütçesi ve CPU ayarı (R4 + UC4 D alanı)
LATENCY_BUDGET_MS = 50.0
CPU_THREADS = 4

LABEL2ID = {"NOT": 0, "OFF": 1}
ID2LABEL = {0: "clean", 1: "offensive"}
SEED = 42


# ── Genel ─────────────────────────────────────────────────────────────────
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def path_size_mb(path: Path) -> float:
    """Dosya ya da klasörün diskteki boyutu (MB)."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


# ── Veri ──────────────────────────────────────────────────────────────────
def load_split(name: str) -> tuple[list[str], np.ndarray]:
    """data/processed/{name}.tsv → (metinler, etiketler 0/1)."""
    if name not in SPLITS:
        raise ValueError(f"Bilinmeyen split: {name}")
    path = DATA_PROCESSED / f"{name}.tsv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} yok. Önce `python src/01_data_prep.py` çalıştırın."
        )
    texts, labels = [], []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL):
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, np.asarray(labels, dtype=np.int64)


TOKEN_RE = re.compile(r"@user|url|\w+|[^\w\s]", re.UNICODE)


def simple_tokenize(text: str) -> list[str]:
    """Normalize edilmiş metin için basit tokenizer (kelime + noktalama/emoji)."""
    return TOKEN_RE.findall(text)


# ── Metrikler ve eşik ─────────────────────────────────────────────────────
def evaluate(y_true, p_off, threshold: float) -> dict:
    """OFF olasılıklarını eşikle etikete çevirip metrikleri hesaplar."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(p_off) >= threshold).astype(int)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "f1_off": float(f1_score(y_true, y_pred, pos_label=1)),
        "f1_not": float(f1_score(y_true, y_pred, pos_label=0)),
        "precision_off": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_off": float(recall_score(y_true, y_pred, pos_label=1)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "threshold": float(threshold),
        "n": int(len(y_true)),
    }


def choose_threshold(y_dev, p_dev) -> float:
    """Eşiği YALNIZCA dev setinde macro-F1'i en yükselten değer olarak seçer.

    Eşitlikte 0.5'e en yakın eşik seçilir. Test seti burada asla kullanılmaz.
    """
    best_t, best_f1 = 0.5, -1.0
    for t in np.round(np.arange(0.05, 0.951, 0.01), 2):
        f1 = f1_score(y_dev, (np.asarray(p_dev) >= t).astype(int), average="macro")
        if f1 > best_f1 + 1e-12 or (abs(f1 - best_f1) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5)):
            best_t, best_f1 = float(t), float(f1)
    return best_t
