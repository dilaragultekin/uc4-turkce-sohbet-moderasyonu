"""Aşama 1 — Veriyi indir, ayrıştır, kaydet ve bütünlüğünü raporla.

Çalıştırma:  python src/01_data_prep.py

Kaynak: OffensEval-TR 2020 (Çöltekin, LREC 2020), CC-BY-2.0.
Hugging Face'teki stefan-it/offenseval2020_tr sürümü kullanılır:
  - train.txt  30 000  (resmi eğitim setinin alt kümesi)
  - dev.txt     1 756  (resmi eğitim setinden ayrılmış doğrulama)
  - test.txt    3 528  (RESMİ TEST SETİ, değiştirilmemiş)
Biçim: "__label__NOT metin" / "__label__OFF metin" (satır başına bir örnek).

Bu betik hiçbir örneği silmez veya değiştirmez; yalnızca biçimi TSV'ye çevirir
ve split'ler arası çakışmaları RAPORLAR (bilimsel dürüstlük: zor/çakışan
örnekler test setinden çıkarılmaz).
"""
from __future__ import annotations

import csv
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import (  # noqa: E402
    DATA_PROCESSED, DATA_RAW, DATASET_REPO, DATASET_REVISION, LABEL2ID,
    RESULTS_DIR, SPLITS, save_json,
)

LINE_RE = re.compile(r"^__label__(NOT|OFF) (.*)$")


def download() -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    paths = {}
    for split in SPLITS:
        paths[split] = Path(hf_hub_download(
            repo_id=DATASET_REPO, repo_type="dataset", filename=f"{split}.txt",
            revision=DATASET_REVISION, local_dir=DATA_RAW / "offenseval2020_tr",
        ))
    return paths


def parse(path: Path) -> list[tuple[str, int]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            m = LINE_RE.match(line)
            if not m:
                raise ValueError(f"{path.name}:{n} beklenmeyen biçim: {line[:80]!r}")
            rows.append((m.group(2), LABEL2ID[m.group(1)]))
    return rows


def main() -> None:
    paths = download()
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    report = {"source": DATASET_REPO, "revision": DATASET_REVISION,
              "license": "CC-BY-2.0", "splits": {}}
    data = {}
    for split, p in paths.items():
        rows = parse(p)
        data[split] = rows
        out = DATA_PROCESSED / f"{split}.tsv"
        with open(out, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            w.writerow(["id", "text", "label"])
            for i, (t, y) in enumerate(rows):
                w.writerow([f"{split}-{i}", t, y])
        labels = np.array([y for _, y in rows])
        n_words = np.array([len(t.split()) for t, _ in rows])
        texts = [t for t, _ in rows]
        report["splits"][split] = {
            "n": len(rows),
            "n_off": int(labels.sum()),
            "n_not": int((labels == 0).sum()),
            "off_ratio": round(float(labels.mean()), 4),
            "words_median": float(np.median(n_words)),
            "words_p95": float(np.percentile(n_words, 95)),
            "words_max": int(n_words.max()),
            "empty_texts": int(sum(1 for t in texts if not t.strip())),
            "duplicate_texts_within": len(texts) - len(set(texts)),
            "sha256_raw": hashlib.sha256(p.read_bytes()).hexdigest(),
        }

    # Split'ler arası çakışma (bilgi; örnekler silinmez)
    sets = {s: {t for t, _ in data[s]} for s in SPLITS}
    report["overlap"] = {
        "train_test": len(sets["train"] & sets["test"]),
        "dev_test": len(sets["dev"] & sets["test"]),
        "train_dev": len(sets["train"] & sets["dev"]),
        "train_test_examples": sorted(sets["train"] & sets["test"])[:5],
    }
    char_counts = Counter(c for t, _ in data["train"] for c in t)
    report["train_char_info"] = {
        "has_turkish_chars": all(char_counts[c] > 0 for c in "çğıöşü"),
        "mention_placeholder_count": sum(t.count("@USER") for t, _ in data["train"]),
    }
    save_json(report, RESULTS_DIR / "data_report.json")

    print(f"Kaynak: {DATASET_REPO} @ {DATASET_REVISION[:10]}")
    for s, r in report["splits"].items():
        print(f"  {s:5s} n={r['n']:6d}  OFF={r['n_off']:5d} ({r['off_ratio']:.1%})  "
              f"kelime medyan={r['words_median']:.0f} p95={r['words_p95']:.0f} "
              f"max={r['words_max']}  boş={r['empty_texts']}  "
              f"tekrar={r['duplicate_texts_within']}")
    print("Çakışma:", {k: v for k, v in report["overlap"].items() if k != "train_test_examples"})
    if report["overlap"]["train_test"]:
        print("  UYARI: train ve test arasında aynı metinler var (veri setinin kendisinden).")
        print("  Bunlar test setinden ÇIKARILMAZ; sunumda sınırlılık olarak raporlanır.")
    print(f"Kaydedildi: {DATA_PROCESSED}/{{train,dev,test}}.tsv, results/data_report.json")


if __name__ == "__main__":
    main()
