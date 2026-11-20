"""Tüm modeller için ortak çıkarım arayüzü (CPU).

06_benchmark.py ve app.py AYNI sınıfları kullanır; böylece raporlanan sayılar
uygulamada çalışan kodun sayılarıdır (R4, R5, R6).

Her predictor:
  predict_proba(texts) -> np.ndarray  (P(offensive))
  threshold                            (dev setinde seçilmiş, meta.json'dan)
  artifact_files                       (boyut ölçümü için ağırlık dosyaları)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common import (
    BERT_MAX_LEN, LEXICON_DIR, MODELS_DIR, load_json, path_size_mb, simple_tokenize,
)
from src.lexicon.normalizer import NORMALIZER_VERSION, normalize_text

MODEL_KEYS = ["rule", "tfidf", "nn", "bert", "bert_int8"]
DISPLAY_NAMES = {
    "rule": "Kural: sözlük + edit distance",
    "tfidf": "TF-IDF + Lojistik Regresyon",
    "nn": "Word2Vec + Sinir Ağı",
    "bert": "BERTurk (FP32)",
    "bert_int8": "BERTurk (INT8 quantized)",
}


class Predictor:
    key: str = ""
    threshold: float = 0.5
    artifact_files: list[Path] = []
    meta: dict = {}

    @property
    def name(self) -> str:
        return DISPLAY_NAMES[self.key]

    def size_mb(self) -> float:
        return sum(path_size_mb(p) for p in self.artifact_files)

    def predict_proba(self, texts) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class RulePredictor(Predictor):
    key = "rule"

    def __init__(self):
        from src.lexicon.matcher import LexiconMatcher

        self.meta = load_json(LEXICON_DIR / "lexicon_meta.json")
        lex = LEXICON_DIR / "offensive_words.txt"
        safe = LEXICON_DIR / "safe_words.txt"
        known = LEXICON_DIR / "known_words.txt"
        self.matcher = LexiconMatcher.from_files(lex, safe, known, **self.meta["matcher_params"])
        self.threshold = float(self.meta["threshold"])
        self.artifact_files = [lex, safe, known]

    def predict_proba(self, texts):
        return np.asarray(self.matcher.predict_proba(texts), dtype=float)


class TfidfPredictor(Predictor):
    key = "tfidf"

    def __init__(self):
        import joblib

        d = MODELS_DIR / "tfidf"
        self.pipe = joblib.load(d / "model.joblib")
        self.meta = load_json(d / "meta.json")
        self.threshold = float(self.meta["threshold"])
        self.artifact_files = [d / "model.joblib"]

    def predict_proba(self, texts):
        return self.pipe.predict_proba([normalize_text(t) for t in texts])[:, 1]


class NNPredictor(Predictor):
    key = "nn"

    def __init__(self):
        import torch

        from src.nn_models import ARCHS

        d = MODELS_DIR / "nn"
        self.cfg = load_json(d / "config.json")
        self.vocab = load_json(d / "vocab.json")
        self.meta = load_json(d / "meta.json")
        cls = ARCHS[self.cfg["arch"]]
        self.model = cls(len(self.vocab), self.cfg["dim"], **self.cfg["arch_kwargs"])
        self.model.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
        self.model.eval()
        self.threshold = float(self.meta["threshold"])
        self.artifact_files = [d / "model.pt", d / "vocab.json"]

    def predict_proba(self, texts, batch_size: int = 256):
        import torch

        from src.nn_models import encode_batch

        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                toks = [simple_tokenize(normalize_text(t)) for t in texts[i:i + batch_size]]
                ids = encode_batch(toks, self.vocab, self.cfg["max_len"], self.model.min_len)
                out.append(torch.softmax(self.model(ids), dim=-1)[:, 1].numpy())
        return np.concatenate(out)


class BertPredictor(Predictor):
    def __init__(self, quantized: bool):
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        self.key = "bert_int8" if quantized else "bert"
        d = MODELS_DIR / ("bertturk_int8" if quantized else "bertturk")
        self.meta = load_json(d / "meta.json")
        trained_with = self.meta.get("normalizer_version")
        if trained_with != NORMALIZER_VERSION:
            raise RuntimeError(
                f"{d.name}: model normalizasyon sürümü '{trained_with}' ile eğitilmiş, kod "
                f"'{NORMALIZER_VERSION}' kullanıyor. Modeli güncel kodla yeniden eğitin "
                f"(04) ve quantize edin (05).")
        self.tokenizer = AutoTokenizer.from_pretrained(d)
        if quantized:
            from src.quantization_utils import count_quantized_linear, quantize_dynamic_int8

            cfg = AutoConfig.from_pretrained(d)
            base = AutoModelForSequenceClassification.from_config(cfg)
            self.model = quantize_dynamic_int8(base)
            # weights_only=False: INT8 state_dict'i paketlenmiş (packed) quantized
            # parametreler içerir; torch>=2.6'nın varsayılan güvenli yükleyicisi bunları
            # reddedebilir. Dosya kendi ürettiğimiz bir artefakt olduğu için güvenilir.
            self.model.load_state_dict(
                torch.load(d / "model_int8.pt", map_location="cpu", weights_only=False))
            n_q, n_fp = count_quantized_linear(self.model)
            if n_q == 0:
                raise RuntimeError("INT8 model yüklendi ama quantized Linear katmanı yok!")
            self.backend = {"int8_linear_layers": n_q, "fp32_linear_layers": n_fp,
                            "engine": torch.backends.quantized.engine}
            self.artifact_files = [d / "model_int8.pt"]
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(d)
            self.backend = {"dtype": str(next(self.model.parameters()).dtype)}
            self.artifact_files = [p for p in d.glob("model*.safetensors")] or list(d.glob("*.bin"))
        self.model.to("cpu").eval()
        self.threshold = float(self.meta["threshold"])

    def predict_proba(self, texts, batch_size: int | None = None):
        """P(offensive).

        INT8 modelde varsayılan batch = 1: dinamik quantization her Linear'da aktivasyon
        ölçeğini O ANKİ tensörün tamamından (dolgu/padding tokenları dahil) hesaplar; bu
        yüzden çıktı, mesajın yanına konan diğer mesajlara bağlı olabilir. Uygulama mesajları
        tek tek işlediği için değerlendirme de tek tek yapılır (raporlanan skor = uygulamadaki
        modelin skoru). FP32'de dolgu dikkat maskesiyle yok sayıldığından batch 32 güvenlidir.
        """
        import torch

        if batch_size is None:
            batch_size = 1 if self.key == "bert_int8" else 32
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                batch = [normalize_text(t, lowercase=False) for t in texts[i:i + batch_size]]
                enc = self.tokenizer(batch, truncation=True, max_length=BERT_MAX_LEN,
                                     padding=True, return_tensors="pt")
                logits = self.model(**enc).logits
                out.append(torch.softmax(logits, dim=-1)[:, 1].numpy())
        return np.concatenate(out)


def load_predictor(key: str) -> Predictor:
    if key == "rule":
        return RulePredictor()
    if key == "tfidf":
        return TfidfPredictor()
    if key == "nn":
        return NNPredictor()
    if key == "bert":
        return BertPredictor(quantized=False)
    if key == "bert_int8":
        return BertPredictor(quantized=True)
    raise KeyError(key)
