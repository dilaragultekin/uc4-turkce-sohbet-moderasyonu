"""Aşama 5 — BERTurk'ü dinamik INT8 ile quantize et ve doğrula (Alan D).

Çalıştırma:  python src/05_quantization.py

Adımlar (tamamı CPU'da; GPU kasıtlı olarak gizlenir):
  1. models/bertturk (FP32) yüklenir, tüm nn.Linear katmanları INT8'e çevrilir.
  2. GERÇEK backend doğrulanır: kaç Linear katmanının quantized olduğu sayılır.
  3. models/bertturk_int8/ altına kaydedilir (state_dict + config + tokenizer).
  4. Kaydedilen model, uygulamanın kullandığı yükleyiciyle (BertPredictor)
     tekrar yüklenir ve bellekteki modelle aynı çıktıyı verdiği kontrol edilir.
  5. Dev setinde FP32 ↔ INT8 karşılaştırması: tahmin uyumu, olasılık farkı,
     macro-F1. INT8 eşiği yine YALNIZCA dev'de seçilir.

Test seti bu betikte YÜKLENMEZ. Resmi gecikme ölçümü 06_benchmark.py'dedir.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # torch import edilmeden önce: yalnızca CPU

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from src.common import (  # noqa: E402
    CPU_THREADS, MODELS_DIR, choose_threshold, evaluate, load_json, load_split,
    path_size_mb, save_json, set_seed,
)
from src.predictors import BertPredictor  # noqa: E402
from src.quantization_utils import count_quantized_linear, quantize_dynamic_int8  # noqa: E402


def main() -> None:
    set_seed()
    torch.set_num_threads(CPU_THREADS)
    src_dir, out = MODELS_DIR / "bertturk", MODELS_DIR / "bertturk_int8"
    if not (src_dir / "meta.json").exists():
        raise SystemExit("Önce `python src/04_bertturk_finetune.py` çalıştırın.")
    print(f"torch {torch.__version__}  CUDA görünür mü: {torch.cuda.is_available()}  "
          f"CPU thread: {torch.get_num_threads()}")
    print("NOT: test seti bu betikte YÜKLENMEZ.")

    fp32 = AutoModelForSequenceClassification.from_pretrained(src_dir).eval()
    tok = AutoTokenizer.from_pretrained(src_dir)
    int8 = quantize_dynamic_int8(fp32)
    n_q, n_fp = count_quantized_linear(int8)
    print(f"INT8 Linear katman: {n_q}   FP32 kalan Linear: {n_fp}   "
          f"engine: {torch.backends.quantized.engine}")
    if n_q == 0:
        raise SystemExit("HATA: quantization uygulanmadı.")

    out.mkdir(parents=True, exist_ok=True)
    torch.save(int8.state_dict(), out / "model_int8.pt")
    fp32.config.save_pretrained(out)
    tok.save_pretrained(out)
    save_json({"threshold": 0.5, "status": "geçici",  # aşağıda güncellenir
               "normalizer_version": load_json(src_dir / "meta.json").get("normalizer_version")},
              out / "meta.json")

    fp32_file = next(iter(src_dir.glob("model*.safetensors")), None) or next(src_dir.glob("*.bin"))
    size_fp32, size_int8 = path_size_mb(fp32_file), path_size_mb(out / "model_int8.pt")
    print(f"Boyut: FP32={size_fp32:.1f} MB  INT8={size_int8:.1f} MB  "
          f"(oran {size_int8 / size_fp32:.2f}; embedding katmanları FP32 kalır)")

    # Dev karşılaştırması (CPU)
    X_dv, y_dv = load_split("dev")
    fp32_pred = BertPredictor(quantized=False)
    t0 = time.time()
    p_fp32 = fp32_pred.predict_proba(X_dv)
    t_fp32 = time.time() - t0

    loaded = BertPredictor(quantized=True)  # uygulamanın kullandığı yol

    # Kaydet-yükle tutarlılığı: bellekteki ve diskten yüklenen INT8'e AYNI girdi tensörü
    with torch.inference_mode():
        from src.lexicon.normalizer import normalize_text
        sample = [normalize_text(t, lowercase=False) for t in X_dv[:16]]
        enc = tok(sample, truncation=True, max_length=128, padding=True, return_tensors="pt")
        p_mem = torch.softmax(int8(**enc).logits, -1)[:, 1].numpy()
        p_load = torch.softmax(loaded.model(**enc).logits, -1)[:, 1].numpy()
    reload_diff = float(np.abs(p_mem - p_load).max())
    print(f"Kaydet/yükle farkı (aynı girdi, maks |Δp|, 16 örnek): {reload_diff:.2e}")
    if reload_diff > 1e-4:
        raise SystemExit("HATA: diskten yüklenen INT8 model bellekteki modelle aynı değil.")

    # Dev tahminleri: uygulamadaki gibi mesaj mesaj (batch = 1)
    t0 = time.time()
    p_int8 = loaded.predict_proba(X_dv)            # varsayılan batch = 1
    t_int8 = time.time() - t0
    # Tanı: aynı model, 32'lik gruplarla — dinamik quantization'ın batch'e bağımlılığı
    t0 = time.time()
    p_int8_b32 = loaded.predict_proba(X_dv, batch_size=32)
    t_int8_b32 = time.time() - t0
    bdiff = np.abs(p_int8 - p_int8_b32)
    print(f"INT8 batch etkisi (batch 1 vs 32): |Δp| ortalama={bdiff.mean():.4f} "
          f"maks={bdiff.max():.4f}  (raporlanan değerler batch 1 ile, uygulamadaki gibi)")

    thr_fp32 = float(load_json(src_dir / "meta.json")["threshold"])
    thr_int8 = choose_threshold(y_dv, p_int8)
    ev_fp32 = evaluate(y_dv, p_fp32, thr_fp32)
    ev_int8 = evaluate(y_dv, p_int8, thr_int8)
    agree = float(np.mean((p_fp32 >= thr_fp32) == (p_int8 >= thr_int8)))
    diff = np.abs(p_fp32 - p_int8)
    print(f"Dev macro-F1  FP32={ev_fp32['macro_f1']:.4f} (eşik {thr_fp32})   "
          f"INT8={ev_int8['macro_f1']:.4f} (eşik {thr_int8})")
    print(f"Tahmin uyumu={agree:.4f}  |Δp| ortalama={diff.mean():.4f} maks={diff.max():.4f}")
    print(f"Dev çıkarım süresi (CPU, {CPU_THREADS} thread, bilgi amaçlı): FP32 batch32={t_fp32:.1f}s  "
          f"INT8 batch32={t_int8_b32:.1f}s  INT8 batch1={t_int8:.1f}s")
    print("Not: Resmî gecikme ölçümü 06_benchmark.py'de (mesaj başına, 200 çalıştırma).")

    save_json({
        "method": "torch.ao.quantization.quantize_dynamic, nn.Linear → qint8",
        "normalizer_version": load_json(src_dir / "meta.json").get("normalizer_version"),
        "engine": torch.backends.quantized.engine, "torch": torch.__version__,
        "int8_linear_layers": n_q, "fp32_linear_layers": n_fp,
        "kept_fp32": "embeddings, LayerNorm",
        "threshold": thr_int8, "dev_metrics": ev_int8, "dev_metrics_fp32": ev_fp32,
        "dev_agreement_with_fp32": agree, "dev_mean_abs_prob_diff": float(diff.mean()),
        "save_reload_max_abs_diff_same_input": reload_diff,
        "eval_batch_size_int8": 1,
        "batch_effect_int8": {"mean_abs_prob_diff_b1_vs_b32": float(bdiff.mean()),
                              "max_abs_prob_diff_b1_vs_b32": float(bdiff.max()),
                              "reason": "dinamik aktivasyon ölçeği dolgu tokenlarını da içerir"},
        "size_mb": {"fp32": size_fp32, "int8": size_int8},
        "dev_seconds_cpu": {"fp32_batch32": t_fp32, "int8_batch32": t_int8_b32,
                            "int8_batch1": t_int8, "threads": CPU_THREADS},
        "source": str(src_dir.relative_to(ROOT)), "selected_on": "dev", "test_set_used": False,
    }, out / "meta.json")
    print(f"Kaydedildi: {out}")


if __name__ == "__main__":
    main()
