"""Aşama 6 — FİNAL değerlendirme: tüm modeller, AYNI test seti, CPU (R3, R4).

Çalıştırma:  python src/06_benchmark.py

Test setinin kullanıldığı TEK yer burasıdır. Her model:
  * macro-F1 (ve OFF sınıfı precision/recall), dev'de seçilmiş eşikle
  * boyut (MB, diskteki ağırlık dosyaları)
  * gecikme: 200 farklı test mesajı, tek tek (batch=1), uçtan uca
    (normalizasyon + tokenizasyon + model), 10 ısınma çalıştırmasından sonra;
    ortalama / medyan / p95 ms
  * gecikme bütçesi: p95 <= LATENCY_BUDGET_MS

GPU kasıtlı olarak gizlenir; tüm sayılar CPU'dur. Donanım bilgisi kaydedilir.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import csv  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.common import (  # noqa: E402
    CPU_THREADS, LATENCY_BUDGET_MS, RESULTS_DIR, evaluate, load_split, save_json,
)
from src.model_store import ensure_models  # noqa: E402
from src.predictors import DISPLAY_NAMES, MODEL_KEYS, load_predictor  # noqa: E402

N_LATENCY = 200
N_WARMUP = 10
N_FAILURES = 15


def cpu_name() -> str:
    try:
        if platform.system() == "Windows":
            r = subprocess.run(["powershell", "-NoProfile", "-Command",
                                "(Get-CimInstance Win32_Processor).Name"],
                               capture_output=True, text=True, timeout=20)
            if r.stdout.strip():
                return r.stdout.strip()
    except Exception:
        pass
    return platform.processor() or "bilinmiyor"


def hardware() -> dict:
    return {"cpu": cpu_name(), "logical_cpus": os.cpu_count(), "torch_threads": CPU_THREADS,
            "os": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "cuda_visible": torch.cuda.is_available()}


def latency(pred, texts) -> dict:
    for t in texts[:N_WARMUP]:
        pred.predict_proba([t])
    ms = []
    for t in texts[N_WARMUP:N_WARMUP + N_LATENCY]:
        s = time.perf_counter()
        pred.predict_proba([t])
        ms.append((time.perf_counter() - s) * 1000)
    ms = np.array(ms)
    return {"runs": int(len(ms)), "mean_ms": float(ms.mean()), "median_ms": float(np.median(ms)),
            "p95_ms": float(np.percentile(ms, 95)), "max_ms": float(ms.max())}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    torch.set_num_threads(CPU_THREADS)
    hw = hardware()
    print("Donanım:", hw)
    X_te, y_te = load_split("test")
    print(f"Test seti: n={len(X_te)}  OFF={int(y_te.sum())}")
    rng = np.random.default_rng(0)
    lat_idx = rng.choice(len(X_te), N_WARMUP + N_LATENCY, replace=False)
    lat_texts = [X_te[i] for i in lat_idx]

    # Model dosyaları yerelde yoksa Hub'dan indirilir (taze klon); varsa ağa çıkılmaz
    dl_errors = ensure_models(MODEL_KEYS)

    rows, details, probs, missing = [], {}, {}, []
    for key in MODEL_KEYS:
        print(f"\n== {DISPLAY_NAMES[key]}")
        if key in dl_errors:
            print(f"  EKSİK: indirilemedi — {dl_errors[key]}")
            missing.append({"model": key, "error": f"indirilemedi — {dl_errors[key]}"})
            continue
        try:
            pred = load_predictor(key)
        except Exception as e:  # eksik model sessizce atlanmaz, raporlanır
            print(f"  EKSİK/HATA: {type(e).__name__}: {e}")
            missing.append({"model": key, "error": f"{type(e).__name__}: {e}"})
            continue
        t0 = time.time()
        p = pred.predict_proba(X_te)
        ev = evaluate(y_te, p, pred.threshold)
        lat = latency(pred, lat_texts)
        size = pred.size_mb()
        probs[key] = p
        details[key] = {"test_metrics": ev, "latency": lat, "size_mb": size,
                        "threshold": pred.threshold, "eval_sec": round(time.time() - t0, 1),
                        "backend": getattr(pred, "backend", None),
                        "dev_macro_f1": pred.meta.get("dev_metrics", {}).get("macro_f1")}
        rows.append({
            "model": DISPLAY_NAMES[key], "key": key,
            "test_macro_f1": round(ev["macro_f1"], 4),
            "test_f1_off": round(ev["f1_off"], 4),
            "test_precision_off": round(ev["precision_off"], 4),
            "test_recall_off": round(ev["recall_off"], 4),
            "dev_macro_f1": round(details[key]["dev_macro_f1"] or float("nan"), 4),
            "threshold": pred.threshold,
            "size_mb": round(size, 2),
            "latency_mean_ms": round(lat["mean_ms"], 2),
            "latency_p95_ms": round(lat["p95_ms"], 2),
            "budget_ms": LATENCY_BUDGET_MS,
            "budget_met_p95": lat["p95_ms"] <= LATENCY_BUDGET_MS,
        })
        print(f"  macro-F1={ev['macro_f1']:.4f}  boyut={size:.1f} MB  "
              f"gecikme ort={lat['mean_ms']:.2f} ms p95={lat['p95_ms']:.2f} ms  "
              f"backend={details[key]['backend']}")

    if not rows:
        raise SystemExit("Hiç model yüklenemedi.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS_DIR / "results_table.csv", rows)

    # Sunum için markdown tablo
    md = ["| Model | Test macro-F1 | Boyut (MB) | Gecikme ort. (ms) | Gecikme p95 (ms) | Bütçe (p95 ≤ "
          f"{LATENCY_BUDGET_MS:.0f} ms) |", "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['test_macro_f1']:.4f} | {r['size_mb']:.1f} | "
                  f"{r['latency_mean_ms']:.2f} | {r['latency_p95_ms']:.2f} | "
                  f"{'✓' if r['budget_met_p95'] else '✗'} |")
    md.append(f"\nDonanım: {hw['cpu']}, {hw['torch_threads']} CPU thread, {hw['os']}; "
              f"gecikme = {N_LATENCY} tek mesaj çalıştırmasının ortalaması "
              f"({N_WARMUP} ısınma sonrası), uçtan uca.")
    (RESULTS_DIR / "results_table.md").write_text("\n".join(md), encoding="utf-8")

    # Mesaj bazında tahminler + hata örnekleri (en emin hatalar, kural tanımlı)
    pred_rows = []
    for i, (t, y) in enumerate(zip(X_te, y_te)):
        r = {"idx": i, "text": t, "label": int(y)}
        r.update({f"p_{k}": round(float(v[i]), 4) for k, v in probs.items()})
        pred_rows.append(r)
    write_csv(RESULTS_DIR / "test_predictions.csv", pred_rows)
    fails = []
    main_key = "bert_int8" if "bert_int8" in probs else next(iter(probs))
    thr = details[main_key]["threshold"]
    p = probs[main_key]
    for kind, mask in (("false_positive", (p >= thr) & (y_te == 0)),
                       ("false_negative", (p < thr) & (y_te == 1))):
        idx = np.where(mask)[0]
        conf = np.abs(p[idx] - thr)
        for i in idx[np.argsort(-conf)][:N_FAILURES]:
            fails.append({"model": main_key, "type": kind, "idx": int(i),
                          "p_off": round(float(p[i]), 4), "text": X_te[i]})
    if fails:
        write_csv(RESULTS_DIR / "failure_examples.csv", fails)

    save_json({"hardware": hw, "latency_protocol": {
                   "runs": N_LATENCY, "warmup": N_WARMUP, "batch_size": 1,
                   "end_to_end": True, "sample_seed": 0, "budget_ms": LATENCY_BUDGET_MS,
                   "budget_criterion": "p95"},
               "models": details, "missing_models": missing, "test_n": int(len(y_te))},
              RESULTS_DIR / "results.json")
    print("\n" + "\n".join(md))
    if missing:
        print("\nUYARI — eksik modeller (R3 için hepsi gerekli):", [m["model"] for m in missing])
    print("Kaydedildi: results/results_table.{csv,md}, results.json, test_predictions.csv, "
          "failure_examples.csv")


if __name__ == "__main__":
    main()
