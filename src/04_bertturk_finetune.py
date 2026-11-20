"""Aşama 4 — BERTurk ince ayarı (Alan C).

Çalıştırma:  python src/04_bertturk_finetune.py
             (VRAM yetmezse: python src/04_bertturk_finetune.py --batch 8 --accum 4)

Model: dbmdz/bert-base-turkish-cased (sabit revision) + sınıflandırma başlığı.
Girdi: normalize_text(lowercase=False) — leet/tekrar/URL normalizasyonu yapılır
ama büyük-küçük harf korunur (cased model).

Eğitim:
  * AdamW, lr=2e-5, weight decay 0.01, %10 lineer warmup, 3 epoch
  * Etkin batch = batch × accum (varsayılan 16 × 2 = 32); GPU'da AMP (fp16)
  * Sınıf ağırlıklı cross-entropy (03 ile tutarlı)
  * Her epoch sonunda dev macro-F1; en iyi epoch kaydedilir
  * Karar eşiği YALNIZCA dev setinde seçilir

Test seti bu betikte YÜKLENMEZ.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup,
)

from src.common import (  # noqa: E402
    BERT_MAX_LEN, BERT_NAME, BERT_REVISION, ID2LABEL, MODELS_DIR, SEED,
    choose_threshold, evaluate, load_split, save_json, set_seed,
)
from src.lexicon.normalizer import NORMALIZER_VERSION, normalize_text  # noqa: E402


def batches(enc, idx, bs, tok, device):
    for i in range(0, len(idx), bs):
        sel = idx[i:i + bs]
        b = tok.pad({"input_ids": [enc["input_ids"][j] for j in sel],
                     "attention_mask": [enc["attention_mask"][j] for j in sel]},
                    return_tensors="pt")
        yield sel, {k: v.to(device) for k, v in b.items()}


@torch.inference_mode()
def predict(model, enc, tok, device, bs=64) -> np.ndarray:
    model.eval()
    out = []
    for _, b in batches(enc, np.arange(len(enc["input_ids"])), bs, tok, device):
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            logits = model(**b).logits
        out.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--limit", type=int, default=0,
                    help="Hızlı deneme: yalnızca ilk N train örneği (0 = hepsi). Sonuç raporlanmaz.")
    args = ap.parse_args()

    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("UYARI: CUDA bulunamadı. CPU'da eğitim saatler sürebilir. Ctrl+C ile durdurun "
              "ve `python -c \"import torch; print(torch.cuda.is_available())\"` kontrol edin.")
    else:
        print(f"Cihaz: {torch.cuda.get_device_name(0)}  "
              f"VRAM={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("NOT: test seti bu betikte YÜKLENMEZ.")

    X_tr, y_tr = load_split("train")
    X_dv, y_dv = load_split("dev")
    if args.limit:
        print(f"*** DENEME MODU: yalnızca {args.limit} train örneği — sonuçlar raporlanmaz ***")
        X_tr, y_tr = X_tr[:args.limit], y_tr[:args.limit]
    tok = AutoTokenizer.from_pretrained(BERT_NAME, revision=BERT_REVISION)
    enc_tr = tok([normalize_text(t, lowercase=False) for t in X_tr],
                 truncation=True, max_length=BERT_MAX_LEN)
    enc_dv = tok([normalize_text(t, lowercase=False) for t in X_dv],
                 truncation=True, max_length=BERT_MAX_LEN)
    lens = np.array([len(x) for x in enc_tr["input_ids"]])
    trunc = float((lens >= BERT_MAX_LEN).mean())
    print(f"Token uzunluğu: medyan={np.median(lens):.0f} p95={np.percentile(lens, 95):.0f} "
          f"max={lens.max()}  kesilen oran={trunc:.4f}")

    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_NAME, revision=BERT_REVISION, num_labels=2,
        id2label=ID2LABEL, label2id={v: k for k, v in ID2LABEL.items()},
    ).to(device)

    counts = np.bincount(y_tr, minlength=2)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(len(y_tr) / (2 * counts), dtype=torch.float32, device=device))
    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)],
         "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=args.lr)
    steps_per_epoch = math.ceil(len(y_tr) / (args.batch * args.accum))
    total_steps = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    out = MODELS_DIR / "bertturk"
    rng = np.random.default_rng(SEED)
    best_f1, hist, t_start = -1.0, [], time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run_loss, n_seen = time.time(), 0.0, 0
        order = rng.permutation(len(y_tr))
        opt.zero_grad()
        for step, (sel, b) in enumerate(batches(enc_tr, order, args.batch, tok, device), 1):
            yb = torch.as_tensor(y_tr[sel], device=device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                loss = loss_fn(model(**b).logits.float(), yb) / args.accum
            scaler.scale(loss).backward()
            if step % args.accum == 0 or step * args.batch >= len(order):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()
            run_loss += loss.item() * args.accum * len(sel)
            n_seen += len(sel)
            if step % 200 == 0:
                el = time.time() - t0
                eta = el / n_seen * (len(order) - n_seen)
                mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
                print(f"  epoch {ep} adım {step}  loss={run_loss / n_seen:.4f}  "
                      f"{n_seen}/{len(order)}  kalan≈{eta / 60:.1f} dk  max VRAM={mem:.2f} GB")
        p = predict(model, enc_dv, tok, device)
        f1 = evaluate(y_dv, p, 0.5)["macro_f1"]
        hist.append({"epoch": ep, "train_loss": round(run_loss / n_seen, 4),
                     "dev_macro_f1@0.5": round(f1, 4), "min": round((time.time() - t0) / 60, 1)})
        print(f"[epoch {ep}] loss={hist[-1]['train_loss']}  dev macro-F1@0.5={f1:.4f}  "
              f"({hist[-1]['min']} dk)")
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(out)
            tok.save_pretrained(out)
            print(f"  ↳ en iyi model kaydedildi: {out}")

    # En iyi epoch'u yükle, eşiği dev'de seç
    model = AutoModelForSequenceClassification.from_pretrained(out).to(device)
    p = predict(model, enc_dv, tok, device)
    thr = choose_threshold(y_dv, p)
    ev = evaluate(y_dv, p, thr)
    save_json({
        "model": BERT_NAME, "revision": BERT_REVISION, "threshold": thr, "dev_metrics": ev,
        "history": hist, "best_epoch": int(np.argmax([h["dev_macro_f1@0.5"] for h in hist]) + 1),
        "hyperparams": {**vars(args), "effective_batch": args.batch * args.accum,
                        "max_len": BERT_MAX_LEN, "warmup": 0.1, "weight_decay": 0.01,
                        "loss": "class-weighted cross-entropy", "amp_fp16": device == "cuda"},
        "input_normalization": "normalize_text(lowercase=False): normal yazım korunur, tamamı büyük harf tokenlar küçültülür",
        "normalizer_version": NORMALIZER_VERSION,
        "truncated_ratio_train": trunc, "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "train_minutes": round((time.time() - t_start) / 60, 1),
        "selected_on": "dev", "test_set_used": False, "trial_limit": args.limit,
    }, out / "meta.json")
    print(f"\nBERTurk dev macro-F1={ev['macro_f1']:.4f}  eşik={thr}  "
          f"toplam süre={(time.time() - t_start) / 60:.1f} dk")


if __name__ == "__main__":
    main()
