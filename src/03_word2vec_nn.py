"""Aşama 3 — Kendi Word2Vec'imiz + sinir ağı (Alan A).

Çalıştırma:  python src/03_word2vec_nn.py

  1) Word2Vec (skip-gram, negatif örnekleme) YALNIZCA train mesajlarıyla eğitilir.
  2) Gömmeler iki mimariye başlangıç ağırlığı olarak verilir (eğitimde ince ayar):
       - TextCNN (1D evrişim, çekirdek 2/3/4, max-pooling, dropout 0.5)
       - MeanFF  (ortalama gömme → 128 gizli birim → çıktı, dropout 0.5)
  3) Eğitim: Adam (stokastik gradyan + geri yayılım), sınıf ağırlıklı
     cross-entropy (dengesizlik), dev macro-F1 ile erken durdurma.
  4) Mimari ve karar eşiği YALNIZCA dev setinde seçilir.

Test seti bu betikte YÜKLENMEZ.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from gensim.models import Word2Vec  # noqa: E402
from torch import nn  # noqa: E402

from src.common import (  # noqa: E402
    MODELS_DIR, SEED, choose_threshold, evaluate, load_split, save_json, set_seed,
    simple_tokenize,
)
from src.lexicon.normalizer import normalize_text  # noqa: E402
from src.nn_models import ARCHS, PAD, UNK, encode_batch  # noqa: E402

W2V = dict(vector_size=200, window=5, min_count=2, sg=1, negative=10, epochs=15)
MAX_LEN = 96
BATCH = 64
LR = 1e-3
MAX_EPOCHS = 15
PATIENCE = 3
ARCH_KWARGS = {
    "cnn": {"num_filters": 100, "kernel_sizes": [2, 3, 4], "dropout": 0.5},
    "ff": {"hidden": 128, "dropout": 0.5},
}
PROBES = ["salak", "aptal", "güzel", "maç", "teşekkürler"]


def predict(model, toks, vocab, device, bs=512) -> np.ndarray:
    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(toks), bs):
            ids = encode_batch(toks[i:i + bs], vocab, MAX_LEN, model.min_len).to(device)
            out.append(torch.softmax(model(ids), -1)[:, 1].float().cpu().numpy())
    return np.concatenate(out)


def train_arch(arch, weights, vocab, tr_toks, y_tr, dv_toks, y_dv, device):
    set_seed()
    model = ARCHS[arch](len(vocab), weights.shape[1], weights, **ARCH_KWARGS[arch]).to(device)
    counts = np.bincount(y_tr, minlength=2)
    cw = torch.tensor(len(y_tr) / (2 * counts), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(SEED)
    best_f1, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        t0, total = time.time(), 0.0
        order = rng.permutation(len(tr_toks))
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            ids = encode_batch([tr_toks[j] for j in idx], vocab, MAX_LEN, model.min_len).to(device)
            yb = torch.as_tensor(y_tr[idx], device=device)
            opt.zero_grad()
            loss = loss_fn(model(ids), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        p = predict(model, dv_toks, vocab, device)
        f1 = evaluate(y_dv, p, 0.5)["macro_f1"]
        hist.append({"epoch": ep, "train_loss": round(total / len(order), 4),
                     "dev_macro_f1@0.5": round(f1, 4), "sec": round(time.time() - t0, 1)})
        print(f"  [{arch}] epoch {ep:2d}  loss={hist[-1]['train_loss']:.4f}  "
              f"dev macro-F1@0.5={f1:.4f}  ({hist[-1]['sec']}s)")
        if f1 > best_f1:
            best_f1, bad = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"  [{arch}] erken durdurma (sabır={PATIENCE})")
                break
    model.load_state_dict(best_state)
    p = predict(model, dv_toks, vocab, device)
    thr = choose_threshold(y_dv, p)
    return model.cpu(), evaluate(y_dv, p, thr), hist


def main() -> None:
    global MAX_EPOCHS
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="Hızlı deneme: yalnızca ilk N train örneği (0 = hepsi). Sonuç raporlanmaz.")
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    args = ap.parse_args()
    MAX_EPOCHS = args.max_epochs

    set_seed()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cihaz: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print("NOT: test seti bu betikte YÜKLENMEZ.")

    X_tr, y_tr = load_split("train")
    X_dv, y_dv = load_split("dev")
    if args.limit:
        print(f"*** DENEME MODU: yalnızca {args.limit} train örneği — sonuçlar raporlanmaz ***")
        X_tr, y_tr = X_tr[:args.limit], y_tr[:args.limit]
    tr_toks = [simple_tokenize(normalize_text(t)) for t in X_tr]
    dv_toks = [simple_tokenize(normalize_text(t)) for t in X_dv]

    # 1) Word2Vec (yalnızca train) ─────────────────────────────────────
    t0 = time.time()
    w2v = Word2Vec(sentences=tr_toks, seed=SEED, workers=1, **W2V)
    print(f"Word2Vec: sözlük={len(w2v.wv)}  boyut={w2v.vector_size}  "
          f"({time.time() - t0:.0f}s)")
    neighbours = {}
    for w in PROBES:
        if w in w2v.wv:
            neighbours[w] = [(n, round(float(s), 3)) for n, s in w2v.wv.most_similar(w, topn=5)]
            print(f"  {w:12s} → {[n for n, _ in neighbours[w]]}")
    (MODELS_DIR / "w2v").mkdir(parents=True, exist_ok=True)
    w2v.save(str(MODELS_DIR / "w2v" / "w2v.model"))

    vocab = {PAD: 0, UNK: 1}
    for w in w2v.wv.index_to_key:
        vocab[w] = len(vocab)
    weights = np.zeros((len(vocab), w2v.vector_size), dtype=np.float32)
    weights[1] = w2v.wv.vectors.mean(0)
    weights[2:] = w2v.wv.vectors
    oov = np.mean([t not in vocab for toks in dv_toks for t in toks])
    print(f"Dev OOV oranı (token): {oov:.3f}")

    # 2) İki mimari, dev ile seçim ─────────────────────────────────────
    results = {}
    for arch in ("cnn", "ff"):
        model, ev, hist = train_arch(arch, weights, vocab, tr_toks, y_tr, dv_toks, y_dv, device)
        results[arch] = (model, ev, hist)
        print(f"  [{arch}] dev macro-F1={ev['macro_f1']:.4f} (eşik={ev['threshold']})")
    arch = max(results, key=lambda a: results[a][1]["macro_f1"])
    model, ev, _ = results[arch]

    out = MODELS_DIR / "nn"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    save_json(vocab, out / "vocab.json")
    save_json({"arch": arch, "dim": int(w2v.vector_size), "max_len": MAX_LEN,
               "arch_kwargs": ARCH_KWARGS[arch]}, out / "config.json")
    save_json({
        "model": f"word2vec+{arch}", "threshold": ev["threshold"], "dev_metrics": ev,
        "word2vec": {**W2V, "vocab_size": len(w2v.wv), "trained_on": "train"},
        "dev_oov_rate": float(oov), "neighbours": neighbours,
        "training": {"optimizer": "Adam", "lr": LR, "batch": BATCH, "max_epochs": MAX_EPOCHS,
                     "patience": PATIENCE, "loss": "class-weighted cross-entropy",
                     "device": device},
        "candidates": {a: {"dev_metrics": r[1], "history": r[2]} for a, r in results.items()},
        "selected_on": "dev", "test_set_used": False, "trial_limit": args.limit,
    }, out / "meta.json")
    print(f"\nSeçilen mimari: {arch}  dev macro-F1={ev['macro_f1']:.4f}  eşik={ev['threshold']}")
    print(f"Kaydedildi: {out}")


if __name__ == "__main__":
    main()
