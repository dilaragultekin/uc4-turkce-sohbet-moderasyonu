"""Aşama 2 — Klasik NLP baseline'ları (Alan B).

Çalıştırma:  python src/02_classical_baseline.py

  1) Kural baseline: build_lexicon.py'nin ürettiği sözlük + edit-distance
     eşleştirici. Burada yalnızca dev sonucu yeniden raporlanır.
  2) TF-IDF + Lojistik Regresyon: kelime (1-2 gram) ve/veya karakter
     (2-5 gram, char_wb) özellikleri; class_weight="balanced" (sınıf
     dengesizliği). Özellik tipi ve C YALNIZCA dev macro-F1 ile seçilir;
     karar eşiği de dev'de seçilir. Model yalnızca train ile eğitilir.

Test seti bu betikte YÜKLENMEZ — final değerlendirme 06_benchmark.py'de.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import FeatureUnion, Pipeline  # noqa: E402

from src.common import (  # noqa: E402
    LEXICON_DIR, MODELS_DIR, SEED, choose_threshold, evaluate, load_json, load_split,
    save_json, set_seed,
)
from src.lexicon.normalizer import normalize_text  # noqa: E402

C_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]
FEATURES = ["word", "char", "word+char"]


def make_pipeline(features: str, C: float) -> Pipeline:
    parts = []
    if "word" in features:
        parts.append(("word", TfidfVectorizer(analyzer="word", token_pattern=r"(?u)[\w@]+",
                                              ngram_range=(1, 2), min_df=2, sublinear_tf=True)))
    if "char" in features:
        parts.append(("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                              min_df=3, sublinear_tf=True, max_features=300_000)))
    return Pipeline([
        ("features", FeatureUnion(parts)),
        ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=3000,
                                   solver="liblinear", random_state=SEED)),
    ])


def main() -> None:
    set_seed()
    X_tr, y_tr = load_split("train")
    X_dv, y_dv = load_split("dev")
    print("NOT: test seti bu betikte YÜKLENMEZ.")

    # 1) Kural baseline (dev) ──────────────────────────────────────────
    lex_meta_path = LEXICON_DIR / "lexicon_meta.json"
    if not lex_meta_path.exists():
        raise SystemExit("Önce `python src/lexicon/build_lexicon.py` çalıştırın.")
    lex = load_json(lex_meta_path)
    print(f"[Kural] sözlük={lex['lexicon_size']} girdi  dev macro-F1="
          f"{lex['dev_metrics']['macro_f1']:.4f}")

    # 2) TF-IDF + LR ───────────────────────────────────────────────────
    Xn_tr = [normalize_text(t) for t in X_tr]
    Xn_dv = [normalize_text(t) for t in X_dv]
    grid, best = [], None
    for feats in FEATURES:
        for C in C_GRID:
            t0 = time.time()
            pipe = make_pipeline(feats, C).fit(Xn_tr, y_tr)
            p = pipe.predict_proba(Xn_dv)[:, 1]
            thr = choose_threshold(y_dv, p)
            ev = evaluate(y_dv, p, thr)
            row = {"features": feats, "C": C, "threshold": thr,
                   "dev_macro_f1": round(ev["macro_f1"], 4),
                   "dev_macro_f1_at_0.5": round(evaluate(y_dv, p, 0.5)["macro_f1"], 4),
                   "fit_sec": round(time.time() - t0, 1)}
            grid.append(row)
            print("  ", row)
            if best is None or ev["macro_f1"] > best[0]:
                best = (ev["macro_f1"], feats, C, thr, ev, pipe)

    _, feats, C, thr, ev, pipe = best
    out = MODELS_DIR / "tfidf"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "model.joblib")
    n_feat = len(pipe.named_steps["features"].get_feature_names_out())
    save_json({"model": "tfidf+logreg", "features": feats, "C": C, "threshold": thr,
               "n_features": n_feat, "dev_metrics": ev, "grid": grid,
               "trained_on": "train", "selected_on": "dev", "test_set_used": False},
              out / "meta.json")
    print(f"\n[TF-IDF] seçilen: features={feats} C={C} eşik={thr}  "
          f"dev macro-F1={ev['macro_f1']:.4f}  özellik sayısı={n_feat}")
    print(f"Kaydedildi: {out}")


if __name__ == "__main__":
    main()
