"""Sözlük oluşturma: belgelenmiş, tekrarlanabilir, TEST SETİNE DOKUNMAYAN prosedür.

Çalıştırma:  python src/lexicon/build_lexicon.py

Kaynaklar
  E) Dış liste: ooguz/turkce-kufur-karaliste (CC-BY-SA-4.0), sabit commit.
  D) Veriden türetilen adaylar: YALNIZCA eğitim (train) setindeki kelime
     istatistikleri.

Prosedür
  1. E listesindeki her girdinin train setindeki desteği (kaç mesajda geçtiği)
     ve kesinliği P(OFF | girdi geçiyor) ölçülür.
       - destek >= MIN_SUPPORT ve kesinlik >= p_ext  → tut
       - destek >= MIN_SUPPORT ve kesinlik <  p_ext  → AT (ör. "allah", "ana")
       - destek <  MIN_SUPPORT (kanıt yok) → uzunluk >= 4 ise tut, kısaysa at
  2. D adayları: train'de en az dd_min_off OFF mesajında geçen ve
     yumuşatılmış kesinliği >= p_dd olan kelimeler.
  3. "Güvenli kelimeler": train'de sık geçen (>= SAFE_MIN_COUNT) ve
     kesinliği < SAFE_MAX_PREC olan kelimeler önek/bulanık eşleşmeden muaf.
     "Bilinen kelimeler": train'de >= KNOWN_MIN_COUNT kez geçen, sözlükte
     olmayan kelimeler yalnızca bulanık eşleşmeden muaf (non-word ilkesi).
  4. Hiperparametreler (p_ext, D kullanımı, dd_min_off, p_dd, bulanık
     politika) YALNIZCA dev setinde macro-F1 ile seçilir. Eşitlikte daha
     küçük sözlük tercih edilir (açıklanabilirlik).
     UC4 edit-distance eşleştirici istediği için seçim yalnızca bulanık
     eşleşmesi AÇIK yapılandırmalar arasından yapılır; "kapalı" satırlar
     yalnızca karşılaştırma (ablation) için raporlanır.
  5. Sonuçlar ve her girdinin gerekçesi CSV/JSON olarak kaydedilir.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common import (  # noqa: E402
    DATA_RAW, EXT_LEXICON_URL, LEXICON_DIR, RESULTS_DIR,
    evaluate, load_split, save_json,
)
from src.lexicon.matcher import PLACEHOLDERS, WORD_RE, LexiconMatcher  # noqa: E402
from src.lexicon.normalizer import match_key, normalize_token  # noqa: E402

MIN_SUPPORT = 3
SAFE_MIN_COUNT = 20
SAFE_MAX_PREC = 0.30
KNOWN_MIN_COUNT = 3
SMOOTH_A = 2.0          # yumuşatma gücü (öncül = train'deki OFF oranı)

GRID = {
    "p_ext": [0.4, 0.5, 0.6],
    "dd": [None, (5, 0.70), (5, 0.85), (10, 0.70), (10, 0.85), (20, 0.70)],
    "use_ext": [True, False],
    # (fuzzy_min_len, two_edit_min_len); None = kapalı (yalnızca ablation)
    "fuzzy": [None, (5, 8), (6, 9), (7, 99)],
}

OUT_RESULTS = RESULTS_DIR / "lexicon"


def download_external() -> tuple[list[str], str]:
    path = DATA_RAW / "karaliste.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"İndiriliyor: {EXT_LEXICON_URL}")
        with urllib.request.urlopen(EXT_LEXICON_URL, timeout=60) as r:
            path.write_bytes(r.read())
    raw = path.read_bytes()
    lines = raw.decode("utf-8").splitlines()
    entries = sorted({normalize_token(l.strip()) for l in lines if l.strip()})
    return entries, hashlib.sha256(raw).hexdigest()


EXCLUSIONS_PATH = LEXICON_DIR / "manual_exclusions.txt"


def load_exclusions() -> dict[str, tuple[str, str]]:
    """manual_exclusions.txt → {eşleştirme anahtarı: (kelime, gerekçe)}."""
    out = {}
    if not EXCLUSIONS_PATH.exists():
        return out
    for line in EXCLUSIONS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        word, _, reason = line.partition("\t")
        out[match_key(normalize_token(word.strip()))] = (word.strip(), reason.strip())
    return out


def single_key(entry: str) -> str | None:
    """Tek kelimelik girdinin eşleştirme anahtarı; çok kelimeliyse None."""
    keys = [match_key(w) for w in WORD_RE.findall(normalize_token(entry))]
    return keys[0] if len(keys) == 1 else None


def message_keys(text: str) -> set[str]:
    keys = set()
    for chunk in text.split():
        for w in WORD_RE.findall(normalize_token(chunk)):
            k = match_key(w)
            if len(k) >= 2 and k not in PLACEHOLDERS:
                keys.add(k)
    return keys


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    t0 = time.time()
    X_tr, y_tr = load_split("train")
    X_dv, y_dv = load_split("dev")
    prior = float(y_tr.mean())
    print(f"train={len(X_tr)}  dev={len(X_dv)}  train OFF oranı={prior:.3f}")
    print("NOT: test seti bu betikte YÜKLENMEZ.")

    # ── 1. Dış liste denetimi (train) ──────────────────────────────────
    ext, ext_sha = download_external()
    print(f"Dış liste: {len(ext)} benzersiz girdi (sha256={ext_sha[:12]}…)")
    excl = load_exclusions()
    print(f"Manuel dışlama listesi: {len(excl)} kelime ({EXCLUSIONS_PATH.name})")

    def is_excluded(entry: str) -> bool:
        return single_key(entry) in excl
    audit_matcher = LexiconMatcher(ext, fuzzy_min_len=None)
    sup, off = Counter(), Counter()
    for text, y in zip(X_tr, y_tr):
        for e in {m.entry for m in audit_matcher.find(text).matches}:
            sup[e] += 1
            off[e] += int(y)
    ext_audit = []
    for e in ext:
        s, o = sup[e], off[e]
        ext_audit.append({"entry": e, "length": len(match_key(e)), "train_support": s,
                          "train_off": o, "train_precision": round(o / s, 4) if s else ""})

    def ext_keep(p_ext: float) -> list[str]:
        keep = []
        for r in ext_audit:
            s = r["train_support"]
            if is_excluded(r["entry"]):
                continue
            if s >= MIN_SUPPORT:
                if r["train_precision"] >= p_ext:
                    keep.append(r["entry"])
            elif r["length"] >= 4:
                keep.append(r["entry"])
        return keep

    # ── 2. Train kelime istatistikleri ─────────────────────────────────
    n_all, n_off = Counter(), Counter()
    surface = defaultdict(Counter)
    for text, y in zip(X_tr, y_tr):
        for chunk in text.split():
            for w in WORD_RE.findall(normalize_token(chunk)):
                surface[match_key(w)][w] += 1
        for k in message_keys(text):
            n_all[k] += 1
            n_off[k] += int(y)

    def smoothed_prec(k: str) -> float:
        return (n_off[k] + SMOOTH_A * prior) / (n_all[k] + SMOOTH_A)

    dd_rows = []
    for k in n_all:
        if n_off[k] >= 5 and smoothed_prec(k) >= 0.70 and not k.isdigit():
            dd_rows.append({"key": k, "surface": surface[k].most_common(1)[0][0],
                            "train_count": n_all[k], "train_off": n_off[k],
                            "smoothed_precision": round(smoothed_prec(k), 4),
                            "manual_exclusion": excl[k][1] if k in excl else ""})
    dd_rows.sort(key=lambda r: (-r["smoothed_precision"], -r["train_off"]))

    def dd_keep(min_off: int, p_dd: float) -> list[str]:
        return [r["surface"] for r in dd_rows
                if r["train_off"] >= min_off and r["smoothed_precision"] >= p_dd
                and not r["manual_exclusion"]]

    # ── 3. Güvenli kelimeler ───────────────────────────────────────────
    safe_rows = [{"key": k, "surface": surface[k].most_common(1)[0][0],
                  "train_count": n_all[k], "train_off": n_off[k]}
                 for k in n_all
                 if n_all[k] >= SAFE_MIN_COUNT and n_off[k] / n_all[k] < SAFE_MAX_PREC]

    # ── 4. Dev üzerinde ızgara arama ───────────────────────────────────
    grid_rows, best = [], None
    seen = set()
    for p_ext, dd, use_ext, fz in itertools.product(*GRID.values()):
        if not use_ext and dd is None:
            continue
        cfg_key = (p_ext if use_ext else None, dd, use_ext, fz)
        if cfg_key in seen:
            continue
        seen.add(cfg_key)
        entries = sorted(set((ext_keep(p_ext) if use_ext else []) +
                             (dd_keep(*dd) if dd else [])))
        lex_keys = {match_key(w) for e in entries for w in WORD_RE.findall(e)}
        safe = [r["surface"] for r in safe_rows if r["key"] not in lex_keys]
        known = [surface[k].most_common(1)[0][0] for k in n_all
                 if n_all[k] >= KNOWN_MIN_COUNT and k not in lex_keys]
        params = {"fuzzy_min_len": fz[0] if fz else None,
                  "two_edit_min_len": fz[1] if fz else 99}
        m = LexiconMatcher(entries, safe, known_words=known, **params)
        kinds = Counter()
        p_dev = []
        for t in X_dv:
            r = m.find(t)
            kinds.update(mm.kind for mm in r.matches)
            p_dev.append(r.score)
        ev = evaluate(y_dv, p_dev, 0.5)
        row = {"use_external": use_ext, "p_ext": p_ext if use_ext else "",
               "dd_min_off": dd[0] if dd else "", "p_dd": dd[1] if dd else "",
               "fuzzy": f"{fz[0]}/{fz[1]}" if fz else "none (ablation)",
               "selectable": fz is not None,
               "lexicon_size": len(entries), "dev_macro_f1": round(ev["macro_f1"], 4),
               "dev_precision_off": round(ev["precision_off"], 4),
               "dev_recall_off": round(ev["recall_off"], 4),
               "dev_fuzzy_matches": kinds["fuzzy"]}
        grid_rows.append(row)
        print("  ", row)
        score = (ev["macro_f1"], -len(entries))
        if fz is not None and (best is None or score > best[0]):
            best = (score, row, entries, safe, known, params, ev)

    _, best_row, entries, safe, known, params, dev_ev = best
    print("\nSeçilen yapılandırma (dev macro-F1):", best_row)

    # ── 5. Kaydet ──────────────────────────────────────────────────────
    header = [
        "# UC4 saldırgan dil sözlüğü — build_lexicon.py tarafından üretildi, elle düzenlemeyin.",
        f"# Kaynaklar: ooguz/turkce-kufur-karaliste (CC-BY-SA-4.0) + OffensEval-TR train istatistikleri",
        f"# Girdi sayısı: {len(entries)}",
    ]
    (LEXICON_DIR / "offensive_words.txt").write_text(
        "\n".join(header + entries) + "\n", encoding="utf-8")
    (LEXICON_DIR / "safe_words.txt").write_text("\n".join(sorted(safe)) + "\n", encoding="utf-8")
    (LEXICON_DIR / "known_words.txt").write_text("\n".join(sorted(known)) + "\n", encoding="utf-8")

    ext_final = set(ext_keep(best_row["p_ext"])) if best_row["use_external"] else set()
    for r in ext_audit:
        s = r["train_support"]
        if is_excluded(r["entry"]):
            r["decision"] = "drop"
            r["reason"] = "manuel dışlama: " + excl[single_key(r["entry"])][1]
        elif not best_row["use_external"]:
            r["decision"], r["reason"] = "drop", "dış liste seçilen yapılandırmada kullanılmadı"
        elif r["entry"] in ext_final:
            r["decision"] = "keep"
            r["reason"] = ("train kesinliği yeterli" if s >= MIN_SUPPORT
                           else "train'de kanıt yok, uzunluk >= 4")
        else:
            r["decision"] = "drop"
            r["reason"] = ("train kesinliği düşük (yanlış alarm riski)" if s >= MIN_SUPPORT
                           else "train'de kanıt yok ve kısa (< 4)")

    write_csv(OUT_RESULTS / "grid_dev.csv", grid_rows)
    write_csv(OUT_RESULTS / "external_audit.csv", ext_audit)
    write_csv(OUT_RESULTS / "datadriven_candidates.csv", dd_rows)
    write_csv(OUT_RESULTS / "safe_words.csv", safe_rows)

    # Train kapsama (bilgi amaçlı; seçim dev ile yapıldı)
    final = LexiconMatcher(entries, safe, known_words=known, **params)
    # Ablation: aynı sözlük, bulanık eşleşme kapalı (edit distance'ın katkısı)
    ablation = evaluate(y_dv, LexiconMatcher(entries, safe, known_words=known,
                                             fuzzy_min_len=None).predict_proba(X_dv), 0.5)
    tr_ev = evaluate(y_tr, final.predict_proba(X_tr), 0.5)
    kinds = Counter(m.kind for t in X_dv for m in final.find(t).matches)

    meta = {
        "sources": {
            "external": {"url": EXT_LEXICON_URL, "license": "CC-BY-SA-4.0",
                         "sha256": ext_sha, "unique_entries": len(ext)},
            "data_driven": "OffensEval-TR 2020 train split (stefan-it bölümü), yalnızca train",
        },
        "rules": {
            "min_support": MIN_SUPPORT, "safe_min_count": SAFE_MIN_COUNT,
            "safe_max_precision": SAFE_MAX_PREC, "known_min_count": KNOWN_MIN_COUNT,
            "smoothing_a": SMOOTH_A,
            "fuzzy_scope": "yalnızca bilinmeyen tokenlar (non-word ilkesi)",
            "prefix_min_len": 4, "max_suffix_len": 6,
            "matching_key": "Türkçe küçük harf + ASCII katlama + tekrar sıkıştırma",
        },
        "selected_config": best_row,
        "matcher_params": params,
        "threshold": 0.5,
        "lexicon_size": len(entries),
        "external_kept": len(ext_final),
        "external_dropped": len(ext) - len(ext_final),
        "data_driven_kept": len(set(entries) - ext_final),
        "safe_words": len(safe),
        "known_words": len(known),
        "dev_metrics": dev_ev,
        "dev_metrics_fuzzy_off_ablation": ablation,
        "train_metrics_info_only": tr_ev,
        "dev_match_kinds": dict(kinds),
        "grid_size": len(grid_rows),
        "manual_exclusions": {
            "file": "src/lexicon/manual_exclusions.txt",
            "sha256": hashlib.sha256(EXCLUSIONS_PATH.read_bytes()).hexdigest()
            if EXCLUSIONS_PATH.exists() else None,
            "listed": len(excl),
            "rule": "tek başına, bağlam olmadan saldırgan olmayan kelime sözlükte olmamalı",
            "removed_from_external": sorted(r["entry"] for r in ext_audit
                                            if r["reason"].startswith("manuel dışlama")),
            "removed_from_data_driven": sorted(r["surface"] for r in dd_rows
                                               if r["manual_exclusion"]),
            "decided_without_test_set": True,
        },
        "test_set_used": False,
        "runtime_sec": round(time.time() - t0, 1),
    }
    save_json(meta, LEXICON_DIR / "lexicon_meta.json")
    print(f"\nSözlük: {len(entries)} girdi  (dış: {len(ext_final)}, veriden: "
          f"{meta['data_driven_kept']})  güvenli kelime: {len(safe)}")
    mx = meta["manual_exclusions"]
    print(f"Manuel dışlama: listede {mx['listed']} kelime → dış listeden çıkarılan "
          f"{len(mx['removed_from_external'])}, train adaylarından çıkarılan "
          f"{len(mx['removed_from_data_driven'])}")
    leaked = [e for e in entries if is_excluded(e)]
    if leaked:
        raise SystemExit(f"HATA: dışlanan kelimeler sözlükte kaldı: {leaked}")
    print(f"Dev macro-F1: {dev_ev['macro_f1']:.4f}   (bulanık kapalı ablation: "
          f"{ablation['macro_f1']:.4f})   Eşleşme türleri (dev): {dict(kinds)}")
    print(f"Kaydedildi: src/lexicon/offensive_words.txt, lexicon_meta.json, results/lexicon/*.csv")


if __name__ == "__main__":
    main()
