"""Entegrasyon testleri: gerçek modeller + uygulamanın kendi fonksiyonları.

Çalıştırma:  python -m pytest tests/test_integration.py -v

* Eğitilmiş modeller yoksa (models/bertturk_int8/meta.json) otomatik ATLANIR.
* Yalnızca elle yazılmış örnek mesajlar kullanılır; TEST SETİ YÜKLENMEZ.
* Hiçbir dosyaya yazmaz; sonuçlar sunuma sayı olarak girmez, yalnızca sistemin
  doğru çalıştığını doğrular.

Kapsam (direktifteki uç durumlar dahil):
  1. Bütün modeller yüklenir; INT8 gerçekten quantized; normalizasyon sürümü eşleşir
  2. Her model her uç girdide [0, 1] aralığında bir olasılık döndürür, çökmez
  3. Aynı mesaj iki kez → aynı sonuç (tekrar eden mesajlar)
  4. Büyük harf düzeltmesi: "it herif" ile "İT HERİF" aynı sonucu alır
  5. Sözlük: dışlanan nötr kelimeler yanlış alarm vermez; maskeleme doğru
  6. Uygulama: etiket + güven + maskeleme + süre kaydı + model değiştirme + yan yana tablo
  7. Hata durumları: eksik model dosyası, yanlış normalizasyon sürümü, boş mesaj
  8. Bariz örneklerde makul davranış (akıl sağlığı kontrolü — metrik değil)
"""
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("gradio")

if not (ROOT / "models" / "bertturk_int8" / "meta.json").exists():
    pytest.skip("Eğitilmiş modeller yok (önce 02-05 çalıştırılmalı)", allow_module_level=True)

from src.lexicon.normalizer import NORMALIZER_VERSION, normalize_text  # noqa: E402

ALL_KEYS = ["rule", "tfidf", "nn", "bert", "bert_int8"]

EDGE_INPUTS = {
    "boş": "",
    "yalnızca boşluk": "   \t  ",
    "tek harf": "a",
    "çok uzun": "merhaba arkadaşlar nasılsınız " * 400,
    "emoji": "😂😂😂 🔥🔥",
    "url ve kullanıcı": "@USER bak şuna https://ornek.com/x?y=1",
    "sayılar ve semboller": "2020 #### 5 kişi %100 $$$ ***",
    "Türkçe-İngilizce karışık": "bro this maç was insane ya",
    "noktalı/noktasız i": "IŞIK İSTANBUL ılık iğne",
    "Türkçe karakter eksik": "gerizekali serefsiz misin",
    "tekrar harf": "salaaaaaak misiiin",
    "leetspeak": "s4l4k m1s1n",
    "harf arası noktalama": "s.a.l.a.k",
    "boşluklu harfler": "s a l a k",
    "görünmez karakter": "mer​haba­ dünya",
    "kontrol karakteri": "merhaba\x00\x07 dünya",
    "sözlük dışı uydurma kelime": "zırtapozlukbilimsel kuantumsal",
    "alıntı içinde hakaret": 'adam bana "salak" dedi, çok üzüldüm',
    "olumsuzlama": "sen salak değilsin, çok zekisin",
    "hakarete benzeyen nötr kelime": "malzeme listesi ve analiz raporu hazır",
}


@pytest.fixture(scope="module")
def app():
    """app.py'yi içe aktarır (modelleri yükler, arayüzü kurar ama BAŞLATMAZ)."""
    import app as app_module

    return app_module


@pytest.fixture(scope="module")
def preds(app):
    return app.PREDICTORS


# ── 1. Yükleme ────────────────────────────────────────────────────────────
def test_all_models_loaded(app):
    assert not app.LOAD_ERRORS, f"Yüklenemeyen modeller: {app.LOAD_ERRORS}"
    assert set(app.PREDICTORS) == set(ALL_KEYS)


def test_int8_is_really_quantized(preds):
    b = preds["bert_int8"].backend
    assert b["int8_linear_layers"] > 0
    assert b["fp32_linear_layers"] == 0
    assert b["engine"] in {"x86", "fbgemm", "onednn"}


def test_normalizer_version_matches(preds):
    for k in ("bert", "bert_int8"):
        assert preds[k].meta.get("normalizer_version") == NORMALIZER_VERSION, k


def test_thresholds_selected_on_dev_not_test(preds):
    for k in ALL_KEYS:
        assert 0.0 < preds[k].threshold < 1.0, k
        assert preds[k].meta.get("test_set_used") is False, f"{k}: test_set_used != False"


# ── 2. Uç girdiler ────────────────────────────────────────────────────────
@pytest.mark.parametrize("key", ALL_KEYS)
@pytest.mark.parametrize("name", list(EDGE_INPUTS))
def test_edge_input_returns_valid_probability(preds, key, name):
    p = preds[key].predict_proba([EDGE_INPUTS[name]])
    assert len(p) == 1
    assert 0.0 <= float(p[0]) <= 1.0, f"{key} / {name}: {p[0]}"


# ── 3. Tekrar eden mesaj ──────────────────────────────────────────────────
@pytest.mark.parametrize("key", ALL_KEYS)
def test_same_message_same_result(preds, key):
    msg = "bugün maç çok güzeldi ama hakem berbattı"
    a = float(preds[key].predict_proba([msg])[0])
    b = float(preds[key].predict_proba([msg])[0])
    assert a == pytest.approx(b, abs=1e-6)


# ── 4. Büyük harf düzeltmesi (uygulamada bulunan hata) ────────────────────
@pytest.mark.parametrize("lower, upper", [
    ("it herif", "İT HERİF"),
    ("pislik", "PİSLİK"),
    ("bugün hava çok güzel", "BUGÜN HAVA ÇOK GÜZEL"),
])
def test_all_caps_same_as_lowercase_for_bert(preds, lower, upper):
    assert normalize_text(lower, lowercase=False) == normalize_text(upper, lowercase=False)
    for k in ("bert", "bert_int8"):
        pl = float(preds[k].predict_proba([lower])[0])
        pu = float(preds[k].predict_proba([upper])[0])
        assert pl == pytest.approx(pu, abs=1e-6), f"{k}: {lower}={pl:.3f} {upper}={pu:.3f}"


# ── 5. Sözlük: dışlanan nötr kelimeler ve maskeleme ───────────────────────
@pytest.mark.parametrize("msg", [
    "Türbanlı öğretmenimiz çok iyi",
    "Bugün çok sıkılmış görünüyorsun",
    "Analiz raporu yarın hazır olacak",
    "Hırsız dün gece yakalandı",
    "Faturayı ödemek için sıraya girdim",
])
def test_excluded_neutral_words_not_flagged_by_lexicon(app, msg):
    assert not app.MATCHER.find(msg).is_offensive, msg


def test_masking_keeps_neutral_words(app):
    masked, word_level = app.mask("sen tam bir salak herifsin dostum")
    assert word_level
    assert "salak" not in masked
    assert masked.startswith("sen tam bir s")
    assert masked.endswith("dostum")


def test_full_mask_when_no_lexicon_match(app):
    masked, word_level = app.mask("senin gibi birinin fikri kimseyi ilgilendirmiyor")
    assert not word_level
    assert "gizlendi" in masked


# ── 6. Uygulama fonksiyonları ─────────────────────────────────────────────
def _send(app, text, model_key="bert_int8", history=None, log=None):
    return app.on_send(text, app.DISPLAY_NAMES[model_key], history or [], log or [])


def test_app_labels_logs_and_compares(app):
    box, history, log, log_tbl, stats, cmp = _send(app, "merhaba, nasılsın?")
    assert box == ""                                     # kutu temizlendi
    assert history[-2]["role"] == "user"
    assert history[-1]["role"] == "assistant"
    assert ("temiz" in history[-1]["content"]) or ("saldırgan" in history[-1]["content"])
    assert "güven" in history[-1]["content"] and "ms" in history[-1]["content"]
    assert len(log) == 1 and log[0][2] == app.SHORT_NAMES["bert_int8"]
    assert isinstance(log[0][5], float) and log[0][5] > 0       # süre kaydedildi
    assert len(cmp) == len(ALL_KEYS)                            # bütün modeller yan yana
    assert cmp[0][0].startswith("★")                            # seçili model ilk satırda


def test_app_model_switch(app):
    _, h, log, *_ = _send(app, "ilk mesaj", "bert_int8")
    _, h, log, _, _, cmp = _send(app, "ikinci mesaj", "tfidf", h, log)
    assert [row[2] for row in log] == [app.SHORT_NAMES["tfidf"], app.SHORT_NAMES["bert_int8"]]
    assert cmp[0][0] == "★ " + app.SHORT_NAMES["tfidf"]
    assert len(h) == 4


def test_app_offensive_message_is_masked(app):
    _, history, *_ = _send(app, "salak herif defol git buradan", "rule")
    assert "saldırgan" in history[-1]["content"]
    assert "salak" not in history[-2]["content"]          # kullanıcı mesajı maskeli gösterildi


@pytest.mark.parametrize("text", ["", "   "])
def test_app_empty_message_does_not_crash(app, text):
    out = _send(app, text)
    assert out[1] == [] and out[2] == []                  # sohbet ve kayıt değişmedi


def test_app_long_message_truncated(app):
    _, history, *_ = _send(app, "a" * 5000)
    assert "ilk 1000 karakter" in history[-1]["content"]


def test_app_unknown_model_does_not_crash(app):
    out = app.on_send("merhaba", "olmayan model", [], [])
    assert out[1] == []


# ── 7. Hata durumları ─────────────────────────────────────────────────────
def test_missing_model_files_raise_clear_error(monkeypatch, tmp_path):
    import src.predictors as P

    monkeypatch.setattr(P, "MODELS_DIR", tmp_path)
    with pytest.raises((FileNotFoundError, OSError)):
        P.load_predictor("tfidf")


def test_wrong_normalizer_version_is_rejected(monkeypatch):
    import src.predictors as P

    monkeypatch.setattr(P, "NORMALIZER_VERSION", "baska-surum")
    with pytest.raises(RuntimeError, match="normalizasyon"):
        P.load_predictor("bert_int8")


# ── 8. Akıl sağlığı (metrik değil; başarısızlık rapor edilir, ayar yapılmaz) ──
@pytest.mark.parametrize("msg", [
    "bugün hava çok güzel, herkese iyi günler",
    "maçtan sonra arkadaşlarla yemek yedik",
    "yardımın için çok teşekkür ederim",
])
def test_sanity_clear_clean_messages(preds, msg):
    p = float(preds["bert_int8"].predict_proba([msg])[0])
    assert p < preds["bert_int8"].threshold, f"temiz mesaj saldırgan bulundu: p={p:.3f}"


@pytest.mark.parametrize("msg", [
    "salak herif defol git",
    "seni gidi şerefsiz pislik",
    "aptal mısın gerizekalı",
])
def test_sanity_clear_offensive_messages(preds, msg):
    p = float(preds["bert_int8"].predict_proba([msg])[0])
    assert p >= preds["bert_int8"].threshold, f"saldırgan mesaj kaçırıldı: p={p:.3f}"
