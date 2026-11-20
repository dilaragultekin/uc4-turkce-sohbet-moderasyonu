"""UC4 — Gerçek zamanlı sohbet moderasyonu uygulaması (R1, R5).

Çalıştırma:  python app.py   → tarayıcıda http://127.0.0.1:7860

UC4 şartı: "chat-like interface where each message is labeled live (clean / offensive)
with confidence; flagged messages are masked; a side panel logs the response time of
every message and lets the user switch between models."

* Sol: sohbet. Her mesaj SEÇİLİ modelle canlı etiketlenir (temiz / saldırgan) + güven.
* Saldırgan mesajlar maskelenir: sözlükte eşleşen kelimeler "s****" biçiminde;
  eşleşen kelime yoksa (model bağlamdan karar verdiyse) mesajın tamamı gizlenir.
* Sağ yan panel:
    - model seçimi (varsayılan: quantized BERTurk INT8 — R5),
    - son mesaj için BÜTÜN modellerin sonucu yan yana (R5'te teşvik edilen),
    - her mesajın yanıt süresi kaydı (seçili model, uçtan uca, CPU),
    - model durumu (INT8 backend kanıtı dahil).

Uygulama CPU'da çalışır (UC4: CPU-only sunucu senaryosu).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import time  # noqa: E402
from datetime import datetime  # noqa: E402

import gradio as gr  # noqa: E402
import torch  # noqa: E402

from src.common import CPU_THREADS, LATENCY_BUDGET_MS  # noqa: E402
from src.model_store import ensure_models  # noqa: E402
from src.predictors import DISPLAY_NAMES, MODEL_KEYS, load_predictor  # noqa: E402

MAX_CHARS = 1000
DEFAULT_MODEL = "bert_int8"
CMP_COLS = ["Model", "Sonuç", "Olasılık", "Eşik", "ms"]
LOG_COLS = ["#", "Saat", "Model", "Sonuç", "Güven", "ms", "Bütçe"]
# Yan paneldeki tablolarda kısa adlar (yana kaydırmadan okunabilsin)
SHORT_NAMES = {"rule": "Kural", "tfidf": "TF-IDF", "nn": "W2V+CNN",
               "bert": "BERT FP32", "bert_int8": "BERT INT8"}
APP_TITLE = "Sohbet Kalkanı — Türkçe Oyun Sohbetinde Canlı Hakaret Tespiti"

torch.set_num_threads(CPU_THREADS)

# ── Eksik model dosyalarını Hugging Face Hub'dan indir (R7; ilk açılışta) ──
# Dosyalar yerelde varsa ağa çıkılmaz. Her indirilen dosya SHA-256 ile doğrulanır.
DOWNLOAD_ERRORS = ensure_models(MODEL_KEYS)

# ── Modelleri yükle (hata olursa uygulama çökmez, durum panelinde görünür) ──
PREDICTORS, LOAD_ERRORS = {}, {}
for _k in MODEL_KEYS:
    if _k in DOWNLOAD_ERRORS:
        LOAD_ERRORS[_k] = f"indirilemedi — {DOWNLOAD_ERRORS[_k]}"
        continue
    try:
        PREDICTORS[_k] = load_predictor(_k)
    except Exception as e:  # noqa: BLE001
        LOAD_ERRORS[_k] = f"{type(e).__name__}: {e}"
MATCHER = PREDICTORS["rule"].matcher if "rule" in PREDICTORS else None
NAME2KEY = {DISPLAY_NAMES[k]: k for k in PREDICTORS}


def status_md() -> str:
    lines = [f"**CPU thread:** {CPU_THREADS} · **Gecikme bütçesi:** {LATENCY_BUDGET_MS:.0f} ms (p95)"]
    for k in MODEL_KEYS:
        if k in PREDICTORS:
            b = getattr(PREDICTORS[k], "backend", None)
            extra = f" — {b}" if k == "bert_int8" and b else ""
            lines.append(f"- ✅ {DISPLAY_NAMES[k]}{extra}")
        else:
            lines.append(f"- ❌ {DISPLAY_NAMES[k]}: `{LOAD_ERRORS.get(k, '?')}`")
    return "\n".join(lines)


def classify(key: str, text: str) -> dict:
    """Tek mesaj, tek model; süre uçtan uca (normalizasyon + tokenizasyon + model)."""
    pred = PREDICTORS[key]
    t0 = time.perf_counter()
    p = float(pred.predict_proba([text])[0])
    ms = (time.perf_counter() - t0) * 1000
    off = p >= pred.threshold
    return {"offensive": off, "p_off": p, "confidence": p if off else 1 - p, "ms": ms}


def mask(text: str) -> tuple[str, bool]:
    """(maskelenmiş metin, kelime düzeyinde mi maskelendi)"""
    if MATCHER is not None:
        r = MATCHER.find(text)
        if r.matches:
            return r.masked_text(), True
    return "▇▇▇ [mesaj gizlendi]", False


def compare_rows(text: str, selected: str, selected_result: dict) -> list[list]:
    """Son mesaj için bütün modellerin sonucu; seçili model ★ ile işaretli, ilk satırda."""
    rows = []
    for k in [selected] + [k for k in PREDICTORS if k != selected]:
        name = ("★ " if k == selected else "") + SHORT_NAMES[k]
        try:
            r = selected_result if k == selected else classify(k, text)
            rows.append([name, "saldırgan" if r["offensive"] else "temiz",
                         round(r["p_off"], 3), PREDICTORS[k].threshold, round(r["ms"], 2)])
        except Exception as e:  # noqa: BLE001
            rows.append([name, f"hata: {type(e).__name__}", None, None, None])
    return rows


def stats_md(log) -> str:
    if not log:
        return ""
    by = {}
    for row in log:
        by.setdefault(row[2], []).append(row[5])
    return "\n".join(f"- {m}: {len(v)} mesaj, ort. {sum(v) / len(v):.1f} ms, maks {max(v):.1f} ms"
                     for m, v in by.items())


def on_send(text, model_name, history, log):
    history = list(history or [])
    log = list(log or [])
    text = (text or "").strip()
    if not text:
        gr.Warning("Boş mesaj gönderilemez.")
        return "", history, log, gr.update(), gr.update(), gr.update()
    note = ""
    if len(text) > MAX_CHARS:
        text, note = text[:MAX_CHARS], f" (ilk {MAX_CHARS} karakter işlendi)"
    key = NAME2KEY.get(model_name)
    if key is None:
        gr.Warning("Seçili model yüklenemedi; durum paneline bakın.")
        return text, history, log, gr.update(), gr.update(), gr.update()
    try:
        r = classify(key, text)          # önce seçili model: kaydedilen süre yalnızca onun
    except Exception as e:  # noqa: BLE001
        history.append({"role": "assistant",
                        "content": f"⚠️ Çıkarım hatası: {type(e).__name__}: {e}"})
        return "", history, log, gr.update(), gr.update(), gr.update()

    if r["offensive"]:
        shown, word_level = mask(text)
        verdict = (f"🔴 **saldırgan** · güven {r['confidence']:.2f} · {r['ms']:.1f} ms · "
                   f"{SHORT_NAMES[key]}" + ("" if word_level else " · (bağlamdan karar; tüm mesaj gizlendi)"))
    else:
        shown = text
        verdict = f"🟢 temiz · güven {r['confidence']:.2f} · {r['ms']:.1f} ms · {SHORT_NAMES[key]}"
    history.append({"role": "user", "content": shown})
    history.append({"role": "assistant", "content": verdict + note})
    log.insert(0, [len(log) + 1, datetime.now().strftime("%H:%M:%S"), SHORT_NAMES[key],
                   "saldırgan" if r["offensive"] else "temiz", round(r["confidence"], 3),
                   round(r["ms"], 2), "✓" if r["ms"] <= LATENCY_BUDGET_MS else "✗"])
    cmp = compare_rows(text, key, r)    # diğer modeller, seçili modelin süresinden SONRA
    return "", history, log, log, stats_md(log), cmp


def make_chatbot():
    try:
        return gr.Chatbot(label="Sohbet", height=520, type="messages")
    except TypeError:  # Gradio 6+: 'messages' varsayılan, 'type' parametresi yok
        return gr.Chatbot(label="Sohbet", height=520)


with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(f"## {APP_TITLE}\n"
                "Mesaj yazın; seçili model mesajı **temiz / saldırgan** olarak etiketler, "
                "saldırgan mesajlar maskelenir. Yan panel bütün modellerin sonucunu ve her "
                "mesajın yanıt süresini gösterir. Tüm çıkarım CPU'da yapılır.")
    with gr.Row():
        # ── Sohbet ──────────────────────────────────────────────────────
        with gr.Column(scale=3):
            chat = make_chatbot()
            with gr.Row():
                box = gr.Textbox(placeholder="Mesajınızı yazın ve Enter'a basın…",
                                 show_label=False, scale=5)
                send = gr.Button("Gönder", variant="primary", scale=1)
        # ── Yan panel ───────────────────────────────────────────────────
        with gr.Column(scale=2):
            choices = [DISPLAY_NAMES[k] for k in MODEL_KEYS if k in PREDICTORS]
            default = DISPLAY_NAMES[DEFAULT_MODEL] if DEFAULT_MODEL in PREDICTORS else (
                choices[0] if choices else None)
            model_dd = gr.Dropdown(choices=choices, value=default,
                                   label="Etiketleyen model (geçiş yapın)")
            gr.Markdown("### Son mesaj — bütün modeller yan yana")
            compare_tbl = gr.Dataframe(headers=CMP_COLS, interactive=False, wrap=True)
            gr.Markdown("### Yanıt süresi kaydı (seçili model, her mesaj)")
            log_tbl = gr.Dataframe(headers=LOG_COLS, interactive=False, wrap=True)
            stats = gr.Markdown()
            with gr.Accordion("Model durumu (INT8 kanıtı)", open=True):
                gr.Markdown(status_md())
    log_state = gr.State([])

    outputs = [box, chat, log_state, log_tbl, stats, compare_tbl]
    box.submit(on_send, [box, model_dd, chat, log_state], outputs)
    send.click(on_send, [box, model_dd, chat, log_state], outputs)

if __name__ == "__main__":
    if DEFAULT_MODEL not in PREDICTORS:
        print(f"UYARI: Varsayılan model ({DEFAULT_MODEL}) yüklenemedi: {LOAD_ERRORS.get(DEFAULT_MODEL)}")
    for k, err in LOAD_ERRORS.items():
        print(f"UYARI: {DISPLAY_NAMES[k]} yüklenemedi: {err}")
    demo.queue().launch(server_name="127.0.0.1", inbrowser=True)
