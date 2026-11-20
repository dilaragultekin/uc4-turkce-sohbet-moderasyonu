"""Gürültülü Türkçe sohbet metni için regex tabanlı normalizasyon (Alan B).

Adımlar (her token için, sırayla):
  1. URL → "url", @mention → "@user"  (veri setindeki anonimleştirmeyle aynı)
  2. Harfler arasındaki leetspeak karakterleri → harf   (s4lak → salak)
  3. Harfler arasındaki ayırıcıları sil                  (s.a.l.a.k → salak)
  4. 3+ tekrar eden karakteri 2'ye indir                  (salaaaak → salaak)
  5. Türkçe-duyarlı küçük harf                            (I → ı, İ → i)
     - lowercase=True  (TF-IDF, Word2Vec, sözlük): bütün tokenlar
     - lowercase=False (BERTurk, cased): YALNIZCA tamamı büyük harf olan
       tokenlar ("İT HERİF" → "it herif"); normal yazım korunur ("Ankara").
       Neden: Bağırarak (büyük harfle) yazılmış kelimeler cased modelin
       sözlüğünde nadir parçalara bölünüyordu ve hakaretler kaçırılıyordu
       (uygulama denemesinde bulundu, 2026-11-19).

Metin düzeyinde ek olarak boşlukla ayrılmış tek harf dizileri birleştirilir
(s a l a k → salak).

Bilinçli tasarım kararları:
  * Leet dönüşümü yalnızca iki tarafında harf olan karakterlere uygulanır;
    böylece "2020", "5 kişi", "salak!" gibi tokenlar bozulmaz.
    İstisna: token başındaki $, @, € (ör. "$alak"). Rakamla başlayan
    "4ptal" gibi biçimler normalizer'da değil, eşleştiricideki edit distance
    ile yakalanır.
  * Çift harfler korunur (dikkat, Allah, hakkı); yalnızca 3+ tekrar kısaltılır.
  * Türkçe karakterleri ASCII'ye katlama (ş→s) burada YAPILMAZ; modeller
    gerçek Türkçe metni görür. Katlama yalnızca sözlük eşleştiricide kullanılır.
"""
from __future__ import annotations

import re
import unicodedata

# Normalizasyon değiştiğinde güncellenir; BERTurk meta.json'una yazılır ve
# yüklenirken karşılaştırılır (eğitim ile çıkarım aynı normalizasyonu kullanmalı).
NORMALIZER_VERSION = "2-allcaps-fold"

LETTER = "A-Za-zÇĞİÖŞÜçğıöşüÂâÎîÛû"
_L = f"[{LETTER}]"

URL_RE = re.compile(r"^(https?://\S+|www\.\S+|url)$", re.IGNORECASE)
MENTION_RE = re.compile(r"^@user\W*$", re.IGNORECASE)
ZERO_WIDTH_RE = re.compile("[​-‏⁠﻿­]")

LEET_MAP = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
            "@": "a", "$": "s", "€": "e", "!": "i", "|": "i"}
_LEET_CHARS = re.escape("".join(LEET_MAP))
LEET_INNER_RE = re.compile(f"(?<={_L})[{_LEET_CHARS}]+(?={_L})")
LEET_START_RE = re.compile(f"^[$@€](?={_L}{{2,}})")
SEPARATOR_RE = re.compile(f"(?<={_L})[.\\-_*·+]+(?={_L})")
REPEAT_RE = re.compile(r"(.)\1{2,}", re.DOTALL)
SPACED_LETTERS_RE = re.compile(f"(?<!\\S)(?:{_L} ){{2,}}{_L}(?!\\S)")

_TR_UPPER = str.maketrans({"I": "ı", "İ": "i"})
_FOLD = str.maketrans("çğıöşüâîû", "cgiosuaiu")


def tr_lower(s: str) -> str:
    """Türkçe-duyarlı küçük harf: I → ı, İ → i (Python lower() bunu yanlış yapar)."""
    return s.translate(_TR_UPPER).lower()


def fold_turkish(s: str) -> str:
    """Eşleştirme için ASCII katlama: ş→s, ı→i ... (küçük harf girdi beklenir)."""
    return s.translate(_FOLD)


def squeeze_all(s: str) -> str:
    """Tüm tekrarları teke indirir (eşleştirme anahtarı için): salaak → salak."""
    return re.sub(r"(.)\1+", r"\1", s)


def match_key(token: str) -> str:
    """Sözlük eşleştirmesinde kullanılan anahtar: küçük harf + katlama + sıkıştırma."""
    return squeeze_all(fold_turkish(tr_lower(token)))


def _tr_upper(ch: str) -> str:
    return {"i": "İ", "ı": "I"}.get(ch, ch.upper())


def _leet_inner(m: re.Match) -> str:
    # İki komşu harf de büyükse büyük harf üret (S4LAK → SALAK); aksi hâlde küçük (S4lak → Salak)
    s = m.string
    upper = s[m.start() - 1].isupper() and m.end() < len(s) and s[m.end()].isupper()
    return "".join(_tr_upper(LEET_MAP[c]) if upper else LEET_MAP[c] for c in m.group(0))


def is_all_caps(token: str) -> bool:
    """En az 2 harf içeren ve bütün harfleri büyük olan token ("HERİF", "İT")."""
    letters = re.findall(_L, token)
    return len(letters) >= 2 and all(c.isupper() for c in letters)


def normalize_token(tok: str, lowercase: bool = True) -> str:
    """Tek bir boşluksuz token'ı normalize eder."""
    if not tok:
        return tok
    if URL_RE.match(tok):
        return "url"
    if MENTION_RE.match(tok):
        return "@user"
    t = LEET_START_RE.sub(lambda m: LEET_MAP[m.group(0)], tok)
    t = LEET_INNER_RE.sub(_leet_inner, t)
    t = SEPARATOR_RE.sub("", t)
    t = REPEAT_RE.sub(r"\1\1", t)
    if lowercase or is_all_caps(t):
        return tr_lower(t)
    return t


def clean_text(text: str) -> str:
    """Unicode NFC + görünmez karakterleri sil + boşlukları sadeleştir."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = ZERO_WIDTH_RE.sub("", text)
    return " ".join(text.split())


def join_spaced_letters(text: str) -> str:
    """'s a l a k' → 'salak' (en az 3 tek harf)."""
    return SPACED_LETTERS_RE.sub(lambda m: m.group(0).replace(" ", ""), text)


def normalize_text(text: str, lowercase: bool = True) -> str:
    """Modellerin gördüğü metin. TF-IDF/Word2Vec: lowercase=True; BERTurk (cased): False."""
    text = join_spaced_letters(clean_text(text))
    return " ".join(normalize_token(t, lowercase) for t in text.split())
