"""Sözlük eşleştirici birim testleri.  Çalıştırma:  python -m pytest tests -q

Testler küçük, elle yazılmış bir oyuncak sözlük kullanır; gerçek sözlük
build_lexicon.py ile train verisinden üretilir.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from src.lexicon.matcher import LexiconMatcher, min_edit_distance  # noqa: E402

TOY = ["salak", "aptal", "mal", "şerefsiz", "gerizekalı"]


@pytest.fixture
def m():
    return LexiconMatcher(TOY, safe_words=["salam"])


def test_min_edit_distance_textbook():
    assert min_edit_distance("kitten", "sitting") == 3
    assert min_edit_distance("intention", "execution") == 5
    assert min_edit_distance("", "abc") == 3
    assert min_edit_distance("salak", "salak") == 0


def test_min_edit_distance_matches_rapidfuzz():
    lev = pytest.importorskip("rapidfuzz.distance").Levenshtein
    rnd = random.Random(0)
    for _ in range(300):
        a = "".join(rnd.choice("abcşı") for _ in range(rnd.randint(0, 8)))
        b = "".join(rnd.choice("abcşı") for _ in range(rnd.randint(0, 8)))
        assert min_edit_distance(a, b) == lev.distance(a, b)


@pytest.mark.parametrize("text, kind", [
    ("sen salak mısın", "exact"),
    ("SALAAAK", "exact"),              # tekrar sıkıştırma + Türkçe küçük harf
    ("s4l4k", "exact"),                # leetspeak
    ("s.a.l.a.k", "exact"),            # ayırıcı
    ("serefsiz", "exact"),             # Türkçe karakter eksik
    ("gerizekali", "exact"),
    ("salaksın", "prefix"),            # Türkçe ek
    ("aptallar", "prefix"),
    ("gerzekalı", "fuzzy"),            # 1 harf eksik (edit distance 1)
])
def test_detects(m, text, kind):
    r = m.find(text)
    assert r.is_offensive, text
    assert r.matches[0].kind == kind


def test_spaced_letters(m):
    r = m.find("bu adam s a l a k")
    assert r.is_offensive
    assert r.masked_text() == "bu adam s * * * *"


@pytest.mark.parametrize("text", [
    "malzeme aldım",          # kısa girdi ('mal') yalnızca tam eşleşir
    "salam yedim",            # 'salak'a 1 uzaklıkta ama güvenli kelime
    "@USER URL",
    "",
    "   ",
    "harika bir maçtı 👏",
])
def test_no_false_positive(m, text):
    assert not m.find(text).is_offensive, text


def test_short_entry_exact(m):
    assert m.find("tam bir mal").is_offensive


def test_safe_word_matters():
    """Güvenli kelime listesi olmadan 'salam' yanlış alarm üretir — mekanizmanın gerekçesi."""
    assert LexiconMatcher(TOY).find("salam yedim").is_offensive


def test_masking_keeps_other_words(m):
    r = m.find("sen salaksın!")
    assert r.masked_text() == "sen s********"
    assert r.score == 1.0


def test_known_words_exempt_from_fuzzy_only():
    """Non-word ilkesi: sık kullanılan gerçek kelime bulanık eşleşmez, nadir biçim eşleşir."""
    m = LexiconMatcher(["beter", "salak"], known_words=["yeter", "salaklar"])
    assert not m.find("yeter artık").is_offensive          # 'beter'e 1 uzaklık, ama bilinen kelime
    assert m.find("bater").matches[0].kind == "fuzzy"      # bilinmeyen biçim → yakalanır
    assert m.find("salaklar").matches[0].kind == "prefix"  # bilinen kelime ÖNEK eşleşmesini engellemez


def test_fuzzy_disabled():
    m = LexiconMatcher(TOY, fuzzy_min_len=None)
    assert not m.find("gerzekalı").is_offensive
