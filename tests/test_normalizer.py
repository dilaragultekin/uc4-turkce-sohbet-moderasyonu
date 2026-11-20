"""Normalizer birim testleri.  Çalıştırma:  python -m pytest tests -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from src.lexicon.normalizer import (  # noqa: E402
    fold_turkish, join_spaced_letters, match_key, normalize_text, normalize_token, tr_lower,
)


def test_turkish_lowercase():
    assert tr_lower("IŞIK") == "ışık"
    assert tr_lower("İSTANBUL") == "istanbul"


@pytest.mark.parametrize("raw, expected", [
    ("s4l4k", "salak"),          # leetspeak harfler arasında
    ("$alak", "salak"),          # token başında $
    ("s.a.l.a.k", "salak"),      # harfler arası ayırıcı
    ("s-a_l*a.k", "salak"),
    ("salaaaaak", "salaak"),     # 3+ tekrar → 2
    ("dikkat", "dikkat"),        # meşru çift harf korunur
    ("2020", "2020"),            # sayı bozulmaz
    ("salak!", "salak!"),        # sondaki ünlem leet sayılmaz
    ("5te", "5te"),              # rakamla başlayan token bozulmaz
    ("@USER", "@user"),
    ("@USER:", "@user"),
    ("https://ornek.com/x", "url"),
    ("URL", "url"),
])
def test_normalize_token(raw, expected):
    assert normalize_token(raw) == expected


def test_case_preserved_for_bert():
    """BERTurk (cased) girdisinde normal büyük harf kullanımı korunur."""
    assert normalize_text("Merhaba Ankara", lowercase=False) == "Merhaba Ankara"
    assert normalize_text("Merhaba S4lak", lowercase=False) == "Merhaba Salak"
    assert normalize_text("Merhaba S4LAK") == "merhaba salak"


@pytest.mark.parametrize("raw, expected", [
    ("İT HERİF", "it herif"),             # uygulamada bulunan hata: tamamı büyük harf
    ("PİSLİK", "pislik"),
    ("BUGÜN MAÇ VAR", "bugün maç var"),
    ("IŞIK", "ışık"),                     # Türkçe I → ı
    ("S4LAK", "salak"),                   # leet + büyük harf
    ("Merhaba HERİF", "Merhaba herif"),   # yalnızca büyük harf token küçülür
    ("A", "A"),                           # tek harf: dokunulmaz
])
def test_all_caps_folded_for_bert(raw, expected):
    assert normalize_text(raw, lowercase=False) == expected


def test_spaced_letters():
    assert join_spaced_letters("s a l a k mısın") == "salak mısın"
    assert join_spaced_letters("o ve ben") == "o ve ben"      # tek harf + kelime birleşmez
    assert join_spaced_letters("a b") == "a b"                # en az 3 harf gerekir


@pytest.mark.parametrize("raw", ["", "   ", "​", None])
def test_empty_inputs(raw):
    assert normalize_text(raw) == ""


def test_whitespace_and_zero_width():
    assert normalize_text("  merhaba​   dünya \n") == "merhaba dünya"


def test_emoji_kept():
    assert "😂😂" in normalize_text("komik 😂😂😂😂")


def test_fold_and_key():
    assert fold_turkish("şerefsiz gerizekalı") == "serefsiz gerizekali"
    assert match_key("SALAAAK") == "salak"
    assert match_key("Şerefsiz") == match_key("serefsiz")
