"""Edit-distance tabanlı sözlük eşleştirici: UC4 kural baseline'ı (Alan B).

Bir mesaj için akış:
  1. Metin boşluklara göre parçalara ("chunk") bölünür; maskeleme bu
     parçalar üzerinde yapılır, böylece kullanıcının gördüğü metin bozulmaz.
  2. Her parça normalize edilir (normalizer.normalize_token) ve içindeki harf
     dizileri eşleştirme anahtarına çevrilir (küçük harf + ş→s katlama +
     tekrar sıkıştırma).
  3. Her anahtar sırayla denenir:
       a) TAM eşleşme
       b) ÖNEK eşleşme (Türkçe ekler: "salaklar" → "salak"), yalnızca
          uzunluğu >= prefix_min_len olan girdiler için
       c) BULANIK eşleşme: Levenshtein (minimum edit distance) <= d;
          d uzunluğa göre 1 ya da 2. YALNIZCA "bilinmeyen" (train'de nadir)
          tokenlara uygulanır: yazım düzeltmedeki non-word ilkesi. "yeter"
          gibi gerçek kelimeler "beter"e bağlanmaz; "gerzekalı" gibi nadir
          gizleme biçimleri yakalanır.
     "Güvenli kelimeler" (eğitim verisinde sık geçen ve çoğunlukla temiz
     mesajlarda görülen kelimeler) önek VE bulanık eşleşmeden muaftır.
  4. Çok kelimeli girdiler ardışık anahtar dizileri üzerinde tam eşleşir.
  5. Boşlukla ayrılmış tek harf dizileri (s a l a k) birleştirilip tek
     parça gibi denetlenir.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .normalizer import _L, clean_text, match_key, normalize_token

try:
    from rapidfuzz import process as _rf_process
    from rapidfuzz.distance import Levenshtein as _rf_lev
except ImportError:  # rapidfuzz yoksa kendi DP uygulamamız kullanılır (yavaş)
    _rf_process = None

WORD_RE = re.compile(f"{_L}+")
PLACEHOLDERS = {"url", "user"}


def min_edit_distance(a: str, b: str) -> int:
    """Klasik dinamik programlama ile Levenshtein mesafesi (ekle/sil/değiştir = 1).

    Jurafsky & Martin, SLP3 Bölüm 2'deki algoritma (değiştirme maliyeti 1).
    rapidfuzz ile aynı sonucu verdiği testlerde doğrulanır.
    """
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


@dataclass
class Match:
    chunk_start: int          # maskelenecek ilk parça indeksi
    chunk_end: int            # son parça indeksi (dahil)
    surface: str              # kullanıcının yazdığı biçim
    entry: str                # eşleşen sözlük girdisi
    kind: str                 # exact | prefix | fuzzy | multiword
    distance: int = 0
    score: float = 1.0        # 1.0 tam/önek; bulanık için 1 - d/len


@dataclass
class MatchResult:
    chunks: list[str]
    matches: list[Match] = field(default_factory=list)

    @property
    def is_offensive(self) -> bool:
        return bool(self.matches)

    @property
    def score(self) -> float:
        return max((m.score for m in self.matches), default=0.0)

    def masked_text(self, mask_char: str = "*") -> str:
        """Eşleşen parçaları ilk harf hariç maskeler: 'salak' → 's****'."""
        first, rest = set(), set()
        for m in self.matches:
            first.add(m.chunk_start)
            rest.update(range(m.chunk_start + 1, m.chunk_end + 1))
        out = []
        for i, c in enumerate(self.chunks):
            if i in first and c:
                out.append(c[0] + mask_char * (len(c) - 1))   # ilk harf görünür
            elif i in rest:
                out.append(mask_char * len(c))                # çok parçalı eşleşmenin devamı
            else:
                out.append(c)
        return " ".join(out)


class LexiconMatcher:
    def __init__(
        self,
        entries,
        safe_words=(),
        prefix_min_len: int = 4,
        max_suffix_len: int = 6,
        fuzzy_min_len: int | None = 5,
        two_edit_min_len: int = 8,
        known_words=(),
    ):
        self.prefix_min_len = prefix_min_len
        self.max_suffix_len = max_suffix_len
        self.fuzzy_min_len = fuzzy_min_len
        self.two_edit_min_len = two_edit_min_len

        self.single: dict[str, str] = {}        # anahtar → orijinal girdi
        self.multi: dict[tuple, str] = {}       # (anahtar, anahtar, ...) → girdi
        for e in entries:
            e = e.strip()
            if not e or e.startswith("#"):
                continue
            keys = tuple(match_key(w) for w in WORD_RE.findall(normalize_token(e)))
            keys = tuple(k for k in keys if k)
            if not keys:
                continue
            if len(keys) == 1:
                self.single.setdefault(keys[0], e)
            else:
                self.multi.setdefault(keys, e)
        self.multi_lengths = sorted({len(k) for k in self.multi}, reverse=True)
        self.long_keys = {k for k in self.single if len(k) >= prefix_min_len}
        fz = fuzzy_min_len or 10**9
        self.fuzzy_keys = sorted(k for k in self.single if len(k) >= fz - 1)
        self.safe = {match_key(w) for w in safe_words if w.strip()}
        # Bilinen kelimeler: gerçek (sık) kelimeler → yalnızca BULANIK eşleşmeden muaf.
        # Yazım düzeltmedeki "non-word error" ilkesi: edit distance yalnızca
        # sözlükte olmayan, nadir biçimlere uygulanır.
        self.known = {match_key(w) for w in known_words if w.strip()}

    # ── tek anahtar ────────────────────────────────────────────────────
    def match_key(self, k: str) -> tuple[str, str, int, float] | None:
        """(girdi, tür, mesafe, skor) ya da None."""
        if len(k) < 2 or k in PLACEHOLDERS:
            return None
        if k in self.single:
            return self.single[k], "exact", 0, 1.0
        if k in self.safe:
            return None
        lo = max(self.prefix_min_len, len(k) - self.max_suffix_len)
        for n in range(len(k) - 1, lo - 1, -1):
            if k[:n] in self.long_keys:
                return self.single[k[:n]], "prefix", 0, 1.0
        if (self.fuzzy_min_len and len(k) >= self.fuzzy_min_len and self.fuzzy_keys
                and k not in self.known):
            d_max = 2 if len(k) >= self.two_edit_min_len else 1
            best = self._fuzzy(k, d_max)
            if best is not None:
                key, d = best
                return self.single[key], "fuzzy", d, 1.0 - d / len(k)
        return None

    def _fuzzy(self, k: str, d_max: int):
        if _rf_process is not None:
            r = _rf_process.extractOne(
                k, self.fuzzy_keys, scorer=_rf_lev.distance, score_cutoff=d_max
            )
            return (r[0], int(r[1])) if r else None
        best = None
        for key in self.fuzzy_keys:
            if abs(len(key) - len(k)) > d_max:
                continue
            d = min_edit_distance(k, key)
            if d <= d_max and (best is None or d < best[1]):
                best = (key, d)
        return best

    # ── mesaj ──────────────────────────────────────────────────────────
    def find(self, text: str) -> MatchResult:
        chunks = clean_text(text).split(" ") if clean_text(text) else []
        res = MatchResult(chunks=chunks)
        if not chunks:
            return res

        # (anahtar, parça indeksi) listesi
        flat: list[tuple[str, int]] = []
        for i, c in enumerate(chunks):
            for w in WORD_RE.findall(normalize_token(c, lowercase=True)):
                flat.append((match_key(w), i))

        matched_chunks: set[int] = set()

        # 1) çok kelimeli girdiler
        for n in self.multi_lengths:
            for s in range(len(flat) - n + 1):
                seq = tuple(k for k, _ in flat[s:s + n])
                if seq in self.multi:
                    a, b = flat[s][1], flat[s + n - 1][1]
                    res.matches.append(Match(a, b, " ".join(chunks[a:b + 1]),
                                             self.multi[seq], "multiword"))
                    matched_chunks.update(range(a, b + 1))

        # 2) tek kelimeler
        for k, i in flat:
            if i in matched_chunks:
                continue
            m = self.match_key(k)
            if m:
                res.matches.append(Match(i, i, chunks[i], *m))
                matched_chunks.add(i)

        # 3) boşlukla ayrılmış tek harf dizileri: "s a l a k"
        i = 0
        while i < len(chunks):
            j = i
            while j < len(chunks) and len(chunks[j]) == 1 and WORD_RE.fullmatch(chunks[j]):
                j += 1
            if j - i >= 3 and not matched_chunks.intersection(range(i, j)):
                m = self.match_key(match_key("".join(chunks[i:j])))
                if m:
                    res.matches.append(Match(i, j - 1, " ".join(chunks[i:j]), *m))
                    matched_chunks.update(range(i, j))
            i = max(j, i + 1)
        return res

    def predict_proba(self, texts) -> list[float]:
        return [self.find(t).score for t in texts]

    # ── kalıcılık ──────────────────────────────────────────────────────
    @classmethod
    def from_files(cls, lexicon_path: Path, safe_path: Path | None = None,
                   known_path: Path | None = None, **params):
        def read(p):
            return Path(p).read_text(encoding="utf-8").splitlines() if p and Path(p).exists() else []
        return cls(read(lexicon_path), read(safe_path), known_words=read(known_path), **params)
