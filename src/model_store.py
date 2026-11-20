"""Eğitilmiş model dosyalarını Hugging Face Hub'dan indirme ve doğrulama (R7).

Model dosyaları git'e girmez (.gitignore: models/*). Bunun yerine:

* `models_manifest.json` (repoda, küçük): Hub deposunun adı, sabitlenmiş commit
  (revision) ve `models/` altındaki her dosyanın boyutu + SHA-256 özeti.
* Uygulama açılırken eksik dosyaları bu sabit commit'ten indirir ve her indirilen
  dosyanın SHA-256 özetini manifest ile karşılaştırır. Uyuşmazsa dosya silinir ve
  o model yüklenmez.

Neden özet kontrolü: `model_int8.pt` (torch.load, weights_only=False) ve
`model.joblib` pickle tabanlıdır; pickle dosyası yüklenirken kod çalıştırabilir.
Bu yüzden yalnızca özeti GitHub'daki manifest ile birebir tutan dosya yüklenir.

Komutlar (proje kök klasöründen):

    python -m src.model_store download              # eksik dosyaları indir (hepsi)
    python -m src.model_store download bert_int8    # yalnızca bir model
    python -m src.model_store verify                # yereldeki dosyaları özetle kontrol et

Yalnızca modelleri Hub'a yükleyen kişi için (README, "Modelleri Hub'a yükleme"):

    python -m src.model_store manifest --repo-id KULLANICI/DEPO
    hf upload KULLANICI/DEPO models . --exclude ".gitkeep" --exclude ".cache/*"
    python -m src.model_store pin                   # Hub'daki son commit'i sabitle + karşılaştır
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

if __package__ in (None, ""):  # `python src/model_store.py` ile de çalışsın
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import MODELS_DIR, ROOT  # noqa: E402

MANIFEST_PATH = ROOT / "models_manifest.json"

# models/ altındaki klasör → o klasöre ihtiyaç duyan uygulama modeli.
# Kural modeli (rule) models/ kullanmaz: sözlüğü src/lexicon/ içinde, git'te.
# w2v/ yalnızca eğitimde kullanılır; arşiv olarak yüklenir, uygulama indirmez.
DIR_TO_MODEL = {"tfidf": "tfidf", "nn": "nn", "bertturk": "bert", "bertturk_int8": "bert_int8"}
SKIP_NAMES = {".gitkeep"}
SKIP_DIRS = {".cache"}


class ModelStoreError(RuntimeError):
    pass


# ───────────────────────── yardımcılar ─────────────────────────
def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    if not path.exists():
        raise ModelStoreError(
            f"{path.name} bulunamadı. Modeller henüz Hub'a yüklenmemiş olabilir.")
    m = json.loads(path.read_text(encoding="utf-8"))
    for key in ("repo_id", "files"):
        if key not in m:
            raise ModelStoreError(f"{path.name} içinde '{key}' alanı yok.")
    return m


def files_for(manifest: dict, model_key: str) -> list[dict]:
    return [f for f in manifest["files"] if model_key in f.get("used_by", [])]


def _looks_complete(local: Path, entry: dict) -> bool:
    """Hızlı kontrol (her açılışta): dosya var ve boyutu doğru mu."""
    return local.is_file() and local.stat().st_size == entry["bytes"]


def _hub_download(repo_id: str, filename: str, revision: str | None, local_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=repo_id, filename=filename,
                                revision=revision, local_dir=str(local_dir)))


# ───────────────────────── indirme ─────────────────────────
def ensure_models(model_keys: Iterable[str], *, models_dir: Path = MODELS_DIR,
                  manifest_path: Path = MANIFEST_PATH,
                  downloader: Callable[[str, str, str | None, Path], Path] = _hub_download,
                  log: Callable[[str], None] = print) -> dict[str, str]:
    """Verilen modellerin eksik dosyalarını indirir ve özetlerini doğrular.

    Hiçbir zaman istisna fırlatmaz; {model_key: hata mesajı} döndürür.
    Boş sözlük = istenen bütün modeller hazır. Dosyalar zaten yerelde ve boyutları
    doğruysa ağa hiç çıkılmaz.
    """
    keys = [k for k in model_keys if k in DIR_TO_MODEL.values()]
    if not keys:
        return {}
    try:
        manifest = load_manifest(manifest_path)
    except ModelStoreError as e:
        # Manifest yoksa yalnızca dosyası eksik olan modeller hata alır
        return {k: str(e) for k in keys
                if not any((models_dir / d).is_dir() for d, m in DIR_TO_MODEL.items() if m == k)}

    errors: dict[str, str] = {}
    for key in keys:
        entries = files_for(manifest, key)
        if not entries:
            errors[key] = f"manifest'te '{key}' için dosya yok"
            continue
        todo = [e for e in entries if not _looks_complete(models_dir / e["path"], e)]
        if not todo:
            continue
        total_mb = sum(e["bytes"] for e in todo) / 2**20
        log(f"[model_store] {key}: {len(todo)} dosya indiriliyor ({total_mb:.0f} MB) "
            f"— {manifest['repo_id']}@{(manifest.get('revision') or 'main')[:10]}")
        for e in todo:
            local = models_dir / e["path"]
            try:
                if local.exists():          # boyutu yanlış (yarım kalmış) dosya
                    local.unlink()
                local.parent.mkdir(parents=True, exist_ok=True)
                got = downloader(manifest["repo_id"], e["path"], manifest.get("revision"),
                                 models_dir)
                got = Path(got)
                if got.resolve() != local.resolve():
                    raise ModelStoreError(f"beklenmeyen konum: {got}")
                digest = sha256_of(local)
                if digest != e["sha256"]:
                    local.unlink(missing_ok=True)
                    raise ModelStoreError(
                        f"{e['path']}: SHA-256 uyuşmuyor (beklenen {e['sha256'][:12]}…, "
                        f"gelen {digest[:12]}…); dosya silindi")
            except Exception as ex:  # noqa: BLE001 — ağ, izin, özet… hepsi kullanıcıya gösterilir
                errors[key] = f"{type(ex).__name__}: {ex}"
                break
        if key not in errors:
            log(f"[model_store] {key}: hazır (SHA-256 doğrulandı)")
    return errors


def verify(model_keys: Iterable[str] | None = None, *, models_dir: Path = MODELS_DIR,
           manifest_path: Path = MANIFEST_PATH) -> list[str]:
    """Yereldeki dosyaların tam özet kontrolü. Sorunlu dosyaların listesini döndürür."""
    manifest = load_manifest(manifest_path)
    wanted = set(model_keys) if model_keys else None
    problems = []
    for e in manifest["files"]:
        if wanted is not None and not wanted.intersection(e.get("used_by", [])):
            continue
        local = models_dir / e["path"]
        if not local.is_file():
            problems.append(f"eksik: {e['path']}")
        elif local.stat().st_size != e["bytes"]:
            problems.append(f"boyut farklı: {e['path']}")
        elif sha256_of(local) != e["sha256"]:
            problems.append(f"SHA-256 farklı: {e['path']}")
    return problems


# ───────────────────────── manifest (yalnızca yükleyen kişi) ─────────────────────────
def build_manifest(repo_id: str, *, models_dir: Path = MODELS_DIR) -> dict:
    files = []
    for p in sorted(models_dir.rglob("*")):
        rel = p.relative_to(models_dir)
        if not p.is_file() or p.name in SKIP_NAMES or rel.parts[0] in SKIP_DIRS:
            continue
        top = rel.parts[0] if len(rel.parts) > 1 else ""
        used_by = [DIR_TO_MODEL[top]] if top in DIR_TO_MODEL else []
        files.append({"path": rel.as_posix(), "bytes": p.stat().st_size,
                      "sha256": sha256_of(p), "used_by": used_by})
    if not files:
        raise ModelStoreError(f"{models_dir} içinde dosya yok")
    missing = sorted(set(DIR_TO_MODEL.values()) - {m for f in files for m in f["used_by"]})
    if missing:
        raise ModelStoreError(f"şu modellerin dosyaları yok: {missing}")
    return {
        "repo_id": repo_id,
        "revision": None,           # `pin` komutu yüklemeden sonra doldurur
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "Model dosyaları git'te değil; src/model_store.py bu listeye göre indirir "
                "ve SHA-256 ile doğrular.",
        "files": files,
    }


def pin(manifest_path: Path = MANIFEST_PATH) -> dict:
    """Hub'daki güncel commit'i manifest'e yazar ve uzak dosyaları manifest ile karşılaştırır.
    Token gerektirmez (depo herkese açık olmalı)."""
    from huggingface_hub import HfApi
    manifest = load_manifest(manifest_path)
    info = HfApi().model_info(manifest["repo_id"], files_metadata=True)
    remote = {s.rfilename: s for s in info.siblings}
    problems = []
    for e in manifest["files"]:
        s = remote.get(e["path"])
        if s is None:
            problems.append(f"Hub'da yok: {e['path']}")
            continue
        if s.size is not None and s.size != e["bytes"]:
            problems.append(f"boyut farklı: {e['path']} (Hub {s.size}, yerel {e['bytes']})")
        lfs_sha = getattr(getattr(s, "lfs", None), "sha256", None)
        if lfs_sha and lfs_sha != e["sha256"]:
            problems.append(f"SHA-256 farklı: {e['path']}")
    if problems:
        raise ModelStoreError("Hub ile manifest uyuşmuyor:\n  " + "\n  ".join(problems))
    manifest["revision"] = info.sha
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    return manifest


# ───────────────────────── komut satırı ─────────────────────────
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="eksik model dosyalarını indir")
    d.add_argument("models", nargs="*", default=sorted(set(DIR_TO_MODEL.values())))
    v = sub.add_parser("verify", help="yereldeki dosyaları SHA-256 ile kontrol et")
    v.add_argument("models", nargs="*")
    m = sub.add_parser("manifest", help="models/ klasöründen manifest üret (yükleyen kişi)")
    m.add_argument("--repo-id", required=True, help="örn. kullaniciadi/depo-adi")
    sub.add_parser("pin", help="Hub'daki commit'i manifest'e sabitle (yükleyen kişi)")
    a = ap.parse_args(argv)

    try:
        if a.cmd == "download":
            unknown = set(a.models) - set(DIR_TO_MODEL.values())
            if unknown:
                ap.error(f"bilinmeyen model: {sorted(unknown)}")
            errors = ensure_models(a.models)
            for k, msg in errors.items():
                print(f"HATA {k}: {msg}", file=sys.stderr)
            if not errors:
                print("Tüm istenen modeller hazır.")
            return 1 if errors else 0
        if a.cmd == "verify":
            problems = verify(a.models or None)
            for p in problems:
                print(p, file=sys.stderr)
            print("Sorun yok." if not problems else f"{len(problems)} sorun bulundu.")
            return 1 if problems else 0
        if a.cmd == "manifest":
            if "/" not in a.repo_id:
                ap.error("--repo-id 'kullaniciadi/depo-adi' biçiminde olmalı")
            man = build_manifest(a.repo_id)
            MANIFEST_PATH.write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
            total = sum(f["bytes"] for f in man["files"]) / 2**20
            print(f"{MANIFEST_PATH.name}: {len(man['files'])} dosya, {total:.0f} MB")
            for f in man["files"]:
                print(f"  {f['bytes'] / 2**20:8.1f} MB  {f['path']:<38} {','.join(f['used_by']) or '(arşiv)'}")
            return 0
        if a.cmd == "pin":
            man = pin()
            print(f"Sabitlendi: {man['repo_id']} @ {man['revision']}")
            print("Hub'daki dosyalar manifest ile uyumlu.")
            return 0
    except ModelStoreError as e:
        print(f"HATA: {e}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
