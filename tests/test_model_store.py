"""src/model_store.py testleri — ağa çıkmaz; Hub yerine sahte bir indirici kullanılır."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src import model_store as ms


@pytest.fixture()
def hub(tmp_path):
    """Sahte 'Hub' klasörü + ondan üretilmiş manifest + boş yerel models/ klasörü."""
    remote = tmp_path / "remote"
    files = {
        "tfidf/model.joblib": b"tfidf-weights" * 100,
        "tfidf/meta.json": b'{"threshold": 0.5}',
        "bertturk_int8/model_int8.pt": b"int8" * 1000,
        "bertturk_int8/config.json": b"{}",
        "nn/model.pt": b"nn" * 50,
        "bertturk/model.safetensors": b"fp32" * 2000,
        "w2v/w2v.model": b"w2v" * 10,
    }
    for rel, data in files.items():
        p = remote / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    (remote / ".gitkeep").write_bytes(b"")
    manifest = ms.build_manifest("someone/some-repo", models_dir=remote)
    manifest_path = tmp_path / "models_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    local = tmp_path / "models"
    local.mkdir()
    calls = []

    def fake_download(repo_id, filename, revision, local_dir):
        calls.append(filename)
        dst = Path(local_dir) / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(remote / filename, dst)
        return dst

    return {"remote": remote, "local": local, "manifest": manifest_path,
            "download": fake_download, "calls": calls, "files": files}


def run(hub, keys, downloader=None):
    return ms.ensure_models(keys, models_dir=hub["local"], manifest_path=hub["manifest"],
                            downloader=downloader or hub["download"], log=lambda _m: None)


def test_manifest_contents(hub):
    man = json.loads(hub["manifest"].read_text(encoding="utf-8"))
    paths = {f["path"]: f for f in man["files"]}
    assert ".gitkeep" not in paths
    assert paths["bertturk_int8/model_int8.pt"]["used_by"] == ["bert_int8"]
    assert paths["bertturk/model.safetensors"]["used_by"] == ["bert"]
    assert paths["w2v/w2v.model"]["used_by"] == []          # arşiv, uygulama indirmez
    assert man["revision"] is None
    for f in man["files"]:
        assert len(f["sha256"]) == 64


def test_manifest_requires_all_app_models(tmp_path):
    (tmp_path / "tfidf").mkdir()
    (tmp_path / "tfidf" / "model.joblib").write_bytes(b"x")
    with pytest.raises(ms.ModelStoreError, match="dosyaları yok"):
        ms.build_manifest("a/b", models_dir=tmp_path)


def test_downloads_missing_files_and_verifies(hub):
    errors = run(hub, ["bert_int8", "tfidf"])
    assert errors == {}
    assert sorted(hub["calls"]) == ["bertturk_int8/config.json", "bertturk_int8/model_int8.pt",
                                    "tfidf/meta.json", "tfidf/model.joblib"]
    assert (hub["local"] / "bertturk_int8/model_int8.pt").read_bytes() == \
        hub["files"]["bertturk_int8/model_int8.pt"]
    assert not (hub["local"] / "w2v").exists()


def test_no_network_when_files_present(hub):
    assert run(hub, ["tfidf"]) == {}
    hub["calls"].clear()

    def must_not_call(*_a):
        raise AssertionError("dosyalar yerelde varken indirme yapılmamalı")
    assert run(hub, ["tfidf"], downloader=must_not_call) == {}


def test_rule_needs_nothing(hub):
    def must_not_call(*_a):
        raise AssertionError("rule modeli için indirme yapılmamalı")
    assert run(hub, ["rule"], downloader=must_not_call) == {}


def test_tampered_download_is_rejected_and_deleted(hub):
    def evil(repo_id, filename, revision, local_dir):
        dst = Path(local_dir) / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        data = hub["files"][filename]
        dst.write_bytes(b"X" * len(data))        # aynı boyut, farklı içerik
        return dst
    errors = run(hub, ["tfidf"], downloader=evil)
    assert "tfidf" in errors and "SHA-256" in errors["tfidf"]
    assert not (hub["local"] / "tfidf/model.joblib").exists()
    assert not (hub["local"] / "tfidf/meta.json").exists()


def test_partial_file_is_redownloaded(hub):
    p = hub["local"] / "nn/model.pt"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"yarim")                       # yarıda kalmış indirme
    assert run(hub, ["nn"]) == {}
    assert p.read_bytes() == hub["files"]["nn/model.pt"]


def test_network_error_is_reported_not_raised(hub):
    def offline(*_a):
        raise OSError("internet yok")
    errors = run(hub, ["bert", "tfidf"], downloader=offline)
    assert set(errors) == {"bert", "tfidf"}
    assert "internet yok" in errors["bert"]


def test_missing_manifest(hub, tmp_path):
    errors = ms.ensure_models(["bert_int8"], models_dir=hub["local"],
                              manifest_path=tmp_path / "yok.json",
                              downloader=hub["download"], log=lambda _m: None)
    assert "bert_int8" in errors and "bulunamadı" in errors["bert_int8"]


def test_verify(hub):
    run(hub, ["tfidf", "nn"])
    assert ms.verify(["tfidf", "nn"], models_dir=hub["local"],
                     manifest_path=hub["manifest"]) == []
    (hub["local"] / "nn/model.pt").write_bytes(b"m" * len(hub["files"]["nn/model.pt"]))
    probs = ms.verify(["nn"], models_dir=hub["local"], manifest_path=hub["manifest"])
    assert probs == ["SHA-256 farklı: nn/model.pt"]
    probs = ms.verify(["bert"], models_dir=hub["local"], manifest_path=hub["manifest"])
    assert probs == ["eksik: bertturk/model.safetensors"]


def test_real_manifest_if_present():
    """Repodaki gerçek manifest (yüklemeden sonra) uygulamanın beklediği dosyaları kapsamalı."""
    if not ms.MANIFEST_PATH.exists():
        pytest.skip("models_manifest.json henüz oluşturulmadı")
    man = ms.load_manifest()
    paths = {f["path"] for f in man["files"]}
    for need in ["tfidf/model.joblib", "nn/model.pt", "nn/vocab.json",
                 "bertturk/model.safetensors", "bertturk_int8/model_int8.pt",
                 "bertturk_int8/tokenizer.json"]:
        assert need in paths, need
    assert "/" in man["repo_id"]
