"""Alan D — Dinamik INT8 quantization yardımcıları.

Yöntem: PyTorch eager-mode dinamik quantization (torch.ao.quantization.quantize_dynamic).
  * Tüm nn.Linear katmanlarının ağırlıkları INT8'e çevrilir (tensör başına ölçek).
  * Aktivasyonlar çıkarım sırasında dinamik olarak INT8'e ölçeklenir.
  * Gömme (embedding) ve LayerNorm katmanları FP32 kalır.
  * Yalnızca CPU'da çalışır (x86/fbgemm çekirdekleri) — UC4'ün CPU-only hedefiyle uyumlu.

Not: PyTorch dokümantasyonu bu API'yi torchao'ya taşınacak şekilde "deprecated"
olarak işaretlemiştir; kaldırılmadığı sürece çalışır. Kaldırılmışsa import hatası
açık bir mesajla bildirilir.
"""
from __future__ import annotations

import torch
from torch import nn


def _set_engine() -> str:
    """x86 CPU için quantized çekirdek seç.

    Öncelik: x86 / fbgemm (Linux, eski Windows sürümleri), yoksa onednn
    (Intel oneDNN; ör. PyTorch 2.15 Windows CPU paketinde tek seçenek).
    Hepsi dinamik INT8 Linear'ı destekler; seçilen motor meta.json'a kaydedilir.
    """
    engines = torch.backends.quantized.supported_engines
    for e in ("x86", "fbgemm", "onednn"):
        if e in engines:
            torch.backends.quantized.engine = e
            return e
    raise RuntimeError(f"Desteklenen x86 quantized engine yok: {engines}")


def quantize_dynamic_int8(model: nn.Module) -> nn.Module:
    try:
        from torch.ao.quantization import quantize_dynamic
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Bu PyTorch sürümünde torch.ao.quantization.quantize_dynamic yok. "
            "Alternatif: ONNX Runtime dinamik quantization (bkz. README)."
        ) from e
    import warnings

    _set_engine()
    model = model.to("cpu").eval()
    with warnings.catch_warnings():  # bilinen "deprecated" uyarıları çıktıyı boğmasın
        warnings.filterwarnings("ignore", message=".*deprecated.*", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*deprecated.*", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message=".*deprecated.*", category=FutureWarning)
        return quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def count_quantized_linear(model: nn.Module) -> tuple[int, int]:
    """(INT8 dinamik Linear sayısı, kalan FP32 Linear sayısı) — gerçek backend kanıtı."""
    import torch.ao.nn.quantized.dynamic as nnqd

    q = sum(isinstance(m, nnqd.Linear) for m in model.modules())
    f = sum(type(m) is nn.Linear for m in model.modules())
    return q, f
