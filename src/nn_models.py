"""Alan A — Kendi Word2Vec gömmelerimizle beslenen sinir ağları.

İki mimari tanımlıdır; hangisinin kullanılacağı 03_word2vec_nn.py içinde
YALNIZCA dev macro-F1 ile seçilir:
  * TextCNN  — 1D evrişim (çekirdek 2/3/4) + global max-pooling + dropout (Kim, 2014)
  * MeanFF   — gömmelerin maskeli ortalaması → gizli katman → çıktı (ileri beslemeli)
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

PAD, UNK = "<pad>", "<unk>"


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, dim: int, weights: np.ndarray | None = None,
                 num_filters: int = 100, kernel_sizes=(2, 3, 4), dropout: float = 0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        if weights is not None:
            self.emb.weight.data.copy_(torch.as_tensor(weights))
        self.convs = nn.ModuleList(nn.Conv1d(dim, num_filters, k) for k in kernel_sizes)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 2)
        self.min_len = max(kernel_sizes)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids).transpose(1, 2)                      # (B, dim, T)
        h = [torch.relu(c(x)).max(dim=2).values for c in self.convs]
        return self.fc(self.drop(torch.cat(h, dim=1)))


class MeanFF(nn.Module):
    def __init__(self, vocab_size: int, dim: int, weights: np.ndarray | None = None,
                 hidden: int = 128, dropout: float = 0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        if weights is not None:
            self.emb.weight.data.copy_(torch.as_tensor(weights))
        self.ff = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(hidden, 2))
        self.min_len = 1

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        mask = (ids != 0).unsqueeze(-1).float()
        mean = (self.emb(ids) * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.ff(mean)


ARCHS = {"cnn": TextCNN, "ff": MeanFF}


def encode_batch(token_lists, vocab: dict, max_len: int, min_len: int) -> torch.Tensor:
    """Token listelerini sıfırla doldurulmuş id tensörüne çevirir."""
    unk = vocab[UNK]
    seqs = [[vocab.get(t, unk) for t in toks[:max_len]] or [unk] for toks in token_lists]
    T = max(min_len, max(len(s) for s in seqs))
    out = torch.zeros(len(seqs), T, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out
