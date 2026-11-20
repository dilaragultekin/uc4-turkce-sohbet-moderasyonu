| Model | Test macro-F1 | Boyut (MB) | Gecikme ort. (ms) | Gecikme p95 (ms) | Bütçe (p95 ≤ 50 ms) |
|---|---|---|---|---|---|
| Kural: sözlük + edit distance | 0.7131 | 0.2 | 0.14 | 0.35 | ✓ |
| TF-IDF + Lojistik Regresyon | 0.7697 | 6.9 | 1.21 | 2.18 | ✓ |
| Word2Vec + Sinir Ağı | 0.7147 | 25.8 | 0.37 | 0.61 | ✓ |
| BERTurk (FP32) | 0.8114 | 442.5 | 22.15 | 40.11 | ✓ |
| BERTurk (INT8 quantized) | 0.8177 | 186.0 | 16.40 | 22.27 | ✓ |

Donanım: 13th Gen Intel(R) Core(TM) i7-13650HX, 4 CPU thread, Windows-11-10.0.26200-SP0; gecikme = 200 tek mesaj çalıştırmasının ortalaması (10 ısınma sonrası), uçtan uca.