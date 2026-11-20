| Model | Test macro-F1 | Boyut (MB) | Gecikme ort. (ms) | Gecikme p95 (ms) | Bütçe (p95 ≤ 50 ms) |
|---|---|---|---|---|---|
| Kural: sözlük + edit distance | 0.7131 | 0.2 | 0.11 | 0.27 | ✓ |
| TF-IDF + Lojistik Regresyon | 0.7697 | 6.9 | 0.96 | 1.26 | ✓ |
| Word2Vec + Sinir Ağı | 0.7147 | 25.8 | 0.26 | 0.37 | ✓ |
| BERTurk (FP32) | 0.8223 | 442.5 | 22.08 | 41.42 | ✓ |
| BERTurk (INT8 quantized) | 0.8150 | 186.0 | 16.10 | 22.83 | ✓ |

Donanım: 13th Gen Intel(R) Core(TM) i7-13650HX, 4 CPU thread, Windows-11-10.0.26200-SP0; gecikme = 200 tek mesaj çalıştırmasının ortalaması (10 ısınma sonrası), uçtan uca.