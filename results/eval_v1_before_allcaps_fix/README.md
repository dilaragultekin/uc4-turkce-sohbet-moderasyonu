# İlk test değerlendirmesi (v1) — büyük harf düzeltmesinden ÖNCE

Bu klasör, `06_benchmark.py`'nin ilk çalıştırmasının (2026-11-19, 21:08) çıktılarını
değiştirilmeden saklar.

İlk değerlendirmeden sonra uygulama denemesinde bir hata bulundu: BERTurk (cased),
tamamı büyük harfle yazılmış hakaretleri kaçırıyordu ("it herif" → saldırgan,
"İT HERİF" → temiz). Düzeltme: BERTurk girdisinde tamamı büyük harf olan tokenlar
Türkçe kurallarla küçültülür (normalizer sürümü 2-allcaps-fold); BERTurk bu
normalizasyonla yeniden eğitildi, eşik yine yalnızca dev setinde seçildi.

Değişikliğin nedeni bir test skoru değil, uygulamada bulunan hatadır. Test sonucu ne
çıkarsa çıksın düzeltilmiş model kullanılacağına önceden karar verilmiştir. Her iki
değerlendirme de raporlanır. Kural, TF-IDF ve Word2Vec+CNN modelleri bu değişiklikten
etkilenmez (girdilerini zaten tamamen küçük harfe çeviriyorlar).
