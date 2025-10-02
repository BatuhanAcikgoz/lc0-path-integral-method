# LC0 Path Integral vs engine.py Karşılaştırma Raporu

**Rapor Tarihi**: 2025-10-02 16:56:43

## Özet

- **Toplam Test**: 8
- **LC0 Başarı Oranı**: 100.0%
- **engine.py Başarı Oranı**: 100.0%
- **Ortalama Hızlanma**: 5280.36x (LC0 lehine)
- **Hamle Benzerliği**: 62.5%

## Performans Analizi

- **LC0 Ortalama Süre**: 1.22 saniye
- **engine.py Ortalama Süre**: 232.22 saniye
- **Medyan Hızlanma**: 989.19x

## Test Pozisyonları

### Opening
- **Açıklama**: Başlangıç pozisyonu
- **FEN**: `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1`

## Test Parametreleri

**Eşit Şartlar**: Her iki sistem de depth=4 (LC0: ~26,000 nodes) ile test edilmiştir.

1. Lambda: 0.01, Samples: 25, Mode: competitive
2. Lambda: 0.05, Samples: 50, Mode: competitive
3. Lambda: 0.1, Samples: 50, Mode: competitive
4. Lambda: 0.1, Samples: 100, Mode: quantum_limit, Reward: policy
5. Lambda: 0.1, Samples: 100, Mode: quantum_limit, Reward: cp_score
6. Lambda: 0.1, Samples: 100, Mode: quantum_limit, Reward: hybrid
7. Lambda: 0.2, Samples: 75, Mode: competitive
8. Lambda: 0.5, Samples: 100, Mode: competitive

## Sonuç

Gömülü LC0 Path Integral sistemi, orijinal engine.py implementasyonuna göre **5280.36x daha hızlı** çalışmaktadır. Hamle benzerliği **62.5%** ile orta seviyededir.