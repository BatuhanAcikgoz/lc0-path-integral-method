
# LC0 Path Integral – AI Prompt Rules (Prompt Rules for Engine Integration)

## 1. Amaç

* LC0 motoruna gömülecek **Path Integral Sampling** metodunu tanımlamak ve motorun doğru bir şekilde çalışmasını sağlamak.
* Python tarafındaki örnekleme mantığı ve log-sum-exp softmax fonksiyonu motor içinde de kullanılacak.
* Motor, tek başına veya toplu sampling için uyumlu olacak.


## 2. UCI Seçenekleri

1. **PathIntegralLambda**

    * Default: 0.1
    * Min: 0.001
    * Max: 10.0
    * Açıklama: Softmax sıcaklığı (λ). Küçük λ → keşif (exploration), büyük λ → keskin seçim (exploitation).

2. **PathIntegralSamples**

    * Default: 50
    * Min: 1
    * Max: 100000
    * Açıklama: Root node’da örneklenen yol sayısı (sample size).

3. **PathIntegralRewardMode**

    * Default: hybrid
    * Values: policy, cp_score, hybrid
    * Açıklama: Ödül tipi.

        * policy → sadece policy head olasılıkları
        * cp_score → sadece centipawn score
        * hybrid → policy * softmax(cp_score)

4. **PathIntegralMode**  

    * Default: competitive
    * Values: competitive, quantum_limit
    * Açıklama: Sampling modu.
   
        * competitive → motorun arama yeteneklerini kullanarak optimal hamleyi bulur.
        * quantum_limit → her yol için policy ve value head kullanılır, ödül modu uygulanır.
---

## 3. Softmax Kuralları

* Motor, Python tarafındaki **log-sum-exp softmax** mantığını kullanacak.
* Formül:

```text
scores: ham centipawn veya policy değerleri
lam: PathIntegralLambda

1. arr_scaled = (scores - max(scores)) * lam
2. log_sum_exp = log(sum(exp(arr_scaled)))
3. probs = exp(arr_scaled - log_sum_exp)
```

* Çıkış: normalize edilmiş olasılık dağılımı.
* Güvenlik: `nan` veya negatif toplam durumunda eşit olasılık verilecek.

---

## 4. Path Integral Sampling Akışı

1. Root node’da N adet yol örneklenir (PathIntegralSamples).
2. Competitive ve Quantum_limit adında 2 tane mod bulunur.
3. Competitive mod motorun arama yeteneklerini kullanarak ve path integral sampling ile optimal hamleyi bulur. Ve competitive modda ödül modu yoktur direkt motorun arama yeteneklerinden yararlanır ve hem motorun temperature ayarından yararlanır hem de path integralin softmax'ından yararlanır tıpkı engine.py'deki gibi.
3. Quantum_limit modunda her yol için:
    * Motor **policy head** ve **value head** kullanılır.
    * Ödül modu (`policy`, `cp_score`, `hybrid`) uygulanır.
    * Hamleler softmax(λ) ile normalize edilir.
    * Yol ve değer kaydedilir.
4. Sonuçlar JSON/CSV veya UCI üzerinden dışa aktarılır.

---

## 5. Quantum Limit Ödül Modları

* **policy**: Sadece policy olasılıkları.
* **cp_score**: Sadece centipawn değerleri.
* **hybrid**: Policy olasılıkları * softmax(cp_score).

---

## 6. Lambda Tarama Kuralları

* Lambda değeri `PathIntegralLambda` ile belirlenir.
* Farklı λ değerleri motor içinde veya dış tarafta `setoption` komutu ile taranabilir.
* Küçük λ → geniş dağılım, yüksek λ → keskin olasılıklar.

---

## 7. Motor Entegrasyonu ve Performans

* **Root node sampling** yapılacak; MCTS state korunmalı.
* **Depth → nodes** dönüşümü adaptif olacak.
* Sampling **GPU üzerinde paralel** yapılacak.
* Tek motor, toplu sampling ile **yüksek performans** sağlanacak.
* Motor, Python tarafındaki mantığı doğrudan implement edecek; ayrıca cache ve log mekanizması kullanılabilir.
* Motor, debug ayarları açıkken performans metrikleri gibi her yolun ( sample'ın ) ne kadar sürede hesaplandığını gösterecek.

---

## 8. Güvenlik ve Stabilite

* Softmax hesaplamasında `NaN` veya `Inf` çıkarsa eşit olasılık ver.
* Bozuk policy veya score değerlerinde fallback uniform probability uygulanacak.
* Motorun crash olmaması için path örneklemede exception handling yapılmalı.

---