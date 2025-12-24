# Yeni Feature'lar - Dip/Zirve Tahmin Sistemi

## 📊 Eklenen Yeni Feature'lar

### 1. **Cycle Length Features** (Döngü Uzunluğu)
- **`Cycle_Length`**: Mevcut döngünün uzunluğu (son pivot'tan bu yana geçen gün sayısı)
- **`Avg_Cycle_Length`**: Son 5 döngünün ortalama uzunluğu
- **`Cycle_Length_Ratio`**: Mevcut döngü uzunluğu / Ortalama döngü uzunluğu
  - > 1.2: Döngü ortalamadan %20+ uzun (tükenme sinyali)
  - < 0.8: Döngü ortalamadan kısa

**Kullanım Amacı**: Döngülerin uzunluğunu takip ederek, uzun süren trendlerin tükenme noktalarını tespit eder.

---

### 2. **Price Exhaustion Features** (Fiyat Tükenmesi)
- **`Price_Exhaustion_5D`**: Son 5 günde fiyat değişimi (%)
- **`Price_Exhaustion_10D`**: Son 10 günde fiyat değişimi (%)
- **`Price_Exhaustion_20D`**: Son 20 günde fiyat değişimi (%)
- **`Price_Exhaustion_5D_ATR`**: ATR'ye normalize edilmiş 5 günlük fiyat tükenmesi
- **`Price_Range_10D`**: Son 10 günde fiyat aralığı (high-low) / low

**Kullanım Amacı**: Fiyatın son dönemde ne kadar hareket ettiğini ölçer. Aşırı yükselişler/düşüşler genellikle dönüş sinyali verir.

---

### 3. **Momentum Decay Features** (Momentum Tükenmesi)
- **`RSI_Decay_1D`**: RSI'nın 1 günde düşüş hızı (negatif diff)
- **`RSI_Decay_3D`**: RSI'nın 3 günde ortalama düşüş hızı
- **`RSI_Decay_5D`**: RSI'nın 5 günde ortalama düşüş hızı
- **`RSI_Decay_Accel`**: RSI düşüş hızının ivmesi (hızlanma/yavaşlama)

**Kullanım Amacı**: Zirve sonrası momentum kaybını ölçer. Hızlı RSI düşüşü, trend değişiminin erken sinyali olabilir.

---

### 4. **Volatility Expansion Features** (Volatilite Genişlemesi)
- **`ATR_Expansion_5D`**: ATR'nin son 5 günde değişim oranı
- **`ATR_Expansion_10D`**: ATR'nin son 10 günde değişim oranı
- **`ATR_vs_Avg`**: Mevcut ATR / Son 20 günlük ortalama ATR
  - > 1.5: Volatilite ortalamadan %50+ yüksek (belirsizlik artışı)
  - < 0.7: Volatilite düşük (sakin piyasa)

**Kullanım Amacı**: Volatilite artışı genellikle trend değişimlerinin habercisidir.

---

### 5. **Exhaustion Score Features** (Tükenme Skorları)
- **`Trend_Exhaustion_Score`**: Zirve tükenme skoru (0-4)
  - RSI > 75: +1
  - RSI Overbought Days > 5: +1
  - Price Exhaustion 10D > 5%: +1
  - Cycle Length Ratio > 1.2: +1
  - **3-4 puan**: Güçlü zirve tükenme sinyali

- **`Dip_Exhaustion_Score`**: Dip tükenme skoru (0-4)
  - RSI < 25: +1
  - RSI Oversold Days > 3: +1
  - Price Exhaustion 10D < -5%: +1
  - Cycle Length Ratio > 1.2: +1
  - **3-4 puan**: Güçlü dip tükenme sinyali

**Kullanım Amacı**: Birden fazla tükenme sinyalini birleştirerek daha güvenilir sinyaller üretir.

---

### 6. **Blowoff Features** (Aşırı Hareket)
- **`Price_Blowoff`**: Son 20 günde en yüksek %5'lik fiyat hareketleri (binary)
- **`Volume_Blowoff`**: Fiyat blowoff + Hacim spike (>1.5x ortalama) (binary)

**Kullanım Amacı**: Aşırı fiyat hareketleri genellikle trend sonlarını işaret eder.

---

### 7. **Oversold Duration** (Aşırı Satım Süresi)
- **`RSI_Oversold_Days`**: RSI < 30 bölgesinde art arda kaç gün kalındığı
  - Mevcut: `RSI_Overbought_Days` (RSI > 70 için)
  - Yeni: `RSI_Oversold_Days` (RSI < 30 için)

**Kullanım Amacı**: Uzun süre aşırı satım bölgesinde kalmak, dip yaklaşımını gösterir.

---

## 🎯 Feature Kullanım Stratejisi

### Peak Model (Zirve Tahmini)
Yeni feature'lar peak model'e eklendi:
- `Cycle_Length_Ratio`: Uzun döngüler tükenme sinyali
- `RSI_Decay_*`: Momentum kaybı
- `ATR_Expansion_*`: Volatilite artışı
- `Trend_Exhaustion_Score`: Kompozit tükenme skoru
- `Price_Blowoff`, `Volume_Blowoff`: Aşırı hareketler

### Dip Model (Dip Tahmini)
Tüm yeni feature'lar dip model'de kullanılabilir:
- `Dip_Exhaustion_Score`: Dip tükenme skoru
- `RSI_Oversold_Days`: Aşırı satım süresi
- `Price_Exhaustion_*`: Fiyat düşüş tükenmesi

---

## 📈 Örnek Kullanım Senaryoları

### Senaryo 1: Zirve Tespiti
```
RSI > 75
+ RSI_Overbought_Days > 5
+ Price_Exhaustion_10D > 5%
+ Cycle_Length_Ratio > 1.2
+ Trend_Exhaustion_Score = 4
→ Güçlü Zirve Sinyali
```

### Senaryo 2: Dip Tespiti
```
RSI < 25
+ RSI_Oversold_Days > 3
+ Price_Exhaustion_10D < -5%
+ ATR_Expansion_10D > 0.2 (volatilite artışı)
+ Dip_Exhaustion_Score = 4
→ Güçlü Dip Sinyali
```

### Senaryo 3: Momentum Tükenmesi
```
RSI_Decay_5D > 5 (RSI 5 günde 5+ puan düştü)
+ RSI_Decay_Accel > 0 (hızlanan düşüş)
+ Price_Blowoff = 1
→ Trend Değişimi Yakın
```

---

## 🔧 Teknik Detaylar

### Hesaplama Sırası
1. `add_zigzag_labels()` - Pivot noktaları belirlenir
2. `add_time_features()` - Cycle_Length hesaplanır
3. `add_cycle_exhaustion_features()` - Tüm yeni feature'lar hesaplanır

### Feature Normalizasyonu
- Tüm feature'lar NaN değerler için 0 ile doldurulur
- Cycle length hesaplamaları integer index kullanır
- ATR normalizasyonu fiyat değişkenliğini hesaba katar

---

## 📊 Feature Importance Beklentisi

**Yüksek Önem Beklenen Feature'lar:**
1. `Trend_Exhaustion_Score` / `Dip_Exhaustion_Score` - Kompozit sinyaller
2. `Cycle_Length_Ratio` - Döngü analizi
3. `RSI_Decay_*` - Momentum kaybı
4. `Price_Exhaustion_10D` - Fiyat tükenmesi
5. `ATR_vs_Avg` - Volatilite genişlemesi

**Orta Önem Beklenen:**
- `Price_Blowoff`, `Volume_Blowoff` - Aşırı hareketler
- `RSI_Oversold_Days` - Aşırı satım süresi

---

## 🚀 Sonraki Adımlar

1. **Model Eğitimi**: Yeni feature'larla modeli yeniden eğit
2. **Feature Importance**: Hangi feature'ların en önemli olduğunu analiz et
3. **Backtesting**: Yeni feature'ların performansını test et
4. **Hyperparameter Tuning**: Yeni feature'lara göre model parametrelerini optimize et

---

## 📝 Notlar

- Tüm feature'lar `ml_engine.py`'deki `all_features` listesine eklendi
- Peak model'e seçili exhaustion feature'ları eklendi
- Dip model tüm yeni feature'ları kullanabilir
- Feature'lar otomatik olarak `add_derived_features()` içinde hesaplanır

