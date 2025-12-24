# Hyperparameter Tuning & Feature Selection

## 🎯 Yapılan İyileştirmeler

### 1. **Gelişmiş Hyperparameter Tuning**
- **RandomizedSearchCV** ile 30 iterasyon (önceden 20)
- Daha geniş parametre aralığı:
  - `n_estimators`: [200, 300, 400, 500, 700, 1000]
  - `max_depth`: [8, 10, 12, 15, 20, 25, None]
  - `min_samples_leaf`: [1, 2, 4, 6, 8]
  - `min_samples_split`: [2, 5, 10, 15, 20]
  - `max_features`: ["sqrt", "log2", 0.5, 0.7, None]
  - `class_weight`: ["balanced", "balanced_subsample", None]

### 2. **Feature Importance-Based Selection**
- **Tuned model** ile feature importance hesaplanır
- **Importance >= 0.05** olan feature'lar tutulur
- Eğer çok az feature kalırsa (<10), en iyi 10 feature tutulur
- Kaldırılan feature'lar loglanır

### 3. **Yeni Eğitim Akışı**

#### Dip Model:
1. Tüm feature'larla hyperparameter tuning yapılır
2. Tuned model'in feature importance'ları hesaplanır
3. Importance >= 0.05 olan feature'lar seçilir
4. Seçilen feature'larla model yeniden fit edilir

#### Peak Model:
1. Oversampling yapılır (minority class 10x artırılır)
2. Tüm feature'larla hyperparameter tuning yapılır
3. Tuned model'in feature importance'ları hesaplanır
4. Importance >= 0.05 olan feature'lar seçilir
5. Seçilen feature'larla model yeniden fit edilir

---

## 📊 Test Sonuçları

### Feature Reduction:
- **Dip Model**: 44 → 10 feature (77% azalma)
- **Peak Model**: 27 → 10 feature (63% azalma)

### Top Features (Importance > 0.05):

#### Dip Model:
1. **Leg_Return** (0.2373) - Son pivot'tan bu yana fiyat değişimi
2. **Drawdown_Pct** (0.1436) - Son yüksekten düşüş yüzdesi
3. **Price_Exhaustion_5D** (0.1246) - Son 5 günde fiyat tükenmesi
4. **Price_Exhaustion_10D** (0.0963) - Son 10 günde fiyat tükenmesi
5. **Cycle_Length** (0.0828) - Mevcut döngü uzunluğu

#### Peak Model:
1. **ATR** (0.1757) - Average True Range
2. **ATR_Expansion_5D** (0.1462) - Volatilite genişlemesi
3. **RSI_Overbought_Days** (0.1428) - Aşırı alım süresi
4. **Kurtosis_20** (0.1125) - Fiyat dağılımı kurtosis
5. **MFI_14** (0.0968) - Money Flow Index

### Model Performance:
- **Dip Model**: Precision: 0.388, Recall: 0.904, Accuracy: 0.582
- **Peak Model**: Precision: 0.539, Recall: 0.760, Accuracy: 0.534

---

## 🔍 Önemli Gözlemler

### Yeni Feature'ların Etkisi:
- **Cycle_Length_Ratio**: Hem Dip hem Peak model'de seçildi
- **Price_Exhaustion_***: Dip model'de güçlü feature'lar
- **ATR_Expansion_5D**: Peak model'de 2. en önemli feature
- **Trend_Exhaustion_Score**: Seçilmedi (muhtemelen diğer feature'larla korelasyonlu)

### Kaldırılan Feature'lar:
- Düşük importance (< 0.05) olan feature'lar otomatik kaldırıldı
- Model daha hızlı ve daha az overfitting riski

---

## 🚀 Kullanım

### Optimize=True (Önerilen):
```python
engine = MLEngine(df)
metrics, backtest_df = engine.train(optimize=True)
# Hyperparameter tuning + Feature selection yapılır
```

### Optimize=False (Hızlı):
```python
engine = MLEngine(df)
metrics, backtest_df = engine.train(optimize=False)
# Sadece base model ile feature selection yapılır
```

---

## 📝 Notlar

1. **Feature Selection**: Tuned model'in importance'larına göre yapılır, bu daha doğru sonuç verir
2. **Minimum Features**: En az 10 feature tutulur (çok az feature kalırsa)
3. **Logging**: Streamlit'te feature selection süreci loglanır
4. **Performance**: Daha az feature = daha hızlı inference, daha az overfitting

---

## 🔄 Sonraki Adımlar

1. **Feature Importance Analizi**: Hangi feature'ların neden seçildiğini analiz et
2. **Threshold Tuning**: 0.05 threshold'unu optimize et (0.03, 0.07, 0.10)
3. **Cross-Validation**: Feature selection'ın stability'sini test et
4. **Ablation Study**: Her feature'ın katkısını ölç

