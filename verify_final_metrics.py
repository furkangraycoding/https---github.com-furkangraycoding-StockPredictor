import pandas as pd
import numpy as np
from ml_engine import MLEngine
from data_loader import load_data
from indicators import TechnicalAnalyzer

def calculate_final_metrics():
    # 1. Veriyi Yükle ve Hazırla
    df, date_col = load_data('BIST100_PREDICTION_READY.csv')
    
    # Teknik özellikleri hesapla (Eğitim öncesi gerekli)
    analyzer = TechnicalAnalyzer(df)
    analyzer.add_rsi()
    analyzer.add_zigzag_labels()
    df = analyzer.add_derived_features()
    
    # Modeli eğit
    engine = MLEngine(df)
    engine.train()
    
    # Tahminleri ve olasılıkları tüm tabloya ekle
    df_all = engine.add_predictions_to_df(engine.df.copy())
    
    # 2. "SAF ZİRVE SİNYALİ" Mantığını Uygula (Sizin Formülünüz)
    # Gap > RSI * 0.45 ve Peak_Prob > 0.85
    df_all['gap'] = (df_all['AI_Peak_Prob'] - df_all['AI_Dip_Prob']) * 100
    df_all['purity_threshold'] = df_all['RSI'] * 0.45
    df_all['is_pure_signal'] = (df_all['AI_Peak_Prob'] > 0.85) & (df_all['gap'] > df_all['purity_threshold'])
    
    # 3. Metrikleri Hesapla
    # TP: Sinyal Var ve O gün bir Peak (Label_Peak == 1)
    # FP: Sinyal Var ama Peak Değil (Label_Peak == 0)
    # FN: Peak Var ama Sinyal Yok (Label_Peak == 1 but Signal == 0)
    # TN: Sinyal Yok ve Peak Değil (Label_Peak == 0 and Signal == 0)
    
    tp = df_all[(df_all['is_pure_signal'] == True) & (df_all['Label_Peak'] == 1)].shape[0]
    fp = df_all[(df_all['is_pure_signal'] == True) & (df_all['Label_Peak'] == 0)].shape[0]
    fn = df_all[(df_all['is_pure_signal'] == False) & (df_all['Label_Peak'] == 1)].shape[0]
    tn = df_all[(df_all['is_pure_signal'] == False) & (df_all['Label_Peak'] == 0)].shape[0]
    
    total_signals = tp + fp
    total_actual_peaks = tp + fn
    
    precision = (tp / total_signals * 100) if total_signals > 0 else 0
    recall = (tp / total_actual_peaks * 100) if total_actual_peaks > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # False Positive Oranı (FP / Toplam Negatif Gün)
    fpr = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0
    
    print("\n" + "="*40)
    print("🎯 SAF ZİRVE SİNYALİ - PERFORMANS RAPORU")
    print("="*40)
    print(f"Toplam Veri Günü         : {len(df_all)}")
    print(f"Üretilen Toplam Sinyal   : {total_signals}")
    print(f"Gerçekleşen Toplam Zirve  : {total_actual_peaks} (Smearing Dahil)")
    print("-" * 40)
    print(f"✅ Doğru Sinyal (TP)     : {tp}")
    print(f"❌ Hatalı Sinyal (FP)    : {fp}")
    print(f"🔘 Kaçırılan Zirve (FN)  : {fn}")
    print("-" * 40)
    print(f"🚀 Hassasiyet (Precision) : %{precision:.2f}")
    print(f"🔍 Yakalama Oranı (Recall): %{recall:.2f}")
    print(f"📉 Hatalı Sinyal Oranı   : %{fpr:.2f} (Tüm günlere göre)")
    print(f"🏆 F1-Skoru              : {f1/100:.3f}")
    print("-" * 40)
    print("\n[YORUM]:")
    if precision > 80:
        print("- Sinyal kalitesi çok yüksek. Gelen sinyale güvenilebilir.")
    if fpr < 5:
        print("- Hatalı sinyal üretme riski (False Alarm) çok düşük.")
    if recall < 50:
        print("- Model çok seçici, bazı ufak zirveleri pas geçiyor olabilir.")
    print("="*40 + "\n")

if __name__ == "__main__":
    calculate_final_metrics()
