import pandas as pd
import numpy as np
from ml_engine import MLEngine
from data_loader import load_data
from indicators import TechnicalAnalyzer

def analyze_final_peak_performance():
    # 1. Veriyi Yükle ve Hazırla
    df, date_col = load_data('BIST100_PREDICTION_READY.csv')
    analyzer = TechnicalAnalyzer(df)
    analyzer.add_rsi()
    analyzer.add_zigzag_labels()
    df = analyzer.add_derived_features()
    
    engine = MLEngine(df)
    engine.train()
    df_all = engine.add_predictions_to_df(engine.df.copy())
    
    # 2. SAF TEPE MANTIĞI (Final: 85% Prob + 0.48x RSI Gap)
    df_all['peak_gap'] = (df_all['AI_Peak_Prob'] - df_all['AI_Dip_Prob']) * 100
    df_all['peak_threshold'] = df_all['RSI'] * 0.48
    
    df_all['is_pure_peak'] = (df_all['AI_Peak_Prob'] >= 0.85) & (df_all['peak_gap'] > df_all['peak_threshold'])
    df_all['is_exact_peak'] = df_all['Tepe'].notna()
    
    # 3. İstatistikleri Hesapla
    total_actual_peaks = df_all['is_exact_peak'].sum()
    pure_peak_signals = df_all[df_all['is_pure_peak'] == True]
    total_signals = len(pure_peak_signals)
    
    exact_hits = 0
    near_hits_1g = 0
    near_hits_2g = 0 # +/- 2 Gün
    near_hits_3g = 0 
    
    hit_indices = set() # Kaç tane benzersiz tepe yakalandı?
    
    for idx in pure_peak_signals.index:
        loc = df_all.index.get_loc(idx)
        
        # +/- 2 Gün Penceresi
        window_2g = df_all.iloc[max(0, loc-2) : min(len(df_all), loc+3)]
        if window_2g['is_exact_peak'].any():
            near_hits_2g += 1
            peaks_in_window = window_2g[window_2g['is_exact_peak']].index
            for p_idx in peaks_in_window:
                hit_indices.add(p_idx)
                
        # Diğer standart metrikler
        if df_all.iloc[loc]['is_exact_peak']:
            exact_hits += 1
        
        window_1g = df_all.iloc[max(0, loc-1) : min(len(df_all), loc+2)]
        if window_1g['is_exact_peak'].any():
            near_hits_1g += 1
            
        window_3g = df_all.iloc[max(0, loc-3) : min(len(df_all), loc+4)]
        if window_3g['is_exact_peak'].any():
            near_hits_3g += 1

    # Recall
    unique_peaks_captured = len(hit_indices)
    recall_rate = (unique_peaks_captured / total_actual_peaks * 100) if total_actual_peaks > 0 else 0

    print("\n" + "="*60)
    print("🚩 SAF TEPE SİNYALİ FİNAL RAPORU (0.48 Eşik)")
    print("="*60)
    print(f"Piyasadaki Toplam Gerçek Tepe Sayısı : {int(total_actual_peaks)}")
    print(f"Üretilen Toplam Saf Tepe Sinyali     : {total_signals}")
    print("-" * 60)
    print(f"📍 TAM ÜSTÜNE (Exact Hit)            : %{(exact_hits/total_signals*100) if total_signals > 0 else 0:.2f}")
    print(f"🎯 +/- 1 GÜN BAŞARISI                : %{(near_hits_1g/total_signals*100) if total_signals > 0 else 0:.2f}")
    print(f"🔥 +/- 2 GÜN BAŞARISI (Kritik Metrik): %{(near_hits_2g/total_signals*100) if total_signals > 0 else 0:.2f}")
    print(f"🛡️ +/- 3 GÜN BAŞARISI                : %{(near_hits_3g/total_signals*100) if total_signals > 0 else 0:.2f}")
    print("-" * 60)
    print(f"📉 RECALL (Tepe Yakalama Oranı)      : %{recall_rate:.2f}")
    print(f"   (Piyasadaki her 100 tepenin {unique_peaks_captured}'si yakalandı)")
    print("-" * 60)
    print("\n🔍 ANALİZ VE YORUM:")
    if total_signals > 0:
        print(f"Tepe sinyallerinin %{(near_hits_2g/total_signals*100):.1f}'si zirvenin +/- 2 günlük menzilindedir.")
    print("Tepe noktaları genellikle dipten daha geniş bir alana yayıldığı için")
    print("nokta atışı başarısı dip modeline göre daha düşüktür.")
    print("="*60 + "\n")

if __name__ == "__main__":
    analyze_final_peak_performance()
