import pandas as pd
import numpy as np
from ml_engine import MLEngine
from data_loader import load_data
from indicators import TechnicalAnalyzer

def analyze_final_hybrid_dip():
    # 1. Veriyi Yükle ve Hazırla
    df, date_col = load_data('BIST100_PREDICTION_READY.csv')
    analyzer = TechnicalAnalyzer(df)
    analyzer.add_rsi()
    analyzer.add_zigzag_labels()
    df = analyzer.add_derived_features()
    
    engine = MLEngine(df)
    engine.train()
    df_all = engine.add_predictions_to_df(engine.df.copy())
    
    # 2. FİNAL HİBRİT MANTIĞI (85|1.1 veya 72|1.3)
    df_all['gap'] = (df_all['AI_Dip_Prob'] - df_all['AI_Peak_Prob']) * 100
    
    def is_pure_final(row):
        prob = row['AI_Dip_Prob']
        gap = row['gap']
        rsi = row['RSI']
        cond_a = (prob >= 0.85) and (gap > rsi * 1.1)
        cond_b = (prob >= 0.72) and (gap > rsi * 1.3)
        return cond_a or cond_b

    df_all['is_pure_dip'] = df_all.apply(is_pure_final, axis=1)
    df_all['is_exact_dip'] = df_all['Dip'].notna()
    
    # 3. İstatistikleri Hesapla
    total_actual_dips = df_all['is_exact_dip'].sum()
    pure_dip_signals = df_all[df_all['is_pure_dip'] == True]
    total_signals = len(pure_dip_signals)
    
    exact_hits = 0
    near_hits_1g = 0
    near_hits_2g = 0 # +/- 2 Gün
    near_hits_3g = 0 
    
    # Her sinyal için en yakın gerçek dibi bul
    actual_dip_indices = df_all[df_all['is_exact_dip']].index
    
    hit_indices = set() # Kaç tane benzersiz dip yakalandı?
    
    for idx in pure_dip_signals.index:
        loc = df_all.index.get_loc(idx)
        
        # +/- 2 Gün Penceresi (Sizin istediğiniz kritik metrik)
        window_2g = df_all.iloc[max(0, loc-2) : min(len(df_all), loc+3)]
        if window_2g['is_exact_dip'].any():
            near_hits_2g += 1
            # Hangi dipleri yakaladığını takip et (Recall için)
            dips_in_window = window_2g[window_2g['is_exact_dip']].index
            for d_idx in dips_in_window:
                hit_indices.add(d_idx)
                
        # Diğer standart metrikler
        if df_all.iloc[loc]['is_exact_dip']:
            exact_hits += 1
        
        window_1g = df_all.iloc[max(0, loc-1) : min(len(df_all), loc+2)]
        if window_1g['is_exact_dip'].any():
            near_hits_1g += 1
            
        window_3g = df_all.iloc[max(0, loc-3) : min(len(df_all), loc+4)]
        if window_3g['is_exact_dip'].any():
            near_hits_3g += 1

    # Precision
    precision_2g = (near_hits_2g / total_signals * 100) if total_signals > 0 else 0
    
    # RECALL (Toplam diplerin kaçı bu +/- 2G sinyalleriyle yakalandı?)
    unique_dips_captured = len(hit_indices)
    recall_rate = (unique_dips_captured / total_actual_dips * 100) if total_actual_dips > 0 else 0

    print("\n" + "="*60)
    print("🏆 SAF DİP SİNYALİ FİNAL RAPORU (Hibrit %72 Barajı)")
    print("="*60)
    print(f"Piyasadaki Toplam Gerçek Dip Sayısı : {int(total_actual_dips)}")
    print(f"Üretilen Toplam Saf Dip Sinyali     : {total_signals}")
    print("-" * 60)
    print(f"📍 TAM ÜSTÜNE (Exact Hit)            : %{(exact_hits/total_signals*100):.2f}")
    print(f"🎯 +/- 1 GÜN BAŞARISI                : %{(near_hits_1g/total_signals*100):.2f}")
    print(f"🔥 +/- 2 GÜN BAŞARISI (Sizin Metrik) : %{precision_2g:.2f}")
    print(f"🛡️ +/- 3 GÜN BAŞARISI                : %{(near_hits_3g/total_signals*100):.2f}")
    print("-" * 60)
    print(f"📈 RECALL (Dip Yakalama Oranı)        : %{recall_rate:.2f}")
    print(f"   (Piyasadaki her 100 dibin {unique_dips_captured}'si yakalandı)")
    print("-" * 60)
    print("\n🔍 ANALİZ VE YORUM:")
    print(f"Sinyallerin %{precision_2g:.1f}'si gerçek dibin maksimum 2 gün uzağındadır.")
    print(f"Bu sinyallerle piyasadaki diplerin %{recall_rate:.1f}'ini kaçırmadan yakalıyorsunuz.")
    print("="*60 + "\n")

if __name__ == "__main__":
    analyze_final_hybrid_dip()
