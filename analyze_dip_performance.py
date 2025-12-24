import pandas as pd
import numpy as np
from ml_engine import MLEngine
from data_loader import load_data
from indicators import TechnicalAnalyzer

def analyze_dip_performance():
    # 1. Veriyi Yükle ve Hazırla
    df, date_col = load_data('BIST100_PREDICTION_READY.csv')
    analyzer = TechnicalAnalyzer(df)
    analyzer.add_rsi()
    analyzer.add_zigzag_labels()
    df = analyzer.add_derived_features()
    
    engine = MLEngine(df)
    engine.train()
    df_all = engine.add_predictions_to_df(engine.df.copy())
    
    # 2. "SAF DİP SİNYALİ" Mantığını Uygula (multiplier: 1.4)
    df_all['dip_gap'] = (df_all['AI_Dip_Prob'] - df_all['AI_Peak_Prob']) * 100
    df_all['dip_threshold'] = df_all['RSI'] * 1.4
    df_all['is_pure_dip'] = (df_all['AI_Dip_Prob'] > 0.85) & (df_all['dip_gap'] > df_all['dip_threshold'])
    df_all['is_exact_dip'] = df_all['Dip'].notna()
    
    # 3. Metrikleri Hesapla
    pure_dips = df_all[df_all['is_pure_dip'] == True]
    total_pure_dips = len(pure_dips)
    
    exact_hits = 0
    near_hits_3g = 0 # +/- 3 Gün
    near_hits_1g = 0 # +/- 1 Gün
    
    for idx in pure_dips.index:
        loc = df_all.index.get_loc(idx)
        
        # Tam gün mü?
        if df_all.iloc[loc]['is_exact_dip']:
            exact_hits += 1
            
        # +/- 1 Gün Penceresi
        window_1g = df_all.iloc[max(0, loc-1) : min(len(df_all), loc+2)]
        if window_1g['is_exact_dip'].any():
            near_hits_1g += 1

        # +/- 3 Gün Penceresi
        window_3g = df_all.iloc[max(0, loc-3) : min(len(df_all), loc+4)]
        if window_3g['is_exact_dip'].any():
            near_hits_3g += 1

    # Genel İsabet (Smeared Label_Dip üzerinden Precision)
    tp = df_all[(df_all['is_pure_dip'] == True) & (df_all['Label_Dip'] == 1)].shape[0]
    precision = (tp / total_pure_dips * 100) if total_pure_dips > 0 else 0

    print("\n" + "="*50)
    print("💎 SAF DİP SİNYALİ (1.4 Eşik) - DETAYLI PERFORMANS")
    print("="*50)
    print(f"Toplam Üretilen Saf Dip Sinyali : {total_pure_dips} Gün")
    print("-" * 50)
    print(f"📍 TAM ÜSTÜNE (Exact Day)       : %{(exact_hits/total_pure_dips*100) if total_pure_dips > 0 else 0:.2f}")
    print(f"🎯 ÇOK YAKIN (+/- 1 GÜN)        : %{(near_hits_1g/total_pure_dips*100) if total_pure_dips > 0 else 0:.2f}")
    print(f"🛡️ GÜVENLİ BÖLGE (+/- 3 GÜN)    : %{(near_hits_3g/total_pure_dips*100) if total_pure_dips > 0 else 0:.2f}")
    print("-" * 50)
    print(f"🚀 GENEL İSABET (Bölgesel)       : %{precision:.2f}")
    print(f"❌ TAM HATALI SİNYAL             : %{100 - (near_hits_3g/total_pure_dips*100) if total_pure_dips > 0 else 0:.2f}")
    print("-" * 50)
    print("\n🔍 ÖZET YORUM:")
    print(f"Sinyallerin %{(near_hits_1g/total_pure_dips*100) if total_pure_dips > 0 else 0:.1f}'i gerçek dibin sadece 1 gün uzağındadır.")
    print("="*50 + "\n")

if __name__ == "__main__":
    analyze_dip_performance()
