import pandas as pd
import numpy as np
from ml_engine import MLEngine
from data_loader import load_data
from indicators import TechnicalAnalyzer

def analyze_peak_hit_distribution():
    # 1. Load and Prepare
    df, date_col = load_data('BIST100_PREDICTION_READY.csv')
    analyzer = TechnicalAnalyzer(df)
    analyzer.add_rsi()
    analyzer.add_zigzag_labels()
    df = analyzer.add_derived_features()
    
    engine = MLEngine(df)
    engine.train()
    df_all = engine.add_predictions_to_df(engine.df.copy())
    
    # 2. Apply Pure Signal Logic
    df_all['gap'] = (df_all['AI_Peak_Prob'] - df_all['AI_Dip_Prob']) * 100
    df_all['threshold'] = df_all['RSI'] * 0.45
    df_all['is_pure'] = (df_all['AI_Peak_Prob'] > 0.85) & (df_all['gap'] > df_all['threshold'])
    
    # Identify exact peak days (Tepe is not NaN)
    df_all['is_exact_peak'] = df_all['Tepe'].notna()
    
    # 3. Analyze where Pure Signals fall
    pure_days = df_all[df_all['is_pure'] == True]
    total_pure = len(pure_days)
    
    exact_hits = 0
    near_hits_window = 0 # Signal within the +/- 3 day window
    pre_peak_hits = 0    # Signal 1-3 days BEFORE the peak (Perfect Exit)
    
    for idx, row in pure_days.iterrows():
        # Check window around this signal for an exact peak
        loc = df_all.index.get_loc(idx)
        
        # Exact Day Hit
        if row['is_exact_peak']:
            exact_hits += 1
            
        # Is there any peak in [t-3, t+3]?
        window = df_all.iloc[max(0, loc-3) : min(len(df_all), loc+4)]
        if window['is_exact_peak'].any():
            near_hits_window += 1
            
        # Is there any peak in [t, t+3]? (Meaning signal is 0-3 days BEFORE or ON the peak)
        early_window = df_all.iloc[loc : min(len(df_all), loc+4)]
        if early_window['is_exact_peak'].any():
            pre_peak_hits += 1

    print("\n" + "="*50)
    print("🎯 SAF ZİRVE SİNYALİ - DAĞILIM ANALİZİ")
    print("="*50)
    print(f"Toplam Üretilen Saf Sinyal Sayısı: {total_pure}")
    print("-" * 50)
    print(f"📍 TAM ÜSTÜNE (Exact Day Hit)     : %{exact_hits/total_pure*100:.2f}  ({exact_hits} gün)")
    print(f"🛡️ GÜVENLİ BÖLGE (+/- 3 Gün)      : %{near_hits_window/total_pure*100:.2f} ({near_hits_window} gün)")
    print(f"💰 ERKEN UYARI (Peak Öncesi 0-3G) : %{pre_peak_hits/total_pure*100:.2f} ({pre_peak_hits} gün)")
    print("-" * 50)
    print(f"Hatalı / Alakasız Sinyal Oranı    : %{ (1 - near_hits_window/total_pure)*100:.2f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    analyze_peak_hit_distribution()
