import pandas as pd
import numpy as np
from ml_engine import MLEngine
from data_loader import load_data
from indicators import TechnicalAnalyzer

def test_dip_thresholds():
    # 1. Load and Prepare
    df, date_col = load_data('BIST100_PREDICTION_READY.csv')
    analyzer = TechnicalAnalyzer(df)
    analyzer.add_rsi()
    analyzer.add_zigzag_labels()
    df = analyzer.add_derived_features()
    
    engine = MLEngine(df)
    engine.train()
    df_all = engine.df.copy()
    df_all = engine.add_predictions_to_df(df_all)
    
    # 2. Results DF
    df_all['dip_gap'] = (df_all['AI_Dip_Prob'] - df_all['AI_Peak_Prob']) * 100
    df_all['is_exact_dip'] = df_all['Dip'].notna()

    def get_metrics(multiplier):
        df_all['dip_threshold'] = df_all['RSI'] * multiplier
        df_all['is_pure_dip'] = (df_all['AI_Dip_Prob'] > 0.85) & (df_all['dip_gap'] > df_all['dip_threshold'])
        
        pure_days = df_all[df_all['is_pure_dip'] == True]
        total = len(pure_days)
        if total == 0: return 0, 0, 0, 0
        
        # Precision (Smeared Label_Dip)
        tp = df_all[(df_all['is_pure_dip'] == True) & (df_all['Label_Dip'] == 1)].shape[0]
        precision = (tp / total) * 100
        
        # Near Hits (+/- 3 days)
        near_hits = 0
        for idx in pure_days.index:
            loc = df_all.index.get_loc(idx)
            window = df_all.iloc[max(0, loc-3) : min(len(df_all), loc+4)]
            if window['is_exact_dip'].any():
                near_hits += 1
        
        near_hit_rate = (near_hits / total) * 100
        return total, tp, precision, near_hit_rate

    print("\n" + "="*65)
    print(f"{'Dip Çarpanı':<15} | {'Sinyal Sayısı':<15} | {'Hassasiyet (%)':<15} | {'+/- 3G Başarı (%)'}")
    print("-" * 65)
    
    for m in [0.8, 1.0, 1.2, 1.4, 1.6]:
        count, tp, prec, near = get_metrics(m)
        print(f"{m:<15.1f} | {count:<15} | {prec:<15.2f} | {near:.2f}")
    print("="*65 + "\n")

if __name__ == "__main__":
    test_dip_thresholds()
