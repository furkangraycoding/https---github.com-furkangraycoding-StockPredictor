import pandas as pd
import numpy as np
from ml_engine import MLEngine
from data_loader import load_data
from indicators import TechnicalAnalyzer

def compare_thresholds():
    # 1. Load and Prepare
    df, date_col = load_data('BIST100_PREDICTION_READY.csv')
    analyzer = TechnicalAnalyzer(df)
    analyzer.add_rsi()
    analyzer.add_zigzag_labels()
    df = analyzer.add_derived_features()
    
    engine = MLEngine(df)
    engine.train()
    df_all = engine.add_predictions_to_df(engine.df.copy())
    
    df_all['gap'] = (df_all['AI_Peak_Prob'] - df_all['AI_Dip_Prob']) * 100
    df_all['is_exact_peak'] = df_all['Tepe'].notna()

    def get_metrics(threshold_val):
        df_all['threshold'] = df_all['RSI'] * threshold_val
        df_all['is_pure'] = (df_all['AI_Peak_Prob'] > 0.85) & (df_all['gap'] > df_all['threshold'])
        
        pure_days = df_all[df_all['is_pure'] == True]
        total_signals = len(pure_days)
        if total_signals == 0: return 0, 0, 0, 0
        
        # Precision (Hassasiyet) - Label_Peak smeared map
        tp = df_all[(df_all['is_pure'] == True) & (df_all['Label_Peak'] == 1)].shape[0]
        precision = (tp / total_signals) * 100
        
        # Near Hits (+/- 3 days)
        near_hits = 0
        for idx in pure_days.index:
            loc = df_all.index.get_loc(idx)
            window = df_all.iloc[max(0, loc-3) : min(len(df_all), loc+4)]
            if window['is_exact_peak'].any():
                near_hits += 1
        
        near_hit_rate = (near_hits / total_signals) * 100
        
        return total_signals, tp, precision, near_hit_rate

    print("\n" + "="*60)
    print(f"{'Eşik Değeri':<15} | {'Sinyal Sayısı':<15} | {'Hassasiyet (%)':<15} | {'+/- 3G Başarı (%)'}")
    print("-" * 60)
    
    for t in [0.48, 0.47, 0.45]:
        sig_count, tp, prec, near = get_metrics(t)
        print(f"{t:<15.2f} | {sig_count:<15} | {prec:<15.2f} | {near:.2f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    compare_thresholds()
