import pandas as pd
import numpy as np
from ml_engine import MLEngine
from data_loader import load_data
from indicators import TechnicalAnalyzer

def compare_dip_thresholds():
    # 1. Load data
    df, date_col = load_data('BIST100_PREDICTION_READY.csv')
    analyzer = TechnicalAnalyzer(df)
    analyzer.add_rsi()
    analyzer.add_zigzag_labels()
    df = analyzer.add_derived_features()
    
    engine = MLEngine(df)
    engine.train()
    df_all = engine.add_predictions_to_df(engine.df.copy())
    
    df_all['dip_gap'] = (df_all['AI_Dip_Prob'] - df_all['AI_Peak_Prob']) * 100
    df_all['is_exact_dip'] = df_all['Dip'].notna()

    def get_dip_metrics(multiplier):
        df_all['dip_threshold'] = df_all['RSI'] * multiplier
        df_all['is_pure_dip'] = (df_all['AI_Dip_Prob'] > 0.85) & (df_all['dip_gap'] > df_all['dip_threshold'])
        
        pure_days = df_all[df_all['is_pure_dip'] == True]
        total = len(pure_days)
        if total == 0: return 0, 0, 0, 0
        
        # Exact Day Hit
        exact = df_all[(df_all['is_pure_dip'] == True) & (df_all['is_exact_dip'] == True)].shape[0]
        
        # +/- 1 Day Hit
        near_hits_1g = 0
        for idx in pure_days.index:
            loc = df_all.index.get_loc(idx)
            window = df_all.iloc[max(0, loc-1) : min(len(df_all), loc+2)]
            if window['is_exact_dip'].any():
                near_hits_1g += 1
                
        # +/- 3 Day Hit
        near_hits_3g = 0
        for idx in pure_days.index:
            loc = df_all.index.get_loc(idx)
            window = df_all.iloc[max(0, loc-3) : min(len(df_all), loc+4)]
            if window['is_exact_dip'].any():
                near_hits_3g += 1
        
        return total, (exact/total*100), (near_hits_1g/total*100), (near_hits_3g/total*100)

    print("\n" + "="*70)
    print(f"{'Eşik':<8} | {'Sinyal #':<8} | {'Tam-Gün %':<12} | {'+/- 1G %':<12} | {'+/- 3G %':<12}")
    print("-" * 70)
    
    for m in [1.4, 1.3, 1.2]:
        count, exact, near1, near3 = get_dip_metrics(m)
        print(f"{m:<8.1f} | {count:<8} | {exact:<12.2f} | {near1:<12.2f} | {near3:<12.2f}")
    print("="*70 + "\n")

if __name__ == "__main__":
    compare_dip_thresholds()
