import pandas as pd
import numpy as np
from ml_engine import MLEngine
from data_loader import load_data
from indicators import TechnicalAnalyzer

def test_flexible_logic():
    # 1. Load data
    df, date_col = load_data('BIST100_PREDICTION_READY.csv')
    analyzer = TechnicalAnalyzer(df)
    analyzer.add_rsi()
    analyzer.add_zigzag_labels()
    df = analyzer.add_derived_features()
    
    engine = MLEngine(df)
    engine.train()
    df_all = engine.add_predictions_to_df(engine.df.copy())
    
    df_all['gap'] = (df_all['AI_Dip_Prob'] - df_all['AI_Peak_Prob']) * 100
    df_all['is_exact_dip'] = df_all['Dip'].notna()

    # Define the flexible logic
    def is_flexible_pure(row):
        prob = row['AI_Dip_Prob']
        gap = row['gap']
        rsi = row['RSI']
        
        # Condition A: 80% Prob + 1.2x RSI Gap
        cond_a = (prob >= 0.80) and (gap > rsi * 1.2)
        # Condition B: 70% Prob + 1.35x RSI Gap
        cond_b = (prob >= 0.70) and (gap > rsi * 1.35)
        
        return cond_a or cond_b

    df_all['is_pure_flexible'] = df_all.apply(is_flexible_pure, axis=1)
    
    # Define current logic (for comparison) - Assuming 0.70 + 1.25 as set last
    df_all['current_pure'] = (df_all['AI_Dip_Prob'] >= 0.70) & (df_all['gap'] > df_all['RSI'] * 1.25)

    def calculate_metrics(col_name):
        subset = df_all[df_all[col_name] == True]
        total = len(subset)
        if total == 0: return 0, 0, 0, 0
        
        exact = df_all[(df_all[col_name] == True) & (df_all['is_exact_dip'] == True)].shape[0]
        
        near1 = 0
        for idx in subset.index:
            loc = df_all.index.get_loc(idx)
            window = df_all.iloc[max(0, loc-1) : min(len(df_all), loc+2)]
            if window['is_exact_dip'].any():
                near1 += 1
                
        near3 = 0
        for idx in subset.index:
            loc = df_all.index.get_loc(idx)
            window = df_all.iloc[max(0, loc-3) : min(len(df_all), loc+4)]
            if window['is_exact_dip'].any():
                near3 += 1
        
        return total, (exact/total*100), (near1/total*100), (near3/total*100)

    print("\n" + "="*75)
    print(f"{'Mantık':<15} | {'Sinyal #':<8} | {'Tam-Gün %':<12} | {'+/- 1G %':<12} | {'+/- 3G %':<12}")
    print("-" * 75)
    
    # Current
    c, e, n1, n3 = calculate_metrics('current_pure')
    print(f"{'Mevcut (0.7/1.25)':<15} | {c:<8} | {e:<12.2f} | {n1:<12.2f} | {n3:<12.2f}")
    
    # Flexible
    c, e, n1, n3 = calculate_metrics('is_pure_flexible')
    print(f"{'Esnek (Yeni)':<15} | {c:<8} | {e:<12.2f} | {n1:<12.2f} | {n3:<12.2f}")
    print("="*75 + "\n")

    # Add 17 Oct specifically to see if it works
    oct17 = df_all[df_all[date_col].astype(str).str.contains('2024-10-17')]
    if not oct17.empty:
        res = is_flexible_pure(oct17.iloc[0])
        print(f"DEBUG: 2024-10-17 tarihindeki Esnek Sinyal Durumu: {res}")
        print(f"DEBUG: 17 Oct Prob: {oct17.iloc[0]['AI_Dip_Prob']:.2f}, Gap: {oct17.iloc[0]['gap']:.1f}, Threshold 1.35xRSI: {oct17.iloc[0]['RSI']*1.35:.1f}")

if __name__ == "__main__":
    test_flexible_logic()
