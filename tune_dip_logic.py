import pandas as pd
import numpy as np
from ml_engine import MLEngine
from data_loader import load_data
from indicators import TechnicalAnalyzer

def tune_flexible_dip():
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

    # Define Search Space
    # Condition A (High Prob) Settings
    probs_a = [0.80, 0.85]
    mults_a = [1.1, 1.2, 1.3]
    
    # Condition B (Low Prob) Settings
    probs_b = [0.65, 0.70, 0.75]
    mults_b = [1.3, 1.4, 1.5, 1.6]

    results = []

    # Target Date to Check (17 Oct 2024 is the benchmark)
    oct17_row = df_all[df_all[date_col].astype(str).str.contains('2024-10-17')]

    for p_a in probs_a:
        for m_a in mults_a:
            for p_b in probs_b:
                for m_b in mults_b:
                    
                    # Apply this combination
                    mask_a = (df_all['AI_Dip_Prob'] >= p_a) & (df_all['gap'] > df_all['RSI'] * m_a)
                    mask_b = (df_all['AI_Dip_Prob'] >= p_b) & (df_all['gap'] > df_all['RSI'] * m_b)
                    df_all['temp_pure'] = mask_a | mask_b
                    
                    subset = df_all[df_all['temp_pure'] == True]
                    total = len(subset)
                    if total == 0: continue
                    
                    # Metrics
                    tp = df_all[(df_all['temp_pure'] == True) & (df_all['Label_Dip'] == 1)].shape[0]
                    precision = (tp / total) * 100
                    
                    near3 = 0
                    for idx in subset.index:
                        loc = df_all.index.get_loc(idx)
                        window = df_all.iloc[max(0, loc-3) : min(len(df_all), loc+4)]
                        if window['is_exact_dip'].any():
                            near3 += 1
                    
                    hit_rate_3g = (near3 / total) * 100
                    score = (precision + hit_rate_3g) / 2 # Balanced Score
                    
                    # 17 Oct coverage check
                    hits_oct17 = False
                    if not oct17_row.empty:
                        idx_17 = oct17_row.index[0]
                        hits_oct17 = df_all.loc[idx_17, 'temp_pure']

                    results.append({
                        'p_a': p_a, 'm_a': m_a, 'p_b': p_b, 'm_b': m_b,
                        'total': total, 'precision': precision, 'hit_3g': hit_rate_3g, 
                        'score': score, 'oct17': hits_oct17
                    })

    # Sort and analyze
    res_df = pd.DataFrame(results)
    # Filter only those that catch 17 Oct if possible
    filtered_df = res_df[res_df['oct17'] == True] if res_df['oct17'].any() else res_df
    
    # Sort by 3G Hit Rate and then Precision
    best_timing = filtered_df.sort_values(by='hit_3g', ascending=False).head(5)
    best_precision = filtered_df.sort_values(by='precision', ascending=False).head(5)
    best_balanced = filtered_df.sort_values(by='score', ascending=False).head(5)

    print("\n=== TUNING RESULTS: TOP 5 BALANCED (Catching 17 Oct) ===")
    print(best_balanced[['p_a', 'm_a', 'p_b', 'm_b', 'total', 'hit_3g', 'precision', 'score']])
    
    print("\n=== TUNING RESULTS: TOP 5 TIMING FOCUS (+/- 3G) ===")
    print(best_timing[['p_a', 'm_a', 'p_b', 'm_b', 'total', 'hit_3g', 'precision', 'score']])

if __name__ == "__main__":
    tune_flexible_dip()
