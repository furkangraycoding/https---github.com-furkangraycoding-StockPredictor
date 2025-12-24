import pandas as pd
import numpy as np
from ml_engine import MLEngine
from data_loader import load_data

def verify_signal_purity_on_peaks():
    # Load data
    df, date_col = load_data('BIST100_PREDICTION_READY.csv')
    
    # Initialize and train engine to get probabilities
    engine = MLEngine(df)
    engine.train()
    
    # Use the dataframe that already has labels added during training
    df_with_preds = engine.add_predictions_to_df(engine.df.copy())
    
    # Identify historical peaks confirmed by ZigZag
    peak_col = next((c for c in df_with_preds.columns if c.lower() == 'tepe'), None)
    if not peak_col:
        print(f"Error: Peak column not found. Available: {df_with_preds.columns.tolist()}")
        return

    peaks = df_with_preds.dropna(subset=[peak_col]).copy()
    
    print(f"Total Confirmed Peaks Found: {len(peaks)}")
    print("-" * 50)
    print(f"{'Date':<12} | {'RSI':<6} | {'Peak%':<6} | {'Dip%':<6} | {'Gap':<6} | {'Threshold':<9} | {'Result'}")
    print("-" * 50)
    
    hits = 0
    for idx, row in peaks.iterrows():
        rsi = row.get('rsi', 50)
        peak_prob = row.get('AI_Peak_Prob', 0)
        dip_prob = row.get('AI_Dip_Prob', 0)
        
        gap = (peak_prob - dip_prob) * 100
        threshold = rsi * 0.45
        
        is_pure = gap > threshold
        result = "✅ SAF" if is_pure else "❌ BELİRSİZ"
        
        if is_pure: hits += 1
        
        date_str = row[date_col].strftime('%d.%m.%Y')
        print(f"{date_str:<12} | {rsi:<6.1f} | {peak_prob*100:<6.1f} | {dip_prob*100:<6.1f} | {gap:<6.1f} | {threshold:<9.1f} | {result}")

    print("-" * 50)
    print(f"Success Rate (Pure Signals on Real Peaks): {hits/len(peaks)*100:.2f}%")

if __name__ == "__main__":
    verify_signal_purity_on_peaks()
