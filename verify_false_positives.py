import pandas as pd
import numpy as np
from ml_engine import MLEngine
from data_loader import load_data

def verify_false_positives():
    # Load data
    df, date_col = load_data('BIST100_PREDICTION_READY.csv')
    
    # Initialize and train engine
    engine = MLEngine(df)
    engine.train()
    
    # Use the internal dataframe from the engine (has labels)
    # add_predictions_to_df adds AI_Peak_Prob and AI_Dip_Prob
    df_with_preds = engine.add_predictions_to_df(engine.df.copy())
    
    # Find all days where a "SAF ZİRVE SİNYALİ" was generated
    # Rule: Peak_Prob > 0.85 AND (Peak% - Dip%) > (RSI * 0.45)
    df_with_preds['gap'] = (df_with_preds['AI_Peak_Prob'] - df_with_preds['AI_Dip_Prob']) * 100
    df_with_preds['threshold'] = df_with_preds['rsi_14'] * 0.45
    
    pure_signals = df_with_preds[
        (df_with_preds['AI_Peak_Prob'] > 0.85) & 
        (df_with_preds['gap'] > df_with_preds['threshold'])
    ].copy()
    
    print(f"Total SAF ZİRVE Signals Generated: {len(pure_signals)}")
    
    # A False Positive is a Pure Signal where NO Label_Peak exists within -2 to +4 days
    # Labels are already smeared in Label_Peak (window=7)
    false_positives = pure_signals[pure_signals['Label_Peak'] == 0]
    
    print(f"False Positives (Signal on Non-Peak Day): {len(false_positives)}")
    print("-" * 50)
    if len(false_positives) > 0:
        print(f"{'Date':<12} | {'Price':<8} | {'Peak%':<6} | {'Dip%':<6} | {'Reason'}")
        print("-" * 50)
        for idx, row in false_positives.head(20).iterrows():
            date_str = row[date_col].strftime('%d.%m.%Y')
            print(f"{date_str:<12} | {row['price']:<8.2f} | {row['AI_Peak_Prob']*100:<6.1f} | {row['AI_Dip_Prob']*100:<6.1f} | Hatalı Sinyal")
    else:
        print("Tebrikler! Hiç hatalı 'Saf Zirve' sinyali üretilmedi.")
    
    print("-" * 50)
    precision = (len(pure_signals) - len(false_positives)) / len(pure_signals) if len(pure_signals) > 0 else 0
    print(f"Saf Sinyal Hassasiyeti (Precision): {precision*100:.2f}%")

if __name__ == "__main__":
    verify_false_positives()
