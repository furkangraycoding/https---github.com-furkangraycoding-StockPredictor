
import pandas as pd
import numpy as np
from indicators import TechnicalAnalyzer
from ml_engine import MLEngine

def debug_rsi_overbought():
    df = pd.read_csv("BIST100_PREDICTION_READY.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Standardize columns for Analyzer
    df = df.rename(columns={'Close': 'price', 'Open': 'Open', 'High': 'High', 'Low': 'Low'})
    
    analyzer = TechnicalAnalyzer(df)
    analyzer.add_rsi()
    analyzer.add_rsi_features() # Calculates RSI_Overbought_Days
    analyzer.add_zigzag_labels()
    analyzer.add_moving_averages()
    analyzer.add_atr()
    analyzer.determine_regime()
    df = analyzer.add_derived_features()
    
    engine = MLEngine(df)
    # Train it briefly or just use as is (weights are randomized if not trained, but here it loads weights in __init__ if they exist? No, it needs training)
    # Actually, the user's app HAS trained it. I will train it here 70/30 split like blind_test.
    metrics, df_pred = engine.train(optimize=False)
    
    # Filter for August 2025
    mask = (df_pred['Date'] >= '2025-08-15') & (df_pred['Date'] <= '2025-08-30')
    # Column names in df_pred after engine.train are Prob_Peak etc.
    cols = ['Date', 'price', 'RSI', 'RSI_Overbought_Days', 'Last_Signal', 'Prob_Peak']
    # Check which ones exist
    available_cols = [c for c in cols if c in df_pred.columns]
    target = df_pred[mask][available_cols]
    
    print("Debug Data for August 2025:")
    print(target.to_string())

if __name__ == "__main__":
    debug_rsi_overbought()
