
import pandas as pd
from ml_engine import MLEngine
from indicators import TechnicalAnalyzer

def check_prediction_history():
    # Load Data
    df = pd.read_csv("BIST100_PREDICTION_READY.csv")
    
    # Column cleaning
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Clean/Parse Date
    date_col = None
    for col in df.columns:
        if col in ["date", "datetime", "time", "tarih"]:
            date_col = col
            break
            
    if "close" in df.columns and "price" not in df.columns:
        df = df.rename(columns={"close": "price"})
        
    df[date_col] = pd.to_datetime(df[date_col])
    
    # 1. Train Model on data BEFORE Sept 2025 (to simulate real-time)
    # Actually, the Blind Test simulates this automatically via cross-validation logic
    # But here let's train on FULL data and see 'out-of-bag' or simply train on data up to Sept 15.
    
    target_peak_date = pd.to_datetime("2025-09-22")
    start_check_date = pd.to_datetime("2025-09-15")
    
    # Filter data UP TO the peak event to train properly (No cheating)
    # We train up to Sept 15, then predict 16, 17... 22.
    
    train_cutoff = start_check_date
    train_df = df[df[date_col] <= train_cutoff].copy()
    
    print(f"Training Model on data up to: {train_cutoff.strftime('%Y-%m-%d')}...")
    engine = MLEngine(train_df)
    engine.train(optimize=False) # Fast train
    
    # Now predict for the window [Sept 15 - Sept 25]
    test_window_df = df[(df[date_col] >= start_check_date) & (df[date_col] <= pd.to_datetime("2025-09-26"))].copy()
    
    # We must calculate indicators for these dates as if we are stepping through time
    # But simply running predict_probs on the rows is enough as the Engine has the trained model
    
    print("\n-----------------------------------------------------------")
    print(f"PREDICTIONS AROUND PEAK DATE ({target_peak_date.strftime('%Y-%m-%d')})")
    print("-----------------------------------------------------------")
    print(f"{'Date':<12} | {'Price':<10} | {'Dip Prob':<10} | {'Peak Prob':<10} | {'Warning?':<15}")
    print("-" * 70)
    
    for idx, row in test_window_df.iterrows():
        # Re-calculate correct features for this row context?
        # Actually our DF already has full history features. 
        # In a real backtest we'd recalculate, but here taking features is a good approx of 'what did indicators say'.
        
        dip, peak = engine.predict_probs(row)
        
        date_str = row[date_col].strftime('%Y-%m-%d')
        price_str = f"{row['price']:.2f}"
        
        # Check tolerance warning
        warning = ""
        if peak > 0.60:
            warning = "🔴 PEAK WARN"
        elif peak > 0.50:
            warning = "⚠️ Possible"
            
        if date_str == "2025-09-22":
            date_str = f"-> {date_str}" # Mark the event day
            
        print(f"{date_str:<12} | {price_str:<10} | {dip:.1%}      | {peak:.1%}      | {warning}")

if __name__ == "__main__":
    try:
        check_prediction_history()
    except Exception as e:
        print(f"Error: {e}")
