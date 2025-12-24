
import pandas as pd
from ml_engine import MLEngine
from indicators import TechnicalAnalyzer

def check_feature_importance():
    # Load Data
    file_path = "BIST100_PREDICTION_READY.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return

    # Basic cleaning
    date_col = None
    for col in df.columns:
        if col.lower() in ["date", "datetime", "time"]:
            date_col = col
            break
    
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        
    df.columns = [c.lower().strip() for c in df.columns]
    if "close" in df.columns and "price" not in df.columns:
        df = df.rename(columns={"close": "price"})
        
    if df["price"].dtype == 'object':
        df["price"] = df["price"].astype(str).str.replace(',', '').astype(float)

    # Initialize Engine
    print("Initializing MLEngine and Training Models (Blind Split + No Cycle Phase)...")
    engine = MLEngine(df)
    
    # Train (optimize=False for speed)
    engine.train(optimize=False)
    
    # Get Importance
    importances = engine.get_feature_importance()
    
    print("\n" + "="*40)
    print("FINAL FEATURE IMPORTANCE RANKINGS")
    print("="*40)
    
    print("\n--- DIP MODEL (Buying Signals - Hybrid) ---")
    dip_imp = importances.sort_values(by="Dip_Importance", ascending=False)["Dip_Importance"]
    print(dip_imp[dip_imp > 0].head(5))
    
    print("\n--- PEAK MODEL (Selling Signals - PURE TECHNICAL) ---")
    peak_imp = importances.sort_values(by="Peak_Importance", ascending=False)["Peak_Importance"]
    print(peak_imp[peak_imp > 0].head(10))

if __name__ == "__main__":
    check_feature_importance()
