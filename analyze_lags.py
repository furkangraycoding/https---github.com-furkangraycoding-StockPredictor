
import pandas as pd
import numpy as np
from indicators import TechnicalAnalyzer

def load_and_analyze():
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
    else:
        print("No date column found.")
        return

    # Normalize columns
    df.columns = [c.lower().strip() for c in df.columns]
    date_col = date_col.lower().strip()
    
    if "close" in df.columns and "price" not in df.columns:
        df = df.rename(columns={"close": "price"})
        
    if df["price"].dtype == 'object':
        df["price"] = df["price"].astype(str).str.replace(',', '').astype(float)

    # Use actual TechnicalAnalyzer to validte the NEW logic
    ta = TechnicalAnalyzer(df)
    ta.add_atr(period=14) # Ensure ATR exists for dynamic threshold
    ta.add_zigzag_labels(threshold_pct=0.03, atr_factor=2.5)
    
    df_analyzed = ta.get_df()
    
    # Extract Events
    # We look for where "Dip" or "Tepe" is NOT NaN.
    # The 'lag' is defined as: Date of Confirmation - Date of Logic Pivot
    # But wait, ZigZag updates the Pivot date row directly.
    # To find confirmation date, we need to simulate the loop or infer it from "Last_Signal" change?
    # Actually, the 'Last_Signal' flips exactly on the confirmation day (index i in the loop).
    # Step:
    # 1. Find rows where Last_Signal changes.
    # 2. If Last_Signal changes from 1 to -1 -> Peak Confirmation. 
    #    The "Tepe" value should be at some previous index.
    #    The lag is (Confirmation Date - Peak Date).
    
    # Let's find confirmation indices
    df_analyzed["Signal_Change"] = df_analyzed["Last_Signal"].diff()
    
    # 1 -> -1 (Change = -2): Peak Confirmed
    # -1 -> 1 (Change = 2): Dip Confirmed
    
    events = []
    
    # Iterate through confirmation points
    confirmations = df_analyzed[df_analyzed["Signal_Change"].abs() == 2].index
    
    # Track the last known pivot dates
    # Since Tepe/Dip columns are sparse, we can find the non-nulls.
    
    # Efficient way:
    # Create list of (Index, Type, Date) for all Pivots
    pivots_idx = df_analyzed[df_analyzed["Dip"].notna() | df_analyzed["Tepe"].notna()].index
    
    for conf_idx in confirmations:
        signal = df_analyzed.loc[conf_idx, "Last_Signal"]
        change = df_analyzed.loc[conf_idx, "Signal_Change"]
        conf_date = df_analyzed.loc[conf_idx, date_col]
        
        # If change is -2 (1 -> -1), we just confirmed a Peak (Type -1)
        # We need to find the most recent "Tepe" row BEFORE this conf_idx
        if change == -2: # Peak Confirmed
             # Find max index in pivots_idx < conf_idx where Tepe is not NaN
             # Actually, just search backwards or filter
             recent_peaks = df_analyzed.loc[:conf_idx-1]["Tepe"].dropna()
             if not recent_peaks.empty:
                 peak_idx = recent_peaks.index[-1]
                 peak_date = df_analyzed.loc[peak_idx, date_col]
                 lag = (conf_date - peak_date).days
                 events.append({
                     'type': 'Peak',
                     'event_date': peak_date,
                     'detect_date': conf_date,
                     'lag_days': lag
                 })
                 
        elif change == 2: # Dip Confirmed
             # Find max index in pivots_idx < conf_idx where Dip is not NaN
             recent_dips = df_analyzed.loc[:conf_idx-1]["Dip"].dropna()
             if not recent_dips.empty:
                 dip_idx = recent_dips.index[-1]
                 dip_date = df_analyzed.loc[dip_idx, date_col]
                 lag = (conf_date - dip_date).days
                 events.append({
                     'type': 'Dip',
                     'event_date': dip_date,
                     'detect_date': conf_date,
                     'lag_days': lag
                 })

    results = pd.DataFrame(events)

    if results.empty:
        print("No pivots found with new settings.")
        return

    total_pivots = len(results)
    avg_lag = results["lag_days"].mean()
    within_4_days = results[results["lag_days"] <= 4]
    
    print("-" * 30)
    print("NEW ANALYSIS (Threshold: 3% + ATR)")
    print("-" * 30)
    print(f"Total Pivots Found: {total_pivots}")
    print(f"Average Detection Lag: {avg_lag:.2f} days")
    print(f"Detected within 4 days: {len(within_4_days)} ({(len(within_4_days)/total_pivots)*100:.2f}%)")
    print("-" * 30)
    
    print("\n--- FAST DETECTIONS (Within 4 Days) ---")
    if not within_4_days.empty:
        print(within_4_days[["type", "event_date", "detect_date", "lag_days"]].head(10).to_string())
    
    print("\nBreakdown by Type:")
    print(results.groupby("type")["lag_days"].agg(['count', 'mean', 'min', 'max']))

if __name__ == "__main__":
    load_and_analyze()
