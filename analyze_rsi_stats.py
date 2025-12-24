
import pandas as pd
import numpy as np
from indicators import TechnicalAnalyzer

def analyze_rsi_behavior():
    # 1. Load Data
    try:
        df = pd.read_csv("BIST100_PREDICTION_READY.csv")
    except:
        print("CSV not found.")
        return

    # Column cleaning
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Date parsing
    date_col = None
    for col in df.columns:
        if col in ["date", "datetime", "time", "tarih"]:
            date_col = col
            break
            
    df[date_col] = pd.to_datetime(df[date_col])
    if "close" in df.columns and "price" not in df.columns:
        df = df.rename(columns={"close": "price"})

    # 2. Calculate RSI
    analyzer = TechnicalAnalyzer(df)
    analyzer.add_rsi(period=14)
    df = analyzer.get_df()
    
    # 3. Analyze Overbought Blocks (RSI > 70)
    THRESHOLD = 70
    
    overbought_mask = df["RSI"] > THRESHOLD
    
    # Identify blocks
    # Convert bool series to int, calculate diff to find starts and ends
    df["OB_Block"] = (overbought_mask != overbought_mask.shift()).cumsum()
    
    # Filter only rows within overbought
    ob_data = df[overbought_mask].copy()
    
    if ob_data.empty:
        print("No RSI > 70 events found.")
        return

    stats = []
    
    for block_id, block_df in ob_data.groupby("OB_Block"):
        start_date = block_df[date_col].iloc[0]
        end_date = block_df[date_col].iloc[-1]
        
        entry_rsi = block_df["RSI"].iloc[0]
        max_rsi = block_df["RSI"].max()
        rsi_surge = max_rsi - entry_rsi
        
        duration_days = len(block_df) # Trading days
        
        # Find Price Peak within this block
        max_price_idx = block_df["price"].idxmax()
        max_price_date = block_df.loc[max_price_idx, date_col]
        peak_price = block_df.loc[max_price_idx, "price"]
        
        # Days from Entry to Price Peak
        days_to_peak = (block_df.index.get_loc(max_price_idx) - block_df.index.get_loc(block_df.index[0]))
        
        stats.append({
            "Start": start_date,
            "Duration": duration_days,
            "RSI_Entry": entry_rsi,
            "RSI_Max": max_rsi,
            "RSI_Surge": rsi_surge,
            "Days_to_Peak": days_to_peak,
            "Peak_Price": peak_price
        })
        
    stats_df = pd.DataFrame(stats)
    
    print("\n" + "="*50)
    print(f"📊 RSI > {THRESHOLD} BÖLGESİ DAVRANIŞ ANALİZİ (BIST100)")
    print("="*50)
    print(f"Toplam Olay Sayısı: {len(stats_df)}")
    
    print("\n--- ZAMANLAMA İSTATİSTİKLERİ ---")
    print(f"Ortalama Kalış Süresi: {stats_df['Duration'].mean():.2f} Gün")
    print(f"Maksimum Kalış Süresi: {stats_df['Duration'].max()} Gün")
    print(f"Zirveye Ulaşma Süresi: {stats_df['Days_to_Peak'].mean():.2f} Gün (RSI 70 olduktan sonra)")
    
    print("\n--- GÜÇ İSTATİSTİKLERİ ---")
    print(f"Ortalama RSI Giriş:    {stats_df['RSI_Entry'].mean():.2f}")
    print(f"Ortalama RSI Zirve:    {stats_df['RSI_Max'].mean():.2f}")
    print(f"Ortalama RSI Artışı:   +{stats_df['RSI_Surge'].mean():.2f} Puan")
    
    print("\n--- SON 5 ÖRNEK ---")
    print(stats_df[["Start", "Duration", "Days_to_Peak", "RSI_Max"]].tail(5).to_string(index=False))

if __name__ == "__main__":
    analyze_rsi_behavior()
