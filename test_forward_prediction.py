"""
Forward-Looking Peak Prediction Test
Test: "If we see X red + Y yellow signals in last 4 days, 
       will there be a peak in the NEXT 4 days?"
"""
import pandas as pd
from ml_engine import MLEngine
from indicators import TechnicalAnalyzer

def calculate_forward_precision(df, label_col, pred_prob_col, 
                                lookback=4, lookahead=4,
                                min_red=2, max_yellow=1):
    """
    For each day where we have qualifying signals in the past 'lookback' days,
    check if there's an actual peak in the next 'lookahead' days.
    
    Returns precision (how often our warnings are correct).
    """
    warnings_issued = 0
    correct_warnings = 0
    
    for i in range(lookback, len(df) - lookahead):
        # Look back window
        past_window = df[pred_prob_col].iloc[i-lookback : i]
        
        red_count = (past_window > 0.60).sum()
        yellow_count = ((past_window > 0.50) & (past_window <= 0.60)).sum()
        
        # Check if we should issue warning
        should_warn = (red_count >= min_red) and (yellow_count <= max_yellow)
        
        if should_warn:
            warnings_issued += 1
            
            # Look ahead window
            future_window = df[label_col].iloc[i : i+lookahead]
            
            # Did a peak actually occur?
            if future_window.sum() > 0:
                correct_warnings += 1
    
    precision = correct_warnings / warnings_issued if warnings_issued > 0 else 0.0
    return precision, correct_warnings, warnings_issued

def main():
    # Load and prepare data
    df = pd.read_csv("BIST100_PREDICTION_READY.csv")
    df.columns = [c.lower().strip() for c in df.columns]
    
    date_col = None
    for col in df.columns:
        if col in ["date", "datetime", "time", "tarih"]:
            date_col = col
            break
            
    if "close" in df.columns and "price" not in df.columns:
        df = df.rename(columns={"close": "price"})
        
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Prepare features
    analyzer = TechnicalAnalyzer(df)
    analyzer.add_zigzag_labels(threshold_pct=0.05)
    df = analyzer.add_derived_features()
    
    # Train model
    print("Training model...")
    engine = MLEngine(df)
    metrics, test_df = engine.train(optimize=False)
    
    if test_df.empty:
        print("No test data available")
        return
    
    # Ensure numeric
    if "AI_Peak_Prob" in test_df.columns:
        test_df["AI_Peak_Prob"] = pd.to_numeric(test_df["AI_Peak_Prob"], errors='coerce').fillna(0)
    
    print("\n" + "="*80)
    print("FORWARD-LOOKING PREDICTION TEST")
    print("="*80)
    print("\nQuestion: 'If we see signals in LAST 4 days, will peak occur in NEXT 4 days?'")
    print("\n" + "-"*80)
    print(f"{'Rule':<40} | {'Precision':<12} | {'Correct/Total Warnings'}")
    print("-"*80)
    
    # Test different combinations
    test_cases = [
        ("2+ Red + any Yellow", 2, 999),
        ("2+ Red + max 1 Yellow", 2, 1),
        ("2+ Red + max 0 Yellow (only red)", 2, 0),
        ("3+ Red + any Yellow", 3, 999),
        ("3+ Red + max 1 Yellow", 3, 1),
    ]
    
    for name, min_red, max_yellow in test_cases:
        prec, correct, total = calculate_forward_precision(
            test_df,
            "Label_Peak",
            "AI_Peak_Prob",
            lookback=4,
            lookahead=4,
            min_red=min_red,
            max_yellow=max_yellow
        )
        print(f"{name:<40} | {prec:.2%}        | {correct}/{total}")
    
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("- High Precision = When we warn, we're usually right")
    print("- Low Total Warnings = Very conservative (might miss peaks)")
    print("- High Total Warnings = Very sensitive (might have false alarms)")
    print("="*80)

if __name__ == "__main__":
    main()
