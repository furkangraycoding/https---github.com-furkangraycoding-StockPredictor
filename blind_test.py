
import pandas as pd
from ml_engine import MLEngine

def run_blind_test():
    print("Initializing Blind Test Validation...")
    
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
    
    # Normalize columns
    df.columns = [c.lower().strip() for c in df.columns]
    if "close" in df.columns and "price" not in df.columns:
        df = df.rename(columns={"close": "price"})
        
    if df["price"].dtype == 'object':
        df["price"] = df["price"].astype(str).str.replace(',', '').astype(float)
        
    print(f"Data Loaded: {len(df)} records.")
    
    # Init Engine
    engine = MLEngine(df)
    
    print("Running Train/Test Split (70/30) & Evaluation with New Defaults...")
    metrics, test_results = engine.train(optimize=False) 
    
    print("\n" + "="*40)
    print("BLIND TEST RESULTS (Unseen Future Data)")
    if "Prob_Peak" in test_results.columns:
        print("\nProb_Peak Statistics:")
        print(test_results["Prob_Peak"].describe())
    print("="*40)
    
    # ---------------------------------------------------------
    # CUSTOM WINDOWED METRICS (2-Day Tolerance)
    # ---------------------------------------------------------
    def calculate_windowed_metrics(df, label_col, pred_col, tolerance=2):
        # 1. Catch Rate (Recall): Fraction of Real Events detected within [t, t+tolerance]
        real_event_indices = df[df[label_col] == 1].index
        detected_count = 0
        total_events = len(real_event_indices)
        
        for idx in real_event_indices:
            # Check window [idx, idx + tolerance]
            # We need integer location, not index label if index is not default
            # Assuming default range index for simplicity or using get_loc
            try:
                loc = df.index.get_loc(idx)
                window_preds = df[pred_col].iloc[loc : loc + tolerance + 1]
                if window_preds.sum() > 0:
                    detected_count += 1
            except:
                pass
                
        w_recall = detected_count / total_events if total_events > 0 else 0.0
        
        # 2. Trust Rate (Precision): Fraction of Predictions that are close to a Real Event [t-tolerance, t]
        pred_event_indices = df[df[pred_col] == 1].index
        valid_pred_count = 0
        total_preds = len(pred_event_indices)
        
        for idx in pred_event_indices:
            # Check window [idx - tolerance, idx] (Did an event happen recently?)
            try:
                loc = df.index.get_loc(idx)
                start_loc = max(0, loc - tolerance)
                window_labels = df[label_col].iloc[start_loc : loc + 1]
                if window_labels.sum() > 0:
                    valid_pred_count += 1
            except:
                pass
                
        w_precision = valid_pred_count / total_preds if total_preds > 0 else 0.0
        
        return w_recall, w_precision, total_events, detected_count
    
    def calculate_persistence_score(df, label_col, pred_col, lookback=5, min_signals=3):
        """
        Checks if the model was 'persistent' before an event.
        For each Real Event at T, check T-5 to T-1.
        Did we get at least 'min_signals' warnings?
        """
        real_event_indices = df[df[label_col] == 1].index
        persistent_count = 0
        total_events = len(real_event_indices)
        
        for idx in real_event_indices:
            try:
                loc = df.index.get_loc(idx)
                # Look back window: [loc - lookback, loc - 1]
                start_loc = max(0, loc - lookback)
                end_loc = loc # exclusive of event day itself
                
                if start_loc < end_loc:
                    window_preds = df[pred_col].iloc[start_loc : end_loc]
                    if window_preds.sum() >= min_signals:
                        persistent_count += 1
            except:
                pass
                
        p_recall = persistent_count / total_events if total_events > 0 else 0.0
        return p_recall, persistent_count

    # Calculate Standard Window metrics
    dip_rec, dip_prec, dip_tot, dip_det = calculate_windowed_metrics(test_results, "Label_Dip", "Pred_Dip", tolerance=5)
    peak_rec, peak_prec, peak_tot, peak_det = calculate_windowed_metrics(test_results, "Label_Peak", "Pred_Peak", tolerance=5)
    
    # Calculate Persistence Sensitivity
    print("\n>>> PERSISTENCE SENSITIVITY ANALYSIS (5 Day Window) <<<")
    print(f"{'Min Signals':<12} | {'Success Rate':<15} | {'Count (detected/total)'}")
    print("-" * 50)
    
    total_peaks = len(test_results[test_results["Label_Peak"] == 1])
    
    for min_sig in range(1, 6):
        rate, count = calculate_persistence_score(test_results, "Label_Peak", "Pred_Peak", lookback=5, min_signals=min_sig)
        print(f"{min_sig:<12} | {rate:.2%}         | {count}/{total_peaks}")

    def calculate_advanced_persistence(df, label_col, prob_col, lookback=7, rule_type="3red_consecutive"):
        real_event_indices = df[df[label_col] == 1].index
        persistent_count = 0
        total_events = len(real_event_indices)
        
        for idx in real_event_indices:
            try:
                loc = df.index.get_loc(idx)
                start_loc = max(0, loc - lookback)
                end_loc = loc
                
                if start_loc < end_loc:
                    window_probs = df[prob_col].iloc[start_loc : end_loc]
                    
                    if rule_type == "3red_consecutive":
                        # Check for ANY 3 consecutive reds in this window
                        is_red = (window_probs > 0.60).astype(int)
                        # Sliding window sum of binary 'is_red'
                        consecutive_reds = is_red.rolling(window=3).sum()
                        if (consecutive_reds >= 3).any():
                            persistent_count += 1
            except:
                pass
        
        rate = persistent_count / total_events if total_events > 0 else 0.0
        return rate, persistent_count

    print("\n>>> SENSITIVITY TEST: Yellow Signal Limits (with 2+ Red > 0.70) <<<")
    for y_limit in [1, 2, 3]:
        def calculate_v5_persistence(df, label_col, prob_col, lookback=5, max_y=1):
            real_indices = df[df[label_col] == 1].index
            count = 0
            for idx in real_indices:
                try:
                    loc = df.index.get_loc(idx)
                    win = df[prob_col].iloc[max(0, loc-5) : loc]
                    reds = (win > 0.70).sum()
                    yellows = ((win > 0.50) & (win <= 0.70)).sum()
                    if reds >= 2 and yellows <= max_y:
                        count += 1
                except: pass
            return count / len(real_indices) if len(real_indices) > 0 else 0

    print("\n>>> PROPOSED THRESHOLD TEST: Red > 0.75, Yellow 0.60-0.75 <<<")
    def calculate_v6_persistence(df, label_col, prob_col, lookback=5, max_y=2):
        real_indices = df[df[label_col] == 1].index
        count = 0
        for idx in real_indices:
            try:
                loc = df.index.get_loc(idx)
                win = df[prob_col].iloc[max(0, loc-5) : loc]
                reds = (win > 0.75).sum()
                yellows = ((win > 0.60) & (win <= 0.75)).sum()
                if reds >= 2 and yellows <= max_y:
                    count += 1
            except: pass
        return count / len(real_indices) if len(real_indices) > 0 else 0

    rate_v6 = calculate_v6_persistence(test_results, "Label_Peak", "Prob_Peak")
    print("\n>>> PROPOSED THRESHOLD TEST: Red > 0.80, Yellow 0.65-0.80 <<<")
    def calculate_v7_persistence(df, label_col, prob_col, lookback=5, max_y=2):
        real_indices = df[df[label_col] == 1].index
        count = 0
        for idx in real_indices:
            try:
                loc = df.index.get_loc(idx)
                win = df[prob_col].iloc[max(0, loc-5) : loc]
                reds = (win > 0.80).sum()
                yellows = ((win > 0.65) & (win <= 0.80)).sum()
                if reds >= 2 and yellows <= max_y:
                    count += 1
            except: pass
        return count / len(real_indices) if len(real_indices) > 0 else 0

    rate_v7 = calculate_v7_persistence(test_results, "Label_Peak", "Prob_Peak")
    print("\n>>> PEAK DETECTION ACCURACY (Window: Peak Day or 2 Days Prior) <<<")
    def calculate_strict_recall(df, label_col, prob_col, threshold=0.80):
        real_peak_indices = df[df[label_col] == 1].index
        hits = 0
        total = len(real_peak_indices)
        
        for idx in real_peak_indices:
            try:
                # Get integer location
                loc = df.index.get_loc(idx)
                # Check window [loc-2, loc] (Current day + 2 days before)
                start_loc = max(0, loc - 2)
                end_loc = loc + 1 # inclusive of current day
                
                window_probs = df[prob_col].iloc[start_loc : end_loc]
                if (window_probs > threshold).any():
                    hits += 1
            except: pass
            
        rate = hits / total if total > 0 else 0
        return rate, hits, total

    # Test with 0.80 threshold
    rate_80, hits_80, total_80 = calculate_strict_recall(test_results, "Label_Peak", "Prob_Peak", threshold=0.80)
    print(f"Red (>0.80) Signal exactly on Peak or 2 days before: {rate_80:.2%} ({hits_80}/{total_80})")
    
    # Test with 0.70 threshold (for comparison)
    rate_70, hits_70, total_70 = calculate_strict_recall(test_results, "Label_Peak", "Prob_Peak", threshold=0.70)
    print(f"Red (>0.70) Signal exactly on Peak or 2 days before: {rate_70:.2%} ({hits_70}/{total_70})")
    
    # FORWARD-LOOKING TEST (using binary predictions)
    print("\n>>> FORWARD-LOOKING PRECISION (4 Day Window) <<<")
    print("Question: 'If we see X signals in LAST 4 days, will peak occur in NEXT 4 days?'")
    print(f"{'Rule':<40} | {'Precision':<12} | {'Warnings (Correct/Total)'}")
    print("-"*80)
    
    def calculate_forward_precision_binary(df, label_col, pred_col, 
                                    lookback=4, lookahead=4,
                                    min_signals=2):
        warnings_issued = 0
        correct_warnings = 0
        
        for i in range(lookback, len(df) - lookahead):
            past_window = df[pred_col].iloc[i-lookback : i]
            signal_count = past_window.sum()
            
            should_warn = (signal_count >= min_signals)
            
            if should_warn:
                warnings_issued += 1
                future_window = df[label_col].iloc[i : i+lookahead]
                if future_window.sum() > 0:
                    correct_warnings += 1
        
        precision = correct_warnings / warnings_issued if warnings_issued > 0 else 0.0
        return precision, correct_warnings, warnings_issued
    
    forward_tests = [
        ("1+ signal in last 4 days", 1),
        ("2+ signals in last 4 days", 2),
        ("3+ signals in last 4 days", 3),
        ("4 signals in last 4 days (all)", 4),
    ]
    
    for name, min_sig in forward_tests:
        prec, correct, total = calculate_forward_precision_binary(
            test_results,
            "Label_Peak",
            "Pred_Peak",
            lookback=4,
            lookahead=4,
            min_signals=min_sig
        )
        print(f"{name:<40} | {prec:.2%}        | {correct}/{total}")
    
    # FORWARD-LOOKING TEST WITH OFFSET (Days 2-5)
    print("\n>>> FORWARD-LOOKING PRECISION (Days 2-5 Ahead) <<<")
    print("Question: 'If we see X signals in LAST 4 days, will peak occur in days 2-5 (not immediate)?'")
    print(f"{'Rule':<40} | {'Precision':<12} | {'Warnings (Correct/Total)'}")
    print("-"*80)
    
    def calculate_forward_precision_offset(df, label_col, pred_col, 
                                    lookback=4, offset_start=2, offset_end=5,
                                    min_signals=2):
        warnings_issued = 0
        correct_warnings = 0
        
        for i in range(lookback, len(df) - offset_end):
            past_window = df[pred_col].iloc[i-lookback : i]
            signal_count = past_window.sum()
            
            should_warn = (signal_count >= min_signals)
            
            if should_warn:
                warnings_issued += 1
                # Check days 2-5 (skip day 0 and 1)
                future_window = df[label_col].iloc[i+offset_start : i+offset_end+1]
                if future_window.sum() > 0:
                    correct_warnings += 1
        
        precision = correct_warnings / warnings_issued if warnings_issued > 0 else 0.0
        return precision, correct_warnings, warnings_issued
    
    offset_tests = [
        ("1+ signal in last 4 days", 1),
        ("2+ signals in last 4 days", 2),
        ("3+ signals in last 4 days", 3),
        ("4 signals in last 4 days (all)", 4),
    ]
    
    for name, min_sig in offset_tests:
        prec, correct, total = calculate_forward_precision_offset(
            test_results,
            "Label_Peak",
            "Pred_Peak",
            lookback=4,
            offset_start=2,
            offset_end=5,
            min_signals=min_sig
        )
        print(f"{name:<40} | {prec:.2%}        | {correct}/{total}")
    
    # Original Strict Metrics for comparison
    print("\n--- Strict (Same Day) Metrics ---")
    print(f"Dip Strict Precision: {metrics['dip'].get('precision', 0):.4f}")
    print(f"Peak Strict Precision: {metrics['peak'].get('precision', 0):.4f}")
    
    # Detailed check
    if not test_results.empty and "Label_Dip" in test_results.columns:
        print("\nPrediction Sample (Last 10 rows of test set):")
        cols = ["price", "Label_Dip", "Pred_Dip", "Label_Peak", "Pred_Peak"]
        # Add date if exists
        date_c = next((c for c in test_results.columns if "date" in c.lower()), None)
        if date_c: cols.insert(0, date_c)
        
        print(test_results[cols].tail(10).to_string())

if __name__ == "__main__":
    run_blind_test()
