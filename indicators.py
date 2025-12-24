import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

class TechnicalAnalyzer:
    def __init__(self, df: pd.DataFrame, price_col: str = "price"):
        """
        Initialize with a dataframe. 
        df: Pandas DataFrame
        price_col: Name of the column containing price data
        """
        self.df = df.copy()
        self.price_col = price_col

    def add_zigzag_labels(self, threshold_pct: float = 0.03, atr_factor: float = 2.5):
        """
        Identifies turning points using the ZigZag algorithm.
        Also adds 'Last_Signal' state: 1 (Dip confirmed, looking for Peak), -1 (Peak confirmed, looking for Dip).
        threshold_pct: Minimum percentage movement to change trend (0.03 = 3%).
        atr_factor: Factor of ATR to use as a dynamic threshold (e.g., 2.5 * ATR).
        """
        if self.price_col not in self.df.columns:
            return self.df
            
        prices = self.df[self.price_col].values
        n = len(prices)
        if n < 2:
            return self.df
            
        # Ensure ATR exists for dynamic thresholding
        has_atr = "ATR" in self.df.columns
        atrs = self.df["ATR"].values if has_atr else np.zeros(n)
            
        # 1 = Up (Last was Dip), -1 = Down (Last was Peak)
        trend = 0 
        last_pivot_idx = 0
        last_pivot_price = prices[0]
        
        self.df["Tepe"] = np.nan
        self.df["Dip"] = np.nan
        self.df["Last_Signal"] = 0 # 0=Unknown, 1=Dip (Bullish leg), -1=Peak (Bearish leg)
        
        # Initial trend detection
        for i in range(1, n):
            if prices[i] > last_pivot_price * (1 + threshold_pct):
                trend = 1
                break
            elif prices[i] < last_pivot_price * (1 - threshold_pct):
                trend = -1
                break
                
        # ZigZag Loop
        for i in range(1, n):
            price = prices[i]
            current_atr = atrs[i] if has_atr and not np.isnan(atrs[i]) else 0
            
            # Dynamic Reversal Threshold:
            # We trigger reversal if price moves > threshold_pct OR > atr_factor * ATR
            # This allows catching high-volatility moves faster than the % threshold.
            
            if trend == 1: # Uptrend (Last confirmed was Dip)
                if price > last_pivot_price:
                    # New high in existing uptrend
                    last_pivot_price = price
                    last_pivot_idx = i
                else:
                    # Check Reversal Criteria
                    # Top -> Down
                    pct_move = (last_pivot_price - price) / last_pivot_price
                    atr_move = (last_pivot_price - price)
                    
                    is_reversal = (pct_move > threshold_pct)
                    if current_atr > 0:
                        is_reversal = is_reversal or (atr_move > atr_factor * current_atr)
                    
                    if is_reversal:
                        # Reversal to downtrend -> Confirm previous PEAK
                        self.df.iloc[last_pivot_idx, self.df.columns.get_loc("Tepe")] = last_pivot_price
                        # Set Last_Signal to -1 (Peak) starting from the day AFTER the peak
                        self.df.iloc[last_pivot_idx + 1:, self.df.columns.get_loc("Last_Signal")] = -1
                        
                        trend = -1
                        last_pivot_price = price
                        last_pivot_idx = i
                        
            else: # Downtrend (Last confirmed was Peak)
                if price < last_pivot_price:
                    # New low in existing downtrend
                    last_pivot_price = price
                    last_pivot_idx = i
                else:
                    # Check Reversal Criteria
                    # Bottom -> Up
                    pct_move = (price - last_pivot_price) / last_pivot_price
                    atr_move = (price - last_pivot_price)
                    
                    is_reversal = (pct_move > threshold_pct)
                    if current_atr > 0:
                        is_reversal = is_reversal or (atr_move > atr_factor * current_atr)
                        
                    if is_reversal:
                        # Reversal to uptrend -> Confirm previous DIP
                        self.df.iloc[last_pivot_idx, self.df.columns.get_loc("Dip")] = last_pivot_price
                        # Set Last_Signal to 1 (Dip) starting from the day AFTER the dip
                        self.df.iloc[last_pivot_idx + 1:, self.df.columns.get_loc("Last_Signal")] = 1
                        
                        trend = 1
                        last_pivot_price = price
                        last_pivot_idx = i

        # NEW: Handle the LAST leg (Unconfirmed Potential Pivot)
        # Use Dynamic Volatility Threshold (ATR) if available, otherwise Fixed %
        # We use 2.0 * ATR as a standard reversal signal.
        
        has_atk = "ATR" in self.df.columns and pd.notna(self.df["ATR"].iloc[-1])
        
        if trend == 1: # Uptrend -> Looking for Peak rejection
             diff = last_pivot_price - prices[-1]
             pct_diff = diff / last_pivot_price
             
             # Default Thresholds (AGGRESSIVE SCALPING MODE)
             atr_mult = 1.0 # 1x ATR is enough
             pct_limit = 0.015 # Max wait 1.5% - Very tight leash
             
             # Fast Track with RSI (SUPER AGGRESSIVE)
             if "RSI" in self.df.columns and self.df["RSI"].iloc[-1] < 60:
                 atr_mult = 0.8 # Less than 1 Std Dev!
                 pct_limit = 0.01 # 1% drop confirms peak if RSI agrees
                 
             if has_atk:
                 atr = self.df["ATR"].iloc[-1]
                 # Trigger if EITHER ATR threshold OR Max Percent threshold is met
                 is_reversal = (diff > (atr_mult * atr)) or (pct_diff > pct_limit)
             else:
                 # Fallback fixed
                 is_reversal = pct_diff > 0.01
                 
             if is_reversal:
                 self.df.iloc[last_pivot_idx, self.df.columns.get_loc("Tepe")] = last_pivot_price
                 self.df.iloc[last_pivot_idx + 1:, self.df.columns.get_loc("Last_Signal")] = -1
                 
        else: # Downtrend -> Looking for Dip bounce
             diff = prices[-1] - last_pivot_price
             pct_diff = diff / last_pivot_price
             
             # Default Thresholds (AGGRESSIVE)
             atr_mult = 1.0
             pct_limit = 0.015
             
             # Fast Track with RSI
             if "RSI" in self.df.columns and self.df["RSI"].iloc[-1] > 40:
                 atr_mult = 0.8
                 pct_limit = 0.01
                 
             if has_atk:
                 atr = self.df["ATR"].iloc[-1]
                 is_reversal = (diff > (atr_mult * atr)) or (pct_diff > pct_limit)
             else:
                 # Fallback fixed
                 is_reversal = pct_diff > 0.01
            
             if is_reversal:
                 self.df.iloc[last_pivot_idx, self.df.columns.get_loc("Dip")] = last_pivot_price
                 self.df.iloc[last_pivot_idx + 1:, self.df.columns.get_loc("Last_Signal")] = 1
             
        return self.df

    def add_rolling_volatility(self, window: int = 20):
        """Calculates annualized rolling volatility."""
        # Log returns
        self.df["Log_Ret"] = np.log(self.df[self.price_col] / self.df[self.price_col].shift(1))
        # Rolling Std Dev * sqrt(252) for annualization
        self.df["Volatility_20"] = self.df["Log_Ret"].rolling(window=window).std() * np.sqrt(252)
        return self.df

    def add_drawdown_features(self, window: int = 60):
        """Calculates distance from recent Highs and Lows."""
        roll_max = self.df[self.price_col].rolling(window=window, min_periods=1).max()
        roll_min = self.df[self.price_col].rolling(window=window, min_periods=1).min()
        
        self.df["Drawdown_Pct"] = (self.df[self.price_col] - roll_max) / roll_max
        self.df["Rally_Pct"] = (self.df[self.price_col] - roll_min) / roll_min
        return self.df

    def add_local_extrema(self, order: int = 5):
        """
        [DEPRECATED] Replaced by add_zigzag_labels. 
        Kept for backward compatibility if needed, but logic should move to ZigZag.
        """
        if self.price_col not in self.df.columns:
            return self.df
            
        prices = self.df[self.price_col].values
        
        self.df["Tepe"] = np.nan
        self.df["Dip"] = np.nan

        if len(prices) > order * 2:
            max_idx = argrelextrema(prices, np.greater, order=order)[0]
            min_idx = argrelextrema(prices, np.less, order=order)[0]

            # We use iloc to set values at specific integer indices
            # Note: This assigns the price value to that specific row in the new columns
            self.df.iloc[max_idx, self.df.columns.get_loc("Tepe")] = self.df.iloc[max_idx][self.price_col]
            self.df.iloc[min_idx, self.df.columns.get_loc("Dip")] = self.df.iloc[min_idx][self.price_col]
            
        return self.df

    def add_rsi(self, period: int = 14):
        """Calculates RSI."""
        delta = self.df[self.price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        self.df["RSI"] = 100 - (100 / (1 + rs))
        return self.df

    def add_moving_averages(self):
        """Calculates SMA 50 and 200."""
        self.df["SMA_50"] = self.df[self.price_col].rolling(window=50).mean()
        self.df["SMA_200"] = self.df[self.price_col].rolling(window=200).mean()
        return self.df

    def add_atr(self, period: int = 14):
        """Calculates Average True Range (ATR)."""
        high = self.df["high"] if "high" in self.df.columns else self.df[self.price_col]
        low = self.df["low"] if "low" in self.df.columns else self.df[self.price_col]
        close = self.df[self.price_col]

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.df["ATR"] = tr.rolling(window=period).mean()
        return self.df

    def determine_regime(self):
        """
        Determines market regime:
        - Bullish: Price > SMA200
        - Bearish: Price < SMA200
        - Recovery: Price > SMA50 but < SMA200
        - Warning: Price < SMA50 but > SMA200
        """
        # Ensure SMAs exist
        if "SMA_200" not in self.df.columns:
            self.add_moving_averages()

        def classify(row):
            price = row[self.price_col]
            sma50 = row["SMA_50"]
            sma200 = row["SMA_200"]
            
            if pd.isna(sma200): return "Unknown"
            
            if price > sma200:
                if price > sma50:
                    return "Bullish"
                else:
                    return "Weak Bull"
            else:
                if price > sma50:
                    return "Recovery"
                else:
                    return "Bearish"

        self.df["Regime"] = self.df.apply(classify, axis=1)
        return self.df
    
    def add_rsi_features(self):
        """Calculates RSI changes and Overbought/Oversold Duration."""
        if "RSI" not in self.df.columns: self.add_rsi()
        
        self.df["RSI_Diff_1D"] = self.df["RSI"].diff(1)
        self.df["RSI_Diff_3D"] = self.df["RSI"].diff(3)
        
        # Calculate consecutive days in Overbought zone (>70)
        # This helps the model wait for the Statistical Peak (~5 days)
        is_ob = (self.df["RSI"] > 70).astype(int)
        grouper = (is_ob != is_ob.shift()).cumsum()
        self.df["RSI_Overbought_Days"] = is_ob.groupby(grouper).cumsum()
        
        # Calculate consecutive days in Oversold zone (<30)
        is_os = (self.df["RSI"] < 30).astype(int)
        grouper_os = (is_os != is_os.shift()).cumsum()
        self.df["RSI_Oversold_Days"] = is_os.groupby(grouper_os).cumsum()
        
        return self.df

    def add_divergence_features(self, window: int = 14):
        """
        Calculates rolling correlation between Price and RSI.
        Negative correlation during uptrend = Bearish Divergence (Peak Warning).
        """
        if "RSI" not in self.df.columns: self.add_rsi()
        
        # Pearson correlation over rolling window
        # We handle NaN by filling with 0 or forward fill in downstream
        self.df[f"RSI_Price_Corr_{window}"] = self.df[self.price_col].rolling(window).corr(self.df["RSI"])
        return self.df


    def add_time_features(self):
        """Calculates days since the last pivot (Dip or Peak)."""
        # Create a combined column for any pivot date
        # If Dip or Tepe is not NaN, we have a pivot
        is_pivot = self.df["Dip"].notna() | self.df["Tepe"].notna()
        
        # We want to know the index of the last True value in is_pivot
        # We can use forward fill on a "Last_Pivot_Idx" columns?
        # Better: Create a 'Last_Pivot_Date' column by ffilling dates
        
        # Ensure we have a date column or index is date
        date_col = next((c for c in self.df.columns if "date" in c.lower() or "tarih" in c.lower()), None)
        if date_col:
             dates = self.df[date_col]
        else:
             dates = self.df.index.to_series()
             
        # Series where value is Date if pivot, else NaN
        pivot_dates = dates.where(is_pivot)
        last_pivot_date = pivot_dates.ffill()
        
        # Calculate diff in days
        try:
            time_diff = (dates - last_pivot_date).dt.days
            self.df["Days_Since_Last_Pivot"] = time_diff.fillna(0)
            
            # -----------------------------------------------------
            # NEW: LEG RETURN (Price change since last confirmed pivot)
            # -----------------------------------------------------
            # Combine Dip and Tepe into a single 'Last_Pivot_Price' column
            pivot_prices = self.df["Dip"].combine_first(self.df["Tepe"])
            last_pivot_price = pivot_prices.ffill()
            
            # Leg Return = (Current Price - Last Pivot Price) / Last Pivot Price
            self.df["Leg_Return"] = (self.df[self.price_col] - last_pivot_price) / last_pivot_price
            self.df["Leg_Return"] = self.df["Leg_Return"].fillna(0)
            
            # NEW: Cycle Phase (Dynamic Binning)
            # Uptrend Avg = 31 days, Downtrend Avg = 19 days
            # We want to categorize "Days Since" into 0 (Early), 1 (Mid), 2 (Late)
            # depending on the CURRENT Trend.
            
            # Ensure Last_Signal is filled
            last_sig = self.df["Last_Signal"].ffill().fillna(0)
            
            # Define Expected Duration based on state
            # If Last_Signal == 1 (Dip happened, we are in Uptrend) -> Expect 31 days
            # If Last_Signal == -1 (Peak happened, we are in Downtrend) -> Expect 19 days
            expected_duration = np.where(last_sig == 1, 31.0, 19.0)
            
            progress = self.df["Days_Since_Last_Pivot"] / expected_duration
            
            # Binning: <0.4 -> 0, 0.4-0.8 -> 1, >0.8 -> 2
            conditions = [
                (progress < 0.4),
                (progress >= 0.4) & (progress < 0.8),
                (progress >= 0.8)
            ]
            choices = [0.0, 1.0, 2.0] # Float for ML friendly
            
            self.df["Cycle_Phase"] = np.select(conditions, choices, default=1.0)
            
            # Remove raw/continuous features to prevent overfitting
            # self.df["Cycle_Progress"] = ... (Removed)
            
        except Exception as e:
            # Fallback for integer index or error
            self.df["Days_Since_Last_Pivot"] = 0
            self.df["Cycle_Progress"] = 0.0
            
        return self.df

    def add_advanced_stats(self):
        """Calculates Volume-Price and Statistical features for high-precision detection."""
        # 1. Money Flow Index (MFI) - RSI for Volume
        # Typical Price = (High + Low + Close) / 3
        tp = (self.df["high"] + self.df["low"] + self.df[self.price_col]) / 3
        mf = tp * self.df["vol."]
        
        delta_tp = tp.diff()
        pos_mf = mf.where(delta_tp > 0, 0).rolling(14).sum()
        neg_mf = mf.where(delta_tp < 0, 0).rolling(14).sum()
        
        mfr = pos_mf / neg_mf
        self.df["MFI_14"] = 100 - (100 / (1 + mfr))
        
        # 2. Volume Z-Score (Institutional Activity)
        vol_mean = self.df["vol."].rolling(20).mean()
        vol_std = self.df["vol."].rolling(20).std()
        self.df["Vol_ZScore"] = (self.df["vol."] - vol_mean) / (vol_std + 1e-9)
        
        # 3. Kurtosis (Fat Tails / Reversal Prep)
        self.df["Kurtosis_20"] = self.df[self.price_col].pct_change().rolling(20).kurt()
        
        # 4. Fractal signals (Higher/Lower extremes in window)
        # 4. Fractal signals (Higher/Lower extremes in window)
        # changed center=True to False to prevent look-ahead bias
        self.df["Fractal_High"] = self.df["high"].rolling(5).max() == self.df["high"]
        self.df["Fractal_Low"] = self.df["low"].rolling(5).min() == self.df["low"]
        
        return self.df

    def add_derived_features(self):
        """Calculates Volatility Adjusted Momentum and other derived metrics."""
        self.add_atr(14)
        
        # Volatility Adjusted Momentum (VAM)
        # 5-day return / ATR
        returns_5d = self.df[self.price_col].diff(5)
        self.df["VAM"] = returns_5d / self.df["ATR"]
        
        # Bollinger Bands Interaction
        if "BB_Lower" not in self.df.columns:
            self.add_bollinger_bands()
            
        self.df["Dist_BB_Lower"] = (self.df[self.price_col] - self.df["BB_Lower"]) / self.df["BB_Lower"]
        self.df["Dist_BB_Upper"] = (self.df["BB_Upper"] - self.df[self.price_col]) / self.df["BB_Upper"]
        
        # Candle Features
        self.add_candle_features()
        
        # NEW: RSI & Time Features
        self.add_rsi_features()
        self.add_time_features()
        self.add_divergence_features() # Divergence Check
        
        # NEW: Advanced Optimization Features
        self.add_advanced_stats()
        
        # NEW: Cycle & Exhaustion Features
        self.add_cycle_exhaustion_features()
        
        return self.df
    
    def add_bollinger_bands(self, period: int = 20, std: int = 2):
        """Calculates Bollinger Bands."""
        sma = self.df[self.price_col].rolling(window=period).mean()
        sigma = self.df[self.price_col].rolling(window=period).std()
        self.df["BB_Middle"] = sma
        self.df["BB_Upper"] = sma + (sigma * std)
        self.df["BB_Lower"] = sma - (sigma * std)
        return self.df

    def add_candle_features(self):
        """Calculates candle shadows for pattern detection."""
        # Ensure col names match CSV check results: lower case 'open', 'high', 'low'
        # based on user provided `head` output: 'date,price,open,high,low,...'
        # But 'price' might be close. Let's handle case sensitivity.
        
        # Helper to find column regardless of case
        def get_col(name):
            for c in self.df.columns:
                if c.lower() == name.lower():
                    return self.df[c]
            return self.df[self.price_col] # Fallback
            
        open_ = get_col("open")
        high = get_col("high")
        low = get_col("low")
        close = self.df[self.price_col]
        
        body_size = (close - open_).abs()
        total_range = high - low
        
        # Lower Shadow: distance from min(open, close) to low
        lower_shadow = pd.concat([open_, close], axis=1).min(axis=1) - low
        
        # Upper Shadow: distance from high to max(open, close)
        upper_shadow = high - pd.concat([open_, close], axis=1).max(axis=1)
        
        # Avoid division by zero
        total_range = total_range.replace(0, 1)
        
        self.df["Lower_Shadow_Ratio"] = lower_shadow / total_range
        self.df["Upper_Shadow_Ratio"] = upper_shadow / total_range
        self.df["Body_Ratio"] = body_size / total_range
        
        return self.df

    def add_cycle_exhaustion_features(self):
        """
        Calculates advanced cycle and exhaustion features for better dip/peak prediction:
        - Cycle Length: Length of current leg (dip-to-peak or peak-to-dip)
        - Price Exhaustion: How much price has moved in recent periods
        - Momentum Decay: Rate of RSI decline after peaks
        - Volatility Expansion: ATR expansion rate
        - Trend Exhaustion: Multiple exhaustion signals combined
        """
        # Ensure required columns exist
        if "RSI" not in self.df.columns:
            self.add_rsi()
        if "ATR" not in self.df.columns:
            self.add_atr(14)
        if "Dip" not in self.df.columns or "Tepe" not in self.df.columns:
            # ZigZag labels should be added before this
            pass
        
        # =====================================================
        # 1. CYCLE LENGTH (Current leg duration)
        # =====================================================
        # Calculate days since last pivot (already exists, but we need cycle length)
        if "Days_Since_Last_Pivot" in self.df.columns:
            # Current cycle length is Days_Since_Last_Pivot
            self.df["Cycle_Length"] = self.df["Days_Since_Last_Pivot"]
        else:
            self.df["Cycle_Length"] = 0
        
        # Historical average cycle length (rolling window of last 5 cycles)
        if "Last_Signal" in self.df.columns:
            # Find pivot points using integer positions
            is_pivot = self.df["Dip"].notna() | self.df["Tepe"].notna()
            pivot_positions = self.df[is_pivot].index.tolist()
            
            if len(pivot_positions) >= 2:
                # Calculate cycle lengths between pivots (using integer positions)
                cycle_lengths = []
                for i in range(1, len(pivot_positions)):
                    # Get integer position in dataframe
                    pos1 = self.df.index.get_loc(pivot_positions[i-1])
                    pos2 = self.df.index.get_loc(pivot_positions[i])
                    cycle_len = pos2 - pos1
                    cycle_lengths.append(float(cycle_len))
                
                # Rolling average of last 5 cycles
                if len(cycle_lengths) > 0:
                    # Map each row to its average cycle length
                    avg_cycle_values = []
                    for idx_pos in range(len(self.df)):
                        # Find which cycle this position belongs to
                        cycle_idx = 0
                        for i in range(1, len(pivot_positions)):
                            pos1 = self.df.index.get_loc(pivot_positions[i-1])
                            pos2 = self.df.index.get_loc(pivot_positions[i])
                            if pos1 <= idx_pos < pos2:
                                cycle_idx = i - 1
                                break
                        
                        if cycle_idx < len(cycle_lengths):
                            # Use last 5 cycles for average
                            recent_cycles = cycle_lengths[max(0, cycle_idx-4):cycle_idx+1]
                            avg_cycle_values.append(np.mean(recent_cycles) if recent_cycles else 0.0)
                        else:
                            # Use overall average
                            avg_cycle_values.append(np.mean(cycle_lengths[-5:]) if len(cycle_lengths) >= 5 else np.mean(cycle_lengths) if cycle_lengths else 0.0)
                    
                    self.df["Avg_Cycle_Length"] = avg_cycle_values
                    # Normalized cycle length (current / average)
                    self.df["Cycle_Length_Ratio"] = np.where(
                        self.df["Avg_Cycle_Length"] > 0,
                        self.df["Cycle_Length"] / self.df["Avg_Cycle_Length"],
                        1.0
                    )
                else:
                    self.df["Avg_Cycle_Length"] = 0.0
                    self.df["Cycle_Length_Ratio"] = 1.0
            else:
                self.df["Avg_Cycle_Length"] = 0.0
                self.df["Cycle_Length_Ratio"] = 1.0
        else:
            self.df["Avg_Cycle_Length"] = 0.0
            self.df["Cycle_Length_Ratio"] = 1.0
        
        # =====================================================
        # 2. PRICE EXHAUSTION (How much price moved recently)
        # =====================================================
        # Price exhaustion in last 5, 10, 20 days
        price_5d = self.df[self.price_col].pct_change(5)
        price_10d = self.df[self.price_col].pct_change(10)
        price_20d = self.df[self.price_col].pct_change(20)
        
        self.df["Price_Exhaustion_5D"] = price_5d.fillna(0)
        self.df["Price_Exhaustion_10D"] = price_10d.fillna(0)
        self.df["Price_Exhaustion_20D"] = price_20d.fillna(0)
        
        # Normalized by ATR (volatility-adjusted exhaustion)
        atr_normalized = self.df["ATR"] / self.df[self.price_col]
        price_exhaustion_atr = np.where(
            atr_normalized > 0,
            price_5d / atr_normalized,
            0
        )
        self.df["Price_Exhaustion_5D_ATR"] = pd.Series(price_exhaustion_atr, index=self.df.index).fillna(0)
        
        # Maximum price move in last N days (peak exhaustion)
        rolling_high_10 = self.df[self.price_col].rolling(10).max()
        rolling_low_10 = self.df[self.price_col].rolling(10).min()
        self.df["Price_Range_10D"] = (rolling_high_10 - rolling_low_10) / rolling_low_10
        
        # =====================================================
        # 3. MOMENTUM DECAY (RSI decline rate after peaks)
        # =====================================================
        # RSI momentum decay: How fast RSI is falling
        rsi_decay_1d = -self.df["RSI"].diff(1)  # Negative diff = decay
        rsi_decay_3d = -self.df["RSI"].diff(3) / 3
        rsi_decay_5d = -self.df["RSI"].diff(5) / 5
        
        self.df["RSI_Decay_1D"] = rsi_decay_1d.fillna(0)
        self.df["RSI_Decay_3D"] = rsi_decay_3d.fillna(0)
        self.df["RSI_Decay_5D"] = rsi_decay_5d.fillna(0)
        
        # RSI acceleration (change in decay rate)
        self.df["RSI_Decay_Accel"] = self.df["RSI_Decay_1D"].diff().fillna(0)
        
        # =====================================================
        # 4. VOLATILITY EXPANSION (ATR expansion rate)
        # =====================================================
        # ATR expansion: How much volatility has increased
        atr_expansion_5d = self.df["ATR"].pct_change(5)
        atr_expansion_10d = self.df["ATR"].pct_change(10)
        
        self.df["ATR_Expansion_5D"] = atr_expansion_5d.fillna(0)
        self.df["ATR_Expansion_10D"] = atr_expansion_10d.fillna(0)
        
        # ATR vs historical average (last 20 days)
        atr_avg_20 = self.df["ATR"].rolling(20).mean()
        atr_vs_avg = np.where(
            atr_avg_20 > 0,
            (self.df["ATR"] - atr_avg_20) / atr_avg_20,
            0
        )
        self.df["ATR_vs_Avg"] = pd.Series(atr_vs_avg, index=self.df.index).fillna(0)
        
        # =====================================================
        # 5. TREND EXHAUSTION COMPOSITE
        # =====================================================
        # Combine multiple exhaustion signals
        # High RSI + Long overbought duration + Price exhaustion + Cycle length
        rsi_exhaustion = (self.df["RSI"] > 75).astype(int)
        ob_duration_exhaustion = (self.df["RSI_Overbought_Days"] > 5).astype(int)
        price_exhaustion = (self.df["Price_Exhaustion_10D"] > 0.05).astype(int)  # 5%+ move
        cycle_exhaustion = (self.df["Cycle_Length_Ratio"] > 1.2).astype(int)  # 20% longer than avg
        
        # Composite exhaustion score (0-4)
        self.df["Trend_Exhaustion_Score"] = (
            rsi_exhaustion + 
            ob_duration_exhaustion + 
            price_exhaustion + 
            cycle_exhaustion
        )
        
        # For dips: Oversold + Long oversold duration + Price exhaustion down
        rsi_oversold = (self.df["RSI"] < 25).astype(int)
        os_duration_exhaustion = (self.df.get("RSI_Oversold_Days", 0) > 3).astype(int)
        price_exhaustion_down = (self.df["Price_Exhaustion_10D"] < -0.05).astype(int)  # -5%+ move
        
        self.df["Dip_Exhaustion_Score"] = (
            rsi_oversold + 
            os_duration_exhaustion + 
            price_exhaustion_down + 
            cycle_exhaustion
        )
        
        # =====================================================
        # 6. PRICE BLOWOFF (Extreme price moves)
        # =====================================================
        # Price blowoff: Extreme moves that often precede reversals
        price_change_abs = self.df[self.price_col].pct_change().abs()
        blowoff_threshold = price_change_abs.rolling(20).quantile(0.95)  # Top 5% moves
        
        self.df["Price_Blowoff"] = (price_change_abs > blowoff_threshold).astype(int)
        
        # Volume-confirmed blowoff (if volume data exists)
        if "vol." in self.df.columns:
            vol_avg = self.df["vol."].rolling(20).mean()
            vol_spike = (self.df["vol."] > vol_avg * 1.5).astype(int)
            self.df["Volume_Blowoff"] = (self.df["Price_Blowoff"] & vol_spike).astype(int)
        else:
            self.df["Volume_Blowoff"] = 0
        
        return self.df

    def get_df(self):
        return self.df
