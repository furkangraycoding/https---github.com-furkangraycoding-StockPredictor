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

    def add_zigzag_labels(self, threshold_pct: float = 0.05):
        """
        Identifies turning points using the ZigZag algorithm.
        Also adds 'Last_Signal' state: 1 (Dip confirmed, looking for Peak), -1 (Peak confirmed, looking for Dip).
        threshold_pct: Minimum percentage movement to change trend (0.05 = 5%).
        """
        if self.price_col not in self.df.columns:
            return self.df
            
        prices = self.df[self.price_col].values
        n = len(prices)
        if n < 2:
            return self.df
            
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
            
            if trend == 1: # Uptrend (Last confirmed was Dip)
                if price > last_pivot_price:
                    # New high in existing uptrend
                    last_pivot_price = price
                    last_pivot_idx = i
                elif price < last_pivot_price * (1 - threshold_pct):
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
                elif price > last_pivot_price * (1 + threshold_pct):
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
        """Calculates RSI changes."""
        if "RSI" not in self.df.columns: self.add_rsi()
        
        self.df["RSI_Diff_1D"] = self.df["RSI"].diff(1)
        self.df["RSI_Diff_3D"] = self.df["RSI"].diff(3)
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
        # Ensure types match
        try:
            time_diff = (dates - last_pivot_date).dt.days
            self.df["Days_Since_Last_Pivot"] = time_diff.fillna(0)
            
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
        self.df["Fractal_High"] = self.df["high"].rolling(5, center=True).max() == self.df["high"]
        self.df["Fractal_Low"] = self.df["low"].rolling(5, center=True).min() == self.df["low"]
        
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
        
        # NEW: Advanced Optimization Features
        self.add_advanced_stats()
        
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

    def get_df(self):
        return self.df
