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

    def add_local_extrema(self, order: int = 5):
        """
        Identifies local peaks (Tepe) and dips (Dip) based on the given order/window.
        Adds 'Tepe' and 'Dip' columns to the internal dataframe.
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
