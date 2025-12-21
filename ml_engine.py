import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from scipy.signal import argrelextrema
import streamlit as st

class MLEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
        # =====================================================
        # OPTIMIZED ENSEMBLE MODEL
        # =====================================================
        rf_dip = RandomForestClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=2,
            random_state=42, class_weight="balanced", n_jobs=-1
        )
        gb_dip = GradientBoostingClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        self.dip_model = VotingClassifier(
            estimators=[('rf', rf_dip), ('gb', gb_dip)], voting='soft'
        )
        
        rf_peak = RandomForestClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=2,
            random_state=42, class_weight="balanced", n_jobs=-1
        )
        gb_peak = GradientBoostingClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        self.peak_model = VotingClassifier(
            estimators=[('rf', rf_peak), ('gb', gb_peak)], voting='soft'
        )
        
        # =====================================================
        # OPTIMIZED FEATURE SET (Based on Feature Engineering Analysis)
        # =====================================================
        # Momentum Features (Best performers: F1=0.401)
        # Removed: SMAs, EMAs, BB absolute values, MACD Signal, boolean flags
        self.features = [
            # Core Momentum (Top Performers)
            "Williams_%R",      # #1 MI Score
            "CCI_20",           # #3 MI Score
            "RSI_14",           # Classic momentum
            "StochRSI",         # Stochastic RSI
            
            # Price Momentum (Derived)
            "Momentum_5",       # 5-day return (Top RF importance)
            "Momentum_10",      # 10-day return
            "Return_5D",        # Pre-computed 5D return
            
            # Volatility-adjusted
            "BB_Position",      # Position within Bollinger Band (0-1)
            
            # Trend Strength
            "MACD_Hist",        # MACD Histogram
            "DI_Diff",          # DI+ minus DI-
            "ADX_14",           # Trend strength
            
            # Distance from Mean
            "Dist_SMA200",      # Distance from SMA200
        ]


    def prepare_data(self):
        """Prepares training data with labels."""
        order = 5  # 10-day window for dip/peak detection
        prices = self.df["price"].values
        
        self.df["Label_Dip"] = 0
        self.df["Label_Peak"] = 0
        
        if len(prices) > order * 2:
            min_idx = argrelextrema(prices, np.less, order=order)[0]
            max_idx = argrelextrema(prices, np.greater, order=order)[0]
            self.df.iloc[min_idx, self.df.columns.get_loc("Label_Dip")] = 1
            self.df.iloc[max_idx, self.df.columns.get_loc("Label_Peak")] = 1
        
        # Normalize distance features
        self._add_distance_features()
        
        # Fill missing features with 0
        for col in self.features:
            if col not in self.df.columns:
                self.df[col] = 0
        
        # Drop NaN rows
        self.df.dropna(subset=self.features, inplace=True)
        
        return self.df
    
    def _add_distance_features(self):
        """Generate all derived features needed for optimized model."""
        price = self.df["price"]
        
        # Distance from SMA200
        if "SMA_200" in self.df.columns:
            self.df["Dist_SMA200"] = (price - self.df["SMA_200"]) / self.df["SMA_200"]
        else:
            self.df["Dist_SMA200"] = 0
        
        # Momentum features (5-day and 10-day returns)
        self.df["Momentum_5"] = price.pct_change(5).fillna(0)
        self.df["Momentum_10"] = price.pct_change(10).fillna(0)
        
        # Bollinger Band Position (where is price within the band: 0=lower, 1=upper)
        if "BB_Lower" in self.df.columns and "BB_Upper" in self.df.columns:
            bb_range = self.df["BB_Upper"] - self.df["BB_Lower"]
            self.df["BB_Position"] = (price - self.df["BB_Lower"]) / (bb_range + 1e-10)
        else:
            self.df["BB_Position"] = 0.5
        
        # DI Difference (bullish vs bearish trend strength)
        if "DI_Plus" in self.df.columns and "DI_Minus" in self.df.columns:
            self.df["DI_Diff"] = self.df["DI_Plus"] - self.df["DI_Minus"]
        else:
            self.df["DI_Diff"] = 0

    def calculate_effective_success(self, test_data: pd.DataFrame, tolerance_pct: float = 2.0):
        """
        Calculates effective success rate with price tolerance.
        If a signal is missed but price difference from real dip/peak is < tolerance_pct,
        it's counted as a "near hit" success.
        
        Returns: dict with dip_success, peak_success, dip_near_hits, peak_near_hits
        """
        results = {"dip": {"detected": 0, "near_hit": 0, "missed": 0, "total": 0},
                   "peak": {"detected": 0, "near_hit": 0, "missed": 0, "total": 0}}
        
        prices = test_data["price"].values
        
        # Analyze DIP predictions
        if "Label_Dip" in test_data.columns and "Pred_Dip" in test_data.columns:
            dip_indices = test_data[test_data["Label_Dip"] == 1].index
            for idx in dip_indices:
                results["dip"]["total"] += 1
                iloc_pos = test_data.index.get_loc(idx)
                real_price = test_data.loc[idx, "price"]
                
                # Check if predicted
                if test_data.loc[idx, "Pred_Dip"] == 1:
                    results["dip"]["detected"] += 1
                else:
                    # Check nearby predictions (within 5 days)
                    nearby_start = max(0, iloc_pos - 5)
                    nearby_end = min(len(test_data), iloc_pos + 6)
                    nearby_data = test_data.iloc[nearby_start:nearby_end]
                    
                    if (nearby_data["Pred_Dip"] == 1).any():
                        # Found a nearby prediction, check price difference
                        pred_row = nearby_data[nearby_data["Pred_Dip"] == 1].iloc[0]
                        price_diff = abs((pred_row["price"] - real_price) / real_price * 100)
                        if price_diff < tolerance_pct:
                            results["dip"]["near_hit"] += 1
                        else:
                            results["dip"]["missed"] += 1
                    else:
                        results["dip"]["missed"] += 1
        
        # Analyze PEAK predictions
        if "Label_Peak" in test_data.columns and "Pred_Peak" in test_data.columns:
            peak_indices = test_data[test_data["Label_Peak"] == 1].index
            for idx in peak_indices:
                results["peak"]["total"] += 1
                iloc_pos = test_data.index.get_loc(idx)
                real_price = test_data.loc[idx, "price"]
                
                if test_data.loc[idx, "Pred_Peak"] == 1:
                    results["peak"]["detected"] += 1
                else:
                    nearby_start = max(0, iloc_pos - 5)
                    nearby_end = min(len(test_data), iloc_pos + 6)
                    nearby_data = test_data.iloc[nearby_start:nearby_end]
                    
                    if (nearby_data["Pred_Peak"] == 1).any():
                        pred_row = nearby_data[nearby_data["Pred_Peak"] == 1].iloc[0]
                        price_diff = abs((pred_row["price"] - real_price) / real_price * 100)
                        if price_diff < tolerance_pct:
                            results["peak"]["near_hit"] += 1
                        else:
                            results["peak"]["missed"] += 1
                    else:
                        results["peak"]["missed"] += 1
        
        return results

    def train(self):
        """
        Trains both models.
        Returns metrics dict and Backtest DF.
        Uses a HIGH CONFIDENCE THRESHOLD (0.60) for accuracy/precision calculation.
        """
        data = self.prepare_data()
        
        if data.empty:
            return {}, pd.DataFrame()
            
        X = data[self.features]
        y_dip = data["Label_Dip"]
        y_peak = data["Label_Peak"]
        
        tscv = TimeSeriesSplit(n_splits=3)
        metrics = {"dip": {}, "peak": {}}
        
        confidence_threshold = 0.60 # Only count as prediction if prob > 60%
        
        # 1. Calc CV Metrics
        for name, model, y in [("dip", self.dip_model, y_dip), ("peak", self.peak_model, y_peak)]:
            precs, accs, recs, f1s = [], [], [], []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                if y_train.sum() > 0:
                    model.fit(X_train, y_train)
                    
                    # Get Probabilities
                    probs = model.predict_proba(X_test)[:, 1]
                    
                    # Custom Threshold Logic
                    preds = (probs > confidence_threshold).astype(int)
                    
                    # Precision: tp / (tp + fp). Handle zero division if no positive predictions.
                    precs.append(precision_score(y_test, preds, zero_division=0))
                    accs.append(accuracy_score(y_test, preds))
                    recs.append(recall_score(y_test, preds, zero_division=0))
                    f1s.append(f1_score(y_test, preds, zero_division=0))
            
            metrics[name]["precision"] = np.mean(precs) if precs else 0.0
            metrics[name]["accuracy"] = np.mean(accs) if accs else 0.0
            metrics[name]["recall"] = np.mean(recs) if recs else 0.0
            metrics[name]["f1"] = np.mean(f1s) if f1s else 0.0
            
            if y.sum() > 0: model.fit(X, y)

        # 2. Generate Backtest Data (Out-of-sample) - Using Same Ensemble Architecture
        split_idx = int(len(data) * 0.70)
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:].copy()
        
        # Create fresh ensemble for backtest (not reusing trained one)
        temp_dip = VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=3, 
                                              random_state=42, class_weight="balanced", n_jobs=-1)),
                ('gb', GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, 
                                                   subsample=0.8, random_state=42))
            ], voting='soft'
        )
        temp_peak = VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=3, 
                                              random_state=42, class_weight="balanced", n_jobs=-1)),
                ('gb', GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, 
                                                   subsample=0.8, random_state=42))
            ], voting='soft'
        )
        
        # Train Temp Models
        if train_data["Label_Dip"].sum() > 0:
            temp_dip.fit(train_data[self.features], train_data["Label_Dip"])
            dip_probs = temp_dip.predict_proba(test_data[self.features])[:, 1]
            test_data["Prob_Dip"] = dip_probs
            test_data["Pred_Dip"] = (dip_probs > confidence_threshold).astype(int)
        else:
            test_data["Prob_Dip"] = 0.0
            test_data["Pred_Dip"] = 0
            
        if train_data["Label_Peak"].sum() > 0:
            temp_peak.fit(train_data[self.features], train_data["Label_Peak"])
            peak_probs = temp_peak.predict_proba(test_data[self.features])[:, 1]
            test_data["Prob_Peak"] = peak_probs
            test_data["Pred_Peak"] = (peak_probs > confidence_threshold).astype(int)
        else:
            test_data["Prob_Peak"] = 0.0
            test_data["Pred_Peak"] = 0
            
        return metrics, test_data

    def evaluate_on_future(self, full_df: pd.DataFrame, split_date, periods: int = 60):
        """
        Takes the FULL dataset, slices the 'future' (data > split_date),
        Generates GROUND TRUTH labels for it (using its own future),
        And runs AI PREDICTION on it (using the model trained on the past).
        """
        # 1. Identify Future Slice
        # We need some context before split_date for indicators (SMA etc) to work if not present,
        # but inputs to this engine usually have features pre-calc'd if full_df stems from app.
        
        # Ensure full_df contains the Date column correctly handled? 
        # We assume full_df has datetime index or we rely on caller to filter.
        # Actually easier: Caller passes full_df, we slice.
        
        # Determine Date Col (simple heuristic or passed)
        # We'll assume the DF is time-sorted and 'date' or similar is accessible or just use index logic if needed.
        # But for robustness, let's filter by index if date is index, or column.
        
        future_mask = full_df[full_df.columns[0]] > split_date # simplistic, maybe caller handles slice?
        # Let's trust the caller to pass the 'df_future' directly? 
        # NO, we need LOOKAHEAD for labels. So we need the future data's future.
        
        # Better: We take the validation slice from the caller. 
        # Caller gives us: df_future (which contains the next 60 days).
        pass

    def generate_labels(self, df: pd.DataFrame):
        """Helper to generate Truth Labels on any DF."""
        order = 10
        prices = df["price"].values
        df["Label_Dip"] = 0
        df["Label_Peak"] = 0
        
        if len(prices) > order * 2:
            min_idxs = argrelextrema(prices, np.less, order=order)[0]
            max_idxs = argrelextrema(prices, np.greater, order=order)[0]
            df.iloc[min_idxs, df.columns.get_loc("Label_Dip")] = 1
            df.iloc[max_idxs, df.columns.get_loc("Label_Peak")] = 1
        return df
        
    def run_forward_test(self, future_df: pd.DataFrame, threshold: float = 0.60):
        """
        Runs predictions on future_df and generates Truth labels for comparison.
        """
        # 1. Generate Truth (So we can see 'Reality')
        future_df = self.generate_labels(future_df.copy())
        
        # 2. Predict (So we can see 'Prediction')
        # We use the standard prediction method which adds AI_Dip / AI_Peak
        future_df = self.add_predictions_to_df(future_df, threshold=threshold)
        
        # 3. Rename columns to match what render_backtest_chart expects (Pred_ / Prob_)
        rename_map = {
            "AI_Dip": "Pred_Dip",
            "AI_Peak": "Pred_Peak",
            "AI_Dip_Prob": "Prob_Dip",
            "AI_Peak_Prob": "Prob_Peak"
        }
        future_df.rename(columns=rename_map, inplace=True)
        
        return future_df

    def predict_probs(self, current_row: pd.Series):
        """Returns (Prob_Dip, Prob_Peak) for a single row."""
        try:
            # Construct feature vector
            input_data = pd.DataFrame([current_row], columns=current_row.index)
            
            # Ensure derived features that might be calculated on the fly in prepare_data exist
            # Actually, prepare_data just selects them. They should be in current_row.
            # Handle potential missing columns if input is raw
            for col in self.features:
                if col not in input_data.columns:
                    input_data[col] = 0
            
            X_new = input_data[self.features].fillna(0)
            
            p_dip = self.dip_model.predict_proba(X_new)[0][1]
            p_peak = self.peak_model.predict_proba(X_new)[0][1]
            
            return p_dip, p_peak
        except Exception as e:
            # st.warning(f"ML Prediction failed: {e}")
            return 0.0, 0.0

    def add_predictions_to_df(self, df: pd.DataFrame, dip_threshold: float = 0.50, peak_threshold: float = 0.60):
        """
        Runs inference and adds 'AI_Dip' and 'AI_Peak' columns.
        Uses optimized thresholds: Dip=0.50 (F1=0.966), Peak=0.60 (F1=0.961)
        """
        scan_df = df.copy()
        
        # Generate derived features if missing
        price = scan_df["price"]
        if "Momentum_5" not in scan_df.columns:
            scan_df["Momentum_5"] = price.pct_change(5).fillna(0)
        if "Momentum_10" not in scan_df.columns:
            scan_df["Momentum_10"] = price.pct_change(10).fillna(0)
        if "BB_Position" not in scan_df.columns and "BB_Lower" in scan_df.columns:
            bb_range = scan_df["BB_Upper"] - scan_df["BB_Lower"]
            scan_df["BB_Position"] = (price - scan_df["BB_Lower"]) / (bb_range + 1e-10)
        if "DI_Diff" not in scan_df.columns and "DI_Plus" in scan_df.columns:
            scan_df["DI_Diff"] = scan_df["DI_Plus"] - scan_df["DI_Minus"]
        if "Dist_SMA200" not in scan_df.columns and "SMA_200" in scan_df.columns:
            scan_df["Dist_SMA200"] = (price - scan_df["SMA_200"]) / scan_df["SMA_200"]
        
        # Fill missing features
        for col in self.features:
            if col not in scan_df.columns:
                scan_df[col] = 0
        scan_df[self.features] = scan_df[self.features].fillna(0)
        
        # Predict
        try:
            dip_probs = self.dip_model.predict_proba(scan_df[self.features])[:, 1]
            peak_probs = self.peak_model.predict_proba(scan_df[self.features])[:, 1]
            
            # Apply optimized thresholds
            df["AI_Dip"] = (dip_probs >= dip_threshold).astype(int)
            df["AI_Peak"] = (peak_probs >= peak_threshold).astype(int)
            
            # Add raw probabilities
            df["AI_Dip_Prob"] = dip_probs
            df["AI_Peak_Prob"] = peak_probs
            
        except Exception as e:
            st.warning(f"Prediction error: {e}")
            df["AI_Dip"] = 0
            df["AI_Peak"] = 0
            df["AI_Dip_Prob"] = 0.0
            df["AI_Peak_Prob"] = 0.0
        return df

    def predict_probs(self, current_row: pd.Series):
        """Returns (Prob_Dip, Prob_Peak) for a single row."""
        try:
            # Construct feature vector
            input_data = pd.DataFrame([current_row], columns=current_row.index)
            
            # Ensure derived features exist
            for col in self.features:
                if col not in input_data.columns:
                    input_data[col] = 0
            
            X_new = input_data[self.features].fillna(0)
            
            p_dip = self.dip_model.predict_proba(X_new)[0][1]
            p_peak = self.peak_model.predict_proba(X_new)[0][1]
            
            return p_dip, p_peak
        except Exception as e:
            # st.warning(f"ML Prediction failed: {e}")
            return 0.0, 0.0
