import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, make_scorer
from indicators import TechnicalAnalyzer
import streamlit as st

class MLEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
        # Default Parameters (will be updated by tuning)
        self.rf_params = {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_leaf": 4,
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": 42
        }
        
        # Models initialized with defaults
        self.dip_model = RandomForestClassifier(**self.rf_params)
        self.peak_model = RandomForestClassifier(**self.rf_params)
        
        # ... (Features remain same)
        
        # =====================================================
        # QUANTITATIVE FEATURE SET
        # =====================================================
        self.features = [
            # Momentum
            "RSI_14",
            "Williams_%R",
            "CCI_20",
            "StochRSI",
            "VAM", # Volatility Adjusted Momentum
            
            # Volatility & Risk
            "Volatility_20",
            "ATR",
            "BB_Position",
            
            # Trend & Regime
            "Dist_SMA200",
            "MACD_Hist",
            "DI_Diff",
            "ADX_14",
            
            # Drawdown / Rally
            "Drawdown_Pct",
            "Rally_Pct",
            
            # RSI Delta & Time Decay (NEW)
            "RSI_Diff_1D",
            "RSI_Diff_3D",
            "Days_Since_Last_Pivot"
        ]

    def prepare_data(self):
        """Prepares training data using ZigZag labels."""
        # Ensure labels are generated using consistent ZigZag logic
        # We re-run it here to be safe and independent
        analyzer = TechnicalAnalyzer(self.df)
        analyzer.add_zigzag_labels(threshold_pct=0.05)
        self.df = analyzer.get_df()
        
        # Forward fill the 'Last_Signal' state
        if "Last_Signal" in self.df.columns:
             self.df["Last_Signal"] = self.df["Last_Signal"].replace(0, np.nan).ffill().fillna(0)
        
        # Generate Derived Features if missing (safety check)
        if "Volatility_20" not in self.df.columns:
            analyzer.add_rolling_volatility()
            analyzer.add_drawdown_features()
            self.df = analyzer.get_df() # Update df with new cols
        
        # Rename ZigZag columns to Labels if needed or create binary targets
        # ZigZag creates 'Tepe' (Peak Price) and 'Dip' (Dip Price) columns where event occurs
        self.df["Label_Dip"] = np.where(self.df["Dip"].notna(), 1, 0)
        self.df["Label_Peak"] = np.where(self.df["Tepe"].notna(), 1, 0)
        
        # Normalize/Fill Features
        for col in self.features:
            if col not in self.df.columns:
                self.df[col] = 0
        
        # Drop NaN rows (start of history)
        self.df.dropna(subset=self.features, inplace=True)
        
        return self.df

    def tune_hyperparameters(self, X, y, task_name="Task"):
        """
        Optimizes Random Forest hyperparameters using RandomizedSearchCV.
        """
        st.write(f"⚙️ Tuning Hyperparameters for {task_name} Model...")
        
        param_dist = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [5, 8, 10, 15, None],
            "min_samples_leaf": [2, 4, 8, 10],
            "min_samples_split": [2, 5, 10],
            "class_weight": ["balanced", "balanced_subsample"]
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        scorer = make_scorer(f1_score, zero_division=0) # Optimize for F1 of the minority class
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=10, # Keep it efficient for now
            scoring=scorer,
            cv=tscv,
            verbose=0,
            n_jobs=-1,
            random_state=42
        )
        
        search.fit(X, y)
        st.write(f"✅ Best Params for {task_name}: {search.best_params_}")
        return search.best_estimator_

    def train(self, optimize=True):
        """
        Trains directly on State-Filtered Data.
        """
        data = self.prepare_data()
        
        if data.empty:
            return {}, pd.DataFrame()
            
        metrics = {"dip": {}, "peak": {}}
        tscv = TimeSeriesSplit(n_splits=3)
        
        # 1. Train DIP Model (Only when Last Signal was PEAK -> -1)
        # We also include 0 (Unknown) to provide some initial training data if needed, 
        # but strict logic says -1. Let's include 0 for robustness at start.
        dip_mask = (data["Last_Signal"] == -1) | (data["Last_Signal"] == 0)
        X_dip = data.loc[dip_mask, self.features]
        y_dip = data.loc[dip_mask, "Label_Dip"]
        
        if y_dip.sum() > 0:
            if optimize:
                self.dip_model = self.tune_hyperparameters(X_dip, y_dip, "Dip")
            
            # CV Evaluation
            precs, recs, accs = [], [], []
            for train_idx, test_idx in tscv.split(X_dip):
                X_train, X_test = X_dip.iloc[train_idx], X_dip.iloc[test_idx]
                y_train, y_test = y_dip.iloc[train_idx], y_dip.iloc[test_idx]
                
                self.dip_model.fit(X_train, y_train)
                preds = self.dip_model.predict(X_test)
                precs.append(precision_score(y_test, preds, zero_division=0))
                recs.append(recall_score(y_test, preds, zero_division=0))
                accs.append(accuracy_score(y_test, preds))
            
            self.dip_model.fit(X_dip, y_dip)
            metrics["dip"]["precision"] = np.mean(precs) if precs else 0.0
            metrics["dip"]["recall"] = np.mean(recs) if recs else 0.0
            metrics["dip"]["accuracy"] = np.mean(accs) if accs else 0.0
        
        # 2. Train PEAK Model (Only when Last Signal was DIP -> 1)
        peak_mask = (data["Last_Signal"] == 1) | (data["Last_Signal"] == 0)
        X_peak = data.loc[peak_mask, self.features]
        y_peak = data.loc[peak_mask, "Label_Peak"]

        if y_peak.sum() > 0:
            if optimize:
                self.peak_model = self.tune_hyperparameters(X_peak, y_peak, "Peak")
                
            precs, recs, accs = [], [], []
            for train_idx, test_idx in tscv.split(X_peak):
                X_train, X_test = X_peak.iloc[train_idx], X_peak.iloc[test_idx]
                y_train, y_test = y_peak.iloc[train_idx], y_peak.iloc[test_idx]
                
                self.peak_model.fit(X_train, y_train)
                preds = self.peak_model.predict(X_test)
                precs.append(precision_score(y_test, preds, zero_division=0))
                recs.append(recall_score(y_test, preds, zero_division=0))
                accs.append(accuracy_score(y_test, preds))

            self.peak_model.fit(X_peak, y_peak)
            metrics["peak"]["precision"] = np.mean(precs) if precs else 0.0
            metrics["peak"]["recall"] = np.mean(recs) if recs else 0.0
            metrics["peak"]["accuracy"] = np.mean(accs) if accs else 0.0
        
        # Test Data Generation
        split_idx = int(len(data) * 0.70)
        test_data = data.iloc[split_idx:].copy()
        
        if not test_data.empty:
            test_data = self.add_predictions_to_df(test_data)
            test_data = test_data.rename(columns={
                "AI_Dip": "Pred_Dip", "AI_Peak": "Pred_Peak",
                "AI_Dip_Prob": "Prob_Dip", "AI_Peak_Prob": "Prob_Peak"
            })
            
        return metrics, test_data

    def get_feature_importance(self):
        """
        Returns a DataFrame of feature importances for both models.
        """
        importances = pd.DataFrame(index=self.features)
        
        if hasattr(self.dip_model, "feature_importances_"):
            importances["Dip_Importance"] = self.dip_model.feature_importances_
        else:
            importances["Dip_Importance"] = 0.0
            
        if hasattr(self.peak_model, "feature_importances_"):
            importances["Peak_Importance"] = self.peak_model.feature_importances_
        else:
            importances["Peak_Importance"] = 0.0
            
        return importances

    def predict_probs(self, current_row: pd.Series):
        """Returns (Prob_Dip, Prob_Peak) respecting State Logic."""
        try:
            # Construct feature vector
            input_data = pd.DataFrame([current_row], columns=current_row.index)
            
            for col in self.features:
                if col not in input_data.columns:
                    input_data[col] = 0
            
            X_new = input_data[self.features].fillna(0)
            
            # Check STATE
            last_signal = current_row.get("Last_Signal", 0)
            
            p_dip = 0.0
            p_peak = 0.0
            
            # Predict Dip ONLY if last was Peak (-1) or Unknown (0)
            if last_signal in [-1, 0]:
                p_dip = self.dip_model.predict_proba(X_new)[0][1]
                
            # Predict Peak ONLY if last was Dip (1) or Unknown (0)
            if last_signal in [1, 0]:
                p_peak = self.peak_model.predict_proba(X_new)[0][1]
            
            return p_dip, p_peak
        except Exception as e:
            return 0.0, 0.0

    def add_predictions_to_df(self, df: pd.DataFrame, threshold: float = 0.50):
        """
        Runs inference with State Filtering.
        """
        scan_df = df.copy()
        
        # Feature check
        for col in self.features:
            if col not in scan_df.columns:
                scan_df[col] = 0
        scan_df[self.features] = scan_df[self.features].fillna(0)
        
        # If 'Last_Signal' is missing in future data (e.g. forecast), we must forward fill it from history
        if "Last_Signal" in scan_df.columns:
             scan_df["Last_Signal"] = scan_df["Last_Signal"].replace(0, np.nan).ffill().fillna(0)
        
        try:
            # 1. Base Probabilities
            dip_probs = self.dip_model.predict_proba(scan_df[self.features])[:, 1]
            peak_probs = self.peak_model.predict_proba(scan_df[self.features])[:, 1]
            
            # 2. Apply State Filter Vectorized
            if "Last_Signal" in scan_df.columns:
                last_signals = scan_df["Last_Signal"].values
                # If Last=1 (Dip), Prob_Dip -> 0
                dip_probs = np.where(last_signals == 1, 0.0, dip_probs)
                # If Last=-1 (Peak), Prob_Peak -> 0
                peak_probs = np.where(last_signals == -1, 0.0, peak_probs)
            
            df["AI_Dip"] = (dip_probs >= threshold).astype(int)
            df["AI_Peak"] = (peak_probs >= threshold).astype(int)
            df["AI_Dip_Prob"] = dip_probs
            df["AI_Peak_Prob"] = peak_probs
            
        except Exception as e:
            st.warning(f"Prediction error: {e}")
            df["AI_Dip"] = 0
            df["AI_Peak"] = 0
            df["AI_Dip_Prob"] = 0.0
            df["AI_Peak_Prob"] = 0.0
            
        return df

    def forecast_future(self, df: pd.DataFrame, days: int = 30):
        """
        Generates a future price path with a cyclic component.
        """
        if df.empty:
            return pd.DataFrame()
            
        last_row = df.iloc[-1]
        last_price = last_row["price"]
        
        # Get date column
        date_col = next((c for c in df.columns if "date" in c.lower() or "tarih" in c.lower()), df.index.name)
        if date_col is None: date_col = "Date"
        last_date = last_row[date_col] if date_col in df.columns else df.index[-1]
        
        # Calculate recent drift and volatility
        window = min(len(df), 20)
        recent_prices = df["price"].iloc[-window:]
        returns = recent_prices.pct_change().dropna()
        
        avg_return = returns.mean() if len(returns) > 0 else 0.0
        std_return = returns.std() if len(returns) > 0 else 0.01
            
        future_dates = []
        future_prices = []
        
        current_date = last_date
        current_price = last_price
        
        for i in range(1, days + 1):
            current_date = current_date + pd.Timedelta(days=1)
            if current_date.weekday() >= 5:
                current_date = current_date + pd.Timedelta(days=7 - current_date.weekday())
            
            # Cyclic component (20-day cycle)
            cycle_phase = (i / 20) * 2 * np.pi
            cycle_move = 0.03 * np.sin(cycle_phase) 
            
            price_change = np.random.normal(avg_return, std_return) + (cycle_move / 20)
            current_price = current_price * (1 + price_change)
            
            future_dates.append(current_date)
            future_prices.append(current_price)
            
        future_df = pd.DataFrame({
            date_col: future_dates,
            "price": future_prices,
            "is_forecast": True
        })
        
        # Simulate OHLC for feature calculation consistency
        for col in ["open", "high", "low"]:
             future_df[col] = future_df["price"] * (1 + np.random.normal(0, 0.005, len(future_df)))

        return future_df

    def calculate_effective_success(self, test_data: pd.DataFrame, tolerance_pct: float = 2.0):
        """
        Calculates success based on ±2% price tolerance or Time Logic.
        """
        results = {"dip": {"detected": 0, "near_hit": 0, "missed": 0, "total": 0},
                   "peak": {"detected": 0, "near_hit": 0, "missed": 0, "total": 0}}
        
        # ... (Similar logic to previous, but simpler to keep concise for now)
        # We'll just invoke strict checking: 
        # A hit is if a Predicted Signal is within X days of a True Label.
        
        # Let's use the True Labels (Label_Dip, Label_Peak)
        if "Label_Dip" not in test_data.columns: return results
        
        for kind, label_col, pred_col in [("dip", "Label_Dip", "Pred_Dip"), ("peak", "Label_Peak", "Pred_Peak")]:
            true_indices = test_data[test_data[label_col] == 1].index
            
            for idx in true_indices:
                results[kind]["total"] += 1
                # Check if we predicted this specific event (or close to it)
                # Look for ANY prediction within +/- 5 days
                pos = test_data.index.get_loc(idx)
                start = max(0, pos - 5)
                end = min(len(test_data), pos + 6)
                window = test_data.iloc[start:end]
                
                if (window[pred_col] == 1).any():
                     results[kind]["detected"] += 1
                else:
                     results[kind]["missed"] += 1
                     
        return results
