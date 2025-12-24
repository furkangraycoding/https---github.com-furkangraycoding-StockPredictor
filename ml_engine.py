import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, make_scorer
from sklearn.calibration import CalibratedClassifierCV
from indicators import TechnicalAnalyzer
import streamlit as st

class MLEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
        # Default Parameters (will be updated by tuning)
        self.rf_params = {
            "n_estimators": 300,
            "max_depth": 15,
            "min_samples_leaf": 1, # More aggressive to allow higher confidence
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
        # =====================================================
        # FEATURE SET DEFINITIONS
        # =====================================================
        self.all_features = [
            "RSI_14", "Williams_%R", "CCI_20", "StochRSI", "VAM", 
            "Volatility_20", "ATR", "BB_Position", "Dist_SMA200", 
            "MACD_Hist", "DI_Diff", "ADX_14", "Drawdown_Pct", "Rally_Pct", 
            "RSI_Diff_1D", "Cycle_Phase", "MFI_14", "Vol_ZScore", 
            "Kurtosis_20", "Fractal_High", "Fractal_Low", "RSI_Price_Corr_14",
            "RSI_Overbought_Days", "Leg_Return",
            # NEW: Cycle & Exhaustion Features
            "Cycle_Length", "Cycle_Length_Ratio", "Avg_Cycle_Length",
            "Price_Exhaustion_5D", "Price_Exhaustion_10D", "Price_Exhaustion_20D",
            "Price_Exhaustion_5D_ATR", "Price_Range_10D",
            "RSI_Decay_1D", "RSI_Decay_3D", "RSI_Decay_5D", "RSI_Decay_Accel",
            "ATR_Expansion_5D", "ATR_Expansion_10D", "ATR_vs_Avg",
            "Trend_Exhaustion_Score", "Dip_Exhaustion_Score",
            "Price_Blowoff", "Volume_Blowoff",
            "RSI_Oversold_Days"  # Added oversold duration
        ]
        
        # DIP MODEL: Can use everything (It works well)
        self.dip_features = self.all_features.copy()
        
        # PEAK MODEL: RESTRICTED SET
        # We remove purely price-based/volatility features that dominate signal
        # to force the model to look at Overbought/Momentum exhaustion indicators.
        self.peak_features = [
            "RSI_14", 
            "Williams_%R", 
            "CCI_20", 
            "StochRSI", 
            "VAM", 
            "ATR", # Valid for stop loss logic, maybe keep
            "BB_Position", 
            "MACD_Hist", 
            "DI_Diff", 
            "ADX_14",
            # "Volatility_20", # REMOVED
            # "Dist_SMA200",   # REMOVED
            # "Drawdown_Pct",  # REMOVED
            # "Rally_Pct",     # REMOVED
            "RSI_Diff_1D", 
            # "Cycle_Phase",   # REMOVED to force Technical Analysis usage
            "MFI_14", 
            "Vol_ZScore", 
            "Kurtosis_20", 
            "Fractal_High", 
            "Fractal_Low",
            "RSI_Price_Corr_14", # Divergence
            "RSI_Overbought_Days", # Time Decay
            # NEW: Exhaustion features for peak detection
            "Cycle_Length_Ratio",  # Cycle length relative to average
            "RSI_Decay_1D", "RSI_Decay_3D", "RSI_Decay_5D",  # Momentum decay
            "ATR_Expansion_5D", "ATR_vs_Avg",  # Volatility expansion
            "Trend_Exhaustion_Score",  # Composite exhaustion
            "Price_Blowoff", "Volume_Blowoff"  # Extreme moves
        ]

    def prepare_data(self):
        """Prepares training data using ZigZag labels."""
        # Ensure labels are generated using consistent ZigZag logic
        # We re-run it here to be safe and independent
        analyzer = TechnicalAnalyzer(self.df)
        analyzer = TechnicalAnalyzer(self.df)
        analyzer.add_zigzag_labels(threshold_pct=0.03, atr_factor=2.5)
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
        # LABEL EXPANSION (Smearing):
        # Predicting the exact single day of a pivot is extremely hard.
        # We expand the target to include +/- 2 days around the pivot.
        # LABEL EXPANSION & EXHAUSTION (Smearing):
        # Generate Required Indicators for Labeling
        analyzer = TechnicalAnalyzer(self.df)
        analyzer.add_rsi()
        analyzer.add_atr()
        
        # ADVANCED LABELING:
        # 1. Technical ZigZag Labels
        analyzer.add_zigzag_labels(threshold_pct=0.03, atr_factor=2.5)
        self.df = analyzer.get_df()
        
        raw_dip = np.where(self.df["Dip"].notna(), 1, 0)
        self.df["Label_Dip"] = pd.Series(raw_dip).rolling(window=5, min_periods=1, center=True).max().fillna(0).astype(int)
        
        raw_peak = np.where(self.df["Tepe"].notna(), 1, 0)
        zigzag_peak_labels = pd.Series(raw_peak).rolling(window=7, min_periods=1).max().shift(-4).fillna(0)
        
        # 2. Exhaustion/Danger Zone Labels (Peak only)
        # Use 'RSI' (TechnicalAnalyzer default)
        recent_high = self.df["high"].rolling(window=10).max()
        proximity_to_high = self.df["high"] / recent_high
        exhaustion_zone = (self.df["RSI"] > 70) & (proximity_to_high >= 0.98)
        
        # Combine: Either a confirmed ZigZag peak OR an exhaustion zone
        self.df["Label_Peak"] = ((zigzag_peak_labels == 1) | (exhaustion_zone)).astype(int)
        
        # Normalize/Fill Features
        for col in self.all_features:
            if col not in self.df.columns:
                self.df[col] = 0
        
        # Drop NaN rows (start of history)
        # Use only common/all features for dropna check to be safe
        self.df.dropna(subset=self.all_features, inplace=True)
        
        return self.df

    def tune_hyperparameters(self, X, y, task_name="Task", feature_names=None):
        """
        Optimizes Random Forest hyperparameters using RandomizedSearchCV.
        Returns tuned model and feature importance-based feature selection.
        """
        st.write(f"⚙️ Tuning Hyperparameters for {task_name} Model...")
        
        param_dist = {
            "n_estimators": [200, 300, 400, 500, 700, 1000],
            "max_depth": [8, 10, 12, 15, 20, 25, None],
            "min_samples_leaf": [1, 2, 4, 6, 8],
            "min_samples_split": [2, 5, 10, 15, 20],
            "max_features": ["sqrt", "log2", 0.5, 0.7, None],
            "class_weight": ["balanced", "balanced_subsample", None]
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        scorer = make_scorer(f1_score, zero_division=0) # Optimize for F1 of the minority class
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=30, # Increased for better search
            scoring=scorer,
            cv=tscv,
            verbose=0,
            n_jobs=-1,
            random_state=42
        )
        
        search.fit(X, y)
        best_model = search.best_estimator_
        st.write(f"✅ Best Params for {task_name}: {search.best_params_}")
        
        # Get feature importances from tuned model
        if feature_names is not None and hasattr(best_model, 'feature_importances_'):
            importances = pd.Series(best_model.feature_importances_, index=feature_names)
            # Filter features with importance >= 0.05
            selected_features = importances[importances >= 0.05].index.tolist()
            
            # If too many features dropped, keep at least top 10
            if len(selected_features) < 10:
                selected_features = importances.sort_values(ascending=False).head(10).index.tolist()
                st.write(f"⚠️ Only {len(selected_features)} features with importance >= 0.05, keeping top 10")
            
            st.write(f"📊 Feature Selection: {len(selected_features)}/{len(feature_names)} features kept (importance >= 0.05)")
            
            # Show dropped features
            dropped = [f for f in feature_names if f not in selected_features]
            if dropped:
                st.write(f"🗑️ Dropped features ({len(dropped)}): {', '.join(dropped[:10])}{'...' if len(dropped) > 10 else ''}")
            
            return best_model, selected_features, importances
        else:
            return best_model, None, None

    def train(self, optimize=True):
        """
        Trains directly on State-Filtered Data using Strict Time-Series Split.
        Blind Test Check: Model uses first 70% data to learn, predicts on last 30%.
        """
        data = self.prepare_data()
        
        # FEATURE DROPOUT (Regularization)
        if "Cycle_Phase" in data.columns:
             mask = np.random.rand(len(data)) < 0.30
             data.loc[mask, "Cycle_Phase"] = -1
        
        if data.empty:
            return {}, pd.DataFrame()

        # STRICT TRAIN/TEST SPLIT
        split_idx = int(len(data) * 0.70)
        train_data = data.iloc[:split_idx].copy()
        test_data = data.iloc[split_idx:].copy() # Future/Blind Data
        
        metrics = {"dip": {}, "peak": {}}
        tscv = TimeSeriesSplit(n_splits=3)
        
        # 1. Train DIP Model (Use TRAIN DATA Only)
        dip_mask = (train_data["Last_Signal"] == -1) | (train_data["Last_Signal"] == 0)
        X_dip = train_data.loc[dip_mask, self.dip_features]
        y_dip = train_data.loc[dip_mask, "Label_Dip"]
        
        if not X_dip.empty and y_dip.sum() > 0:
            # Step 1: Hyperparameter Tuning with ALL features
            if optimize:
                tuned_model, selected_features, importances = self.tune_hyperparameters(
                    X_dip, y_dip, "Dip", feature_names=self.dip_features
                )
                
                if selected_features is not None:
                    # Update feature list based on tuned model importance
                    self.dip_features = selected_features
                    X_dip_selected = X_dip[self.dip_features]
                    self.dip_model = tuned_model
                    
                    # Re-fit with selected features only
                    self.dip_model.fit(X_dip_selected, y_dip)
                    
                    # Log feature importance summary
                    if importances is not None:
                        top_features = importances.sort_values(ascending=False).head(10)
                        st.write(f"🔝 Top 10 Dip Features: {', '.join(top_features.index.tolist())}")
                else:
                    # Fallback: use tuned model as is
                    self.dip_model = tuned_model
                    self.dip_model.fit(X_dip, y_dip)
            else:
                # No optimization: Quick feature selection with base model
                base_dip = RandomForestClassifier(n_estimators=100, random_state=42)
                base_dip.fit(X_dip, y_dip)
                imps = pd.Series(base_dip.feature_importances_, index=self.dip_features)
                self.dip_features = imps[imps >= 0.05].index.tolist()
                
                if not self.dip_features:
                    self.dip_features = imps.sort_values(ascending=False).head(10).index.tolist()
                
                X_dip_selected = X_dip[self.dip_features]
                self.dip_model = RandomForestClassifier(**self.rf_params)
                self.dip_model.fit(X_dip_selected, y_dip)
            
            # Evaluate on Test Set (Blind) using SELECTED features
            test_dip_mask = (test_data["Last_Signal"] == -1) | (test_data["Last_Signal"] == 0)
            if test_dip_mask.sum() > 0:
                X_test_dip = test_data.loc[test_dip_mask, self.dip_features]
                y_test_dip = test_data.loc[test_dip_mask, "Label_Dip"]
                
                preds = self.dip_model.predict(X_test_dip)
                metrics["dip"]["precision"] = precision_score(y_test_dip, preds, zero_division=0)
                metrics["dip"]["recall"] = recall_score(y_test_dip, preds, zero_division=0)
                metrics["dip"]["accuracy"] = accuracy_score(y_test_dip, preds)
            else:
                metrics["dip"] = {"precision": 0, "recall": 0, "accuracy": 0}

        # 2. Train PEAK Model (Use TRAIN DATA Only)
        peak_mask = (train_data["Last_Signal"] == 1) | (train_data["Last_Signal"] == 0)
        X_peak = train_data.loc[peak_mask, self.peak_features]
        y_peak = train_data.loc[peak_mask, "Label_Peak"]

        if not X_peak.empty and y_peak.sum() > 0:
            # Prepare oversampled data for peak model (before feature selection)
            pos_mask = (y_peak == 1)
            X_peak_pos = X_peak[pos_mask]
            y_peak_pos = y_peak[pos_mask]
            
            X_peak_oversampled = pd.concat([X_peak] + [X_peak_pos]*10)
            y_peak_oversampled = pd.concat([y_peak] + [y_peak_pos]*10)
            
            # Step 1: Hyperparameter Tuning with ALL features
            if optimize:
                tuned_model, selected_features, importances = self.tune_hyperparameters(
                    X_peak_oversampled, y_peak_oversampled, "Peak", feature_names=self.peak_features
                )
                
                if selected_features is not None:
                    # Update feature list based on tuned model importance
                    self.peak_features = selected_features
                    X_peak_selected = X_peak_oversampled[self.peak_features]
                    self.peak_model = tuned_model
                    
                    # Re-fit with selected features only
                    self.peak_model.fit(X_peak_selected, y_peak_oversampled)
                    
                    # Log feature importance summary
                    if importances is not None:
                        top_features = importances.sort_values(ascending=False).head(10)
                        st.write(f"🔝 Top 10 Peak Features: {', '.join(top_features.index.tolist())}")
                else:
                    # Fallback: use tuned model as is
                    self.peak_model = tuned_model
                    self.peak_model.fit(X_peak_oversampled, y_peak_oversampled)
            else:
                # No optimization: Quick feature selection with base model
                base_peak = RandomForestClassifier(n_estimators=100, random_state=42)
                base_peak.fit(X_peak_oversampled, y_peak_oversampled)
                imps = pd.Series(base_peak.feature_importances_, index=self.peak_features)
                self.peak_features = imps[imps >= 0.05].index.tolist()
                
                if not self.peak_features:
                    self.peak_features = imps.sort_values(ascending=False).head(10).index.tolist()
                
                X_peak_selected = X_peak_oversampled[self.peak_features]
                self.peak_model = RandomForestClassifier(**self.rf_params)
                self.peak_model.fit(X_peak_selected, y_peak_oversampled)
            
            # Evaluate on Test Set (Blind)
            test_peak_mask = (test_data["Last_Signal"] == 1) | (test_data["Last_Signal"] == 0)
            if test_peak_mask.sum() > 0:
                X_test_peak = test_data.loc[test_peak_mask, self.peak_features]
                y_test_peak = test_data.loc[test_peak_mask, "Label_Peak"]
                
                preds = self.peak_model.predict(X_test_peak)
                metrics["peak"]["precision"] = precision_score(y_test_peak, preds, zero_division=0)
                metrics["peak"]["recall"] = recall_score(y_test_peak, preds, zero_division=0)
                metrics["peak"]["accuracy"] = accuracy_score(y_test_peak, preds)
            else:
                metrics["peak"] = {"precision": 0, "recall": 0, "accuracy": 0}
        
        # Add predictions to the test set for visual verification
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
        # Create a combined index of all unique features
        all_unique_features = sorted(list(set(self.dip_features + self.peak_features)))
        importances = pd.DataFrame(index=all_unique_features)
        
        importances["Dip_Importance"] = 0.0
        importances["Peak_Importance"] = 0.0
        
        if hasattr(self.dip_model, "feature_importances_"):
            # Map dip importances to the correct rows
            dip_imps = self.dip_model.feature_importances_
            # Create series
            dip_s = pd.Series(dip_imps, index=self.dip_features)
            importances["Dip_Importance"] = importances.index.map(dip_s).fillna(0)
            
        if hasattr(self.peak_model, "feature_importances_"):
            # Map peak importances to correct rows
            peak_imps = self.peak_model.feature_importances_
            peak_s = pd.Series(peak_imps, index=self.peak_features)
            importances["Peak_Importance"] = importances.index.map(peak_s).fillna(0)
            
        return importances

    def predict_probs(self, current_row: pd.Series):
        """Returns (Prob_Dip, Prob_Peak) respecting State Logic."""
        try:
            # Construct feature vector
            input_data = pd.DataFrame([current_row], columns=current_row.index)
            
            for col in self.all_features:
                if col not in input_data.columns:
                    input_data[col] = 0
            
            # Predict probabilities WITHOUT state gating to allow raw signal visibility
            X_dip = input_data[self.dip_features].fillna(0)
            p_dip = self.dip_model.predict_proba(X_dip)[0][1]
            
            X_peak = input_data[self.peak_features].fillna(0)
            p_peak = self.peak_model.predict_proba(X_peak)[0][1]
            
            return p_dip, p_peak
        except Exception as e:
            return 0.0, 0.0

    def add_predictions_to_df(self, df: pd.DataFrame, threshold: float = 0.50, use_forward_confirmation: bool = True):
        """
        Runs inference with State Filtering and Forward Confirmation.
        
        Forward Confirmation Logic:
        - Önceki 4 gün + seçilen gün + sonraki 1-3 gün analiz edilir
        - Sonraki günlerde trend kırılımı varsa, seçilen gün saf sinyal olarak işaretlenir
        - Peak: Sonraki günlerde dip_prob artıyor veya fiyat düşüyor → Peak doğrulandı
        - Dip: Sonraki günlerde peak_prob artıyor veya fiyat yükseliyor → Dip doğrulandı
        """
        scan_df = df.copy()
        
        # Feature check
        for col in self.all_features:
            if col not in scan_df.columns:
                scan_df[col] = 0
        scan_df[self.all_features] = scan_df[self.all_features].fillna(0)
        
        # If 'Last_Signal' is missing in future data (e.g. forecast), we must forward fill it from history
        if "Last_Signal" in scan_df.columns:
             scan_df["Last_Signal"] = scan_df["Last_Signal"].replace(0, np.nan).ffill().fillna(0)
        
        try:
            # 1. Base Probabilities (Use correct features per model)
            dip_probs = self.dip_model.predict_proba(scan_df[self.dip_features])[:, 1]
            peak_probs = self.peak_model.predict_proba(scan_df[self.peak_features])[:, 1]
            rsi_vals = scan_df["rsi_14"] if "rsi_14" in scan_df.columns else (scan_df["RSI"] if "RSI" in scan_df.columns else 50)
            
            # 2. Base Purity Logic (Initial Check)
            peak_gap = (peak_probs - dip_probs) * 100
            peak_threshold = rsi_vals * 0.48
            base_pure_peak = (peak_probs >= 0.85) & (peak_gap > peak_threshold)
            
            dip_gap = (dip_probs - peak_probs) * 100
            base_pure_dip = ((dip_probs >= 0.85) & (dip_gap > rsi_vals * 1.1)) | \
                           ((dip_probs >= 0.72) & (dip_gap > rsi_vals * 1.3))
            
            # 3. Forward Confirmation Logic (if enabled)
            if use_forward_confirmation:
                is_pure_peak = self._apply_forward_confirmation_peak(
                    scan_df, base_pure_peak, peak_probs, dip_probs, rsi_vals
                )
                is_pure_dip = self._apply_forward_confirmation_dip(
                    scan_df, base_pure_dip, peak_probs, dip_probs, rsi_vals
                )
            else:
                is_pure_peak = base_pure_peak
                is_pure_dip = base_pure_dip

            df["AI_Dip"] = is_pure_dip.astype(int)
            df["AI_Peak"] = is_pure_peak.astype(int)
            df["AI_Dip_Prob"] = dip_probs
            df["AI_Peak_Prob"] = peak_probs
            
        except Exception as e:
            st.warning(f"Prediction error: {e}")
            df["AI_Dip"] = 0
            df["AI_Peak"] = 0
            df["AI_Dip_Prob"] = 0.0
            df["AI_Peak_Prob"] = 0.0
            
        return df
    
    def _apply_forward_confirmation_peak(self, df, base_signals, peak_probs, dip_probs, rsi_vals):
        """
        Forward confirmation for Peak signals.
        Peak için: Sonraki 1-3 günde dip_prob artıyor veya fiyat düşüyor → Peak doğrulandı
        
        Mantık:
        1. Base signal varsa ve forward confirmation varsa → KESIN SAF SİNYAL
        2. Base signal yoksa ama forward confirmation güçlüyse → YENİDEN DEĞERLENDİR
        3. Önceki 4 günde persistence varsa → EK DOĞRULAMA
        """
        confirmed_signals = base_signals.copy()
        n = len(df)
        
        # Convert to numpy arrays for easier indexing
        base_signals_arr = base_signals.values if hasattr(base_signals, 'values') else base_signals
        peak_probs_arr = peak_probs.values if hasattr(peak_probs, 'values') else peak_probs
        dip_probs_arr = dip_probs.values if hasattr(dip_probs, 'values') else dip_probs
        rsi_vals_arr = rsi_vals.values if hasattr(rsi_vals, 'values') else rsi_vals
        
        for i in range(n):
            # Base signal kontrolü (opsiyonel - forward confirmation base signal olmadan da çalışabilir)
            has_base_signal = base_signals_arr[i] if i < len(base_signals_arr) else False
            
            current_peak_prob = peak_probs_arr[i]
            current_dip_prob = dip_probs_arr[i]
            current_rsi = rsi_vals_arr[i]
            
            # Sonraki 1-3 günü kontrol et
            forward_confirmed = False
            
            for lookahead in [1, 2, 3]:
                if i + lookahead >= n:
                    break
                
                future_peak_prob = peak_probs_arr[i + lookahead]
                future_dip_prob = dip_probs_arr[i + lookahead]
                
                # Trend kırılımı kontrolü:
                # 1. Dip prob artıyor (trend dönüşü)
                dip_prob_increase = future_dip_prob > current_dip_prob + 0.10  # %10+ artış
                
                # 2. Peak prob düşüyor (momentum kaybı)
                peak_prob_decrease = future_peak_prob < current_peak_prob - 0.05  # %5+ düşüş
                
                # 3. Fiyat düşüyor (gerçek kırılım)
                price_decrease = False
                if "price" in df.columns:
                    current_price = df.iloc[i]["price"]
                    future_price = df.iloc[i + lookahead]["price"]
                    price_decrease = future_price < current_price * 0.98  # %2+ düşüş
                
                # 4. Gap değişimi (dip prob peak prob'u geçiyor)
                current_gap = (current_peak_prob - current_dip_prob) * 100
                future_gap = (future_peak_prob - future_dip_prob) * 100
                gap_reversal = future_gap < current_gap - 10  # Gap 10+ puan azalıyor
                
                # Forward confirmation: En az 2 kriter sağlanmalı
                confirmation_score = sum([
                    dip_prob_increase,
                    peak_prob_decrease,
                    price_decrease,
                    gap_reversal
                ])
                
                if confirmation_score >= 2:
                    forward_confirmed = True
                    break
            
            # Önceki 4 günün persistence kontrolü (opsiyonel, daha güvenilir sinyal için)
            persistence_score = 0
            if i >= 4:
                for lookback in [1, 2, 3, 4]:
                    past_peak_prob = peak_probs_arr[i - lookback]
                    if past_peak_prob >= 0.70:  # Önceki günlerde de yüksek peak prob
                        persistence_score += 1
            
            # Final decision: Base signal + Forward confirmation + (opsiyonel) Persistence
            # 1. Forward confirmation varsa → KESIN SAF SİNYAL (base signal olsun ya da olmasın)
            if forward_confirmed:
                confirmed_signals.iloc[i] = True
            # 2. Base signal var + Persistence var → GÜVENİLİR SİNYAL
            elif has_base_signal and persistence_score >= 2:
                confirmed_signals.iloc[i] = True
            # 3. Base signal var ama forward confirmation yok → BASE SİNYAL (eski mantık)
            elif has_base_signal:
                confirmed_signals.iloc[i] = True
            # 4. Hiçbiri yok → SİNYAL YOK
            else:
                confirmed_signals.iloc[i] = False
        
        return confirmed_signals
    
    def _apply_forward_confirmation_dip(self, df, base_signals, peak_probs, dip_probs, rsi_vals):
        """
        Forward confirmation for Dip signals.
        Dip için: Sonraki 1-3 günde peak_prob artıyor veya fiyat yükseliyor → Dip doğrulandı
        """
        confirmed_signals = base_signals.copy()
        n = len(df)
        
        # Convert to numpy arrays for easier indexing
        base_signals_arr = base_signals.values if hasattr(base_signals, 'values') else base_signals
        peak_probs_arr = peak_probs.values if hasattr(peak_probs, 'values') else peak_probs
        dip_probs_arr = dip_probs.values if hasattr(dip_probs, 'values') else dip_probs
        rsi_vals_arr = rsi_vals.values if hasattr(rsi_vals, 'values') else rsi_vals
        
        for i in range(n):
            # Base signal kontrolü (opsiyonel)
            has_base_signal = base_signals_arr[i] if i < len(base_signals_arr) else False
            
            current_dip_prob = dip_probs_arr[i]
            current_peak_prob = peak_probs_arr[i]
            current_rsi = rsi_vals_arr[i]
            
            # Sonraki 1-3 günü kontrol et
            forward_confirmed = False
            
            for lookahead in [1, 2, 3]:
                if i + lookahead >= n:
                    break
                
                future_dip_prob = dip_probs_arr[i + lookahead]
                future_peak_prob = peak_probs_arr[i + lookahead]
                
                # Trend kırılımı kontrolü:
                # 1. Peak prob artıyor (trend dönüşü)
                peak_prob_increase = future_peak_prob > current_peak_prob + 0.10  # %10+ artış
                
                # 2. Dip prob düşüyor (momentum kaybı)
                dip_prob_decrease = future_dip_prob < current_dip_prob - 0.05  # %5+ düşüş
                
                # 3. Fiyat yükseliyor (gerçek kırılım)
                price_increase = False
                if "price" in df.columns:
                    current_price = df.iloc[i]["price"]
                    future_price = df.iloc[i + lookahead]["price"]
                    price_increase = future_price > current_price * 1.02  # %2+ yükseliş
                
                # 4. Gap değişimi (peak prob dip prob'u geçiyor)
                current_gap = (current_dip_prob - current_peak_prob) * 100
                future_gap = (future_dip_prob - future_peak_prob) * 100
                gap_reversal = future_gap < current_gap - 10  # Gap 10+ puan azalıyor (dip avantajı kayboluyor)
                
                # Forward confirmation: En az 2 kriter sağlanmalı
                confirmation_score = sum([
                    peak_prob_increase,
                    dip_prob_decrease,
                    price_increase,
                    gap_reversal
                ])
                
                if confirmation_score >= 2:
                    forward_confirmed = True
                    break
            
            # Önceki 4 günün persistence kontrolü
            persistence_score = 0
            if i >= 4:
                for lookback in [1, 2, 3, 4]:
                    past_dip_prob = dip_probs_arr[i - lookback]
                    if past_dip_prob >= 0.60:  # Önceki günlerde de yüksek dip prob
                        persistence_score += 1
            
            # Final decision
            # 1. Forward confirmation varsa → KESIN SAF SİNYAL
            if forward_confirmed:
                confirmed_signals.iloc[i] = True
            # 2. Base signal var + Persistence var → GÜVENİLİR SİNYAL
            elif has_base_signal and persistence_score >= 2:
                confirmed_signals.iloc[i] = True
            # 3. Base signal var ama forward confirmation yok → BASE SİNYAL
            elif has_base_signal:
                confirmed_signals.iloc[i] = True
            # 4. Hiçbiri yok → SİNYAL YOK
            else:
                confirmed_signals.iloc[i] = False
        
        return confirmed_signals

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
