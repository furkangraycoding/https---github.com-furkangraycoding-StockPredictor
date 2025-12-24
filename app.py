import streamlit as st
import pandas as pd
from data_loader import load_data
from indicators import TechnicalAnalyzer
from ml_engine import MLEngine
from ui_components import (
    render_header, 
    render_kpi_metrics, 
    render_plotly_chart, 
    render_ai_insights,
    render_scenario_simulator
)

# =====================================================
# CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="BIST100 AI Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# DATA LOADING
# =====================================================
FILE_PATH = "BIST100_PREDICTION_READY.csv"
df, DATE_COL = load_data(FILE_PATH)

if df.empty:
    st.error("Failed to load data. Please check the file path.")
    st.stop()

# =====================================================
# SIDEBAR CONTROLS
# =====================================================
st.sidebar.header("🛠 Configuration")

# Date Selection
min_date = df[DATE_COL].min().date()
max_date = df[DATE_COL].max().date()

selected_date = st.sidebar.date_input(
    "Analysis Date",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)
selected_date = pd.to_datetime(selected_date)

# Lookback Window
st.sidebar.subheader("Time Window")
months_lookback = st.sidebar.slider("Lookback (Months)", 3, 24, 6)

# =====================================================
# FEATURE ENGINEERING (ON FULL DATASET)
# =====================================================
# We calculate indicators on the WHOLE dataset first to ensure
# 1. Moving averages are valid even at the start of our "viewing window"
# 2. We have features ready for "Forward Testing" (Future data)
with st.spinner("Calculating Technical Indicators..."):
    # Run analysis on the full history
    analyzer = TechnicalAnalyzer(df)
    
    # Add Base Indicators (Safe to calculate globally)
    analyzer.add_moving_averages()
    analyzer.add_rsi()
    analyzer.add_atr()
    analyzer.determine_regime()
    # Note: ZigZag and Derived Features (State) must be calculated dynamically 
    # to avoid look-ahead bias in the simulation.
    
    df_with_basics = analyzer.get_df()

# =====================================================
# DATA SLICING (SIMULATE PAST)
# =====================================================
# For Training and "Current Status", we must pretend we are at 'selected_date'
mask_past = df_with_basics[DATE_COL] <= selected_date
df_analyzed = df_with_basics[mask_past].copy()

if df_analyzed.empty:
    st.error("Selected date is before the start of data. Please pick a later date.")
    st.stop()

# DYNAMIC ENRICHMENT (No Leakage)
# Re-run ZigZag and State logic only on the known history
with st.spinner("Simulating Real-Time State..."):
    sim_analyzer = TechnicalAnalyzer(df_analyzed)
    sim_analyzer.add_zigzag_labels(threshold_pct=0.05)
    sim_analyzer.add_rolling_volatility()
    sim_analyzer.add_drawdown_features()
    df_analyzed = sim_analyzer.add_derived_features()

# =====================================================
# MACHINE LEARNING ENGINE
# =====================================================
@st.cache_resource(hash_funcs={pd.DataFrame: lambda x: x.shape}) # Re-train if shape changes (new date)
def get_trained_model(data):
    # Train on available history
    engine = MLEngine(data)
    # Use True optimization for best results as requested, though it adds a delay
    metrics, backtest_df = engine.train(optimize=True) 
    return engine, metrics, backtest_df

with st.spinner("Training AI Prediction Models (Dips & Peaks)..."):
    ml_engine, metrics, backtest_df = get_trained_model(df_analyzed)
    print(f"FINAL_METRICS_LOG: {metrics}")
    
    # Predict for the CURRENT state (Last row)
    if not df_analyzed.empty:
        current_row = df_analyzed.iloc[-1]
        dip_prob, peak_prob = ml_engine.predict_probs(current_row)
    else:
        dip_prob, peak_prob = 0.0, 0.0

# Model Status (Read-only)
st.sidebar.divider()
st.sidebar.markdown(f"**🤖 AI Model Stats**")

# Calculate effective success rate (with 2% tolerance)
if not backtest_df.empty:
    eff_results = ml_engine.calculate_effective_success(backtest_df, tolerance_pct=2.0)
    
    dip_total = eff_results["dip"]["total"]
    dip_success = eff_results["dip"]["detected"] + eff_results["dip"]["near_hit"]
    dip_rate = (dip_success / dip_total * 100) if dip_total > 0 else 0
    
    peak_total = eff_results["peak"]["total"]
    peak_success = eff_results["peak"]["detected"] + eff_results["peak"]["near_hit"]
    peak_rate = (peak_success / peak_total * 100) if peak_total > 0 else 0
    
    st.sidebar.metric("Dip Başarı (%2 tolerans)", f"%{dip_rate:.0f}", 
                      delta=f"{dip_success}/{dip_total}")
    st.sidebar.metric("Peak Başarı (%2 tolerans)", f"%{peak_rate:.0f}",
                      delta=f"{peak_success}/{peak_total}")
else:
    st.sidebar.info("Backtest verisi yok.")


# =====================================================
# DATA WINDOWING & FUTURE PROJECTION (PAST + FUTURE)
# =====================================================
# Window: [selected_date - months_lookback] TO [selected_date + 7 days]
window_start = selected_date - pd.DateOffset(months=months_lookback)

# Base window from existing data (Simulated History)
df_window = df_analyzed[
    (df_analyzed[DATE_COL] >= window_start) & 
    (df_analyzed[DATE_COL] <= selected_date)
].copy()

# ENRICH HISTORY WITH AI PREDICTIONS (with Forward Confirmation)
# Forward confirmation: Önceki 4 gün + seçilen gün + sonraki 1-3 gün analizi
df_window = ml_engine.add_predictions_to_df(df_window, use_forward_confirmation=True)

# FUTURE PROJECTION: If we are at the end or want to see what's next
with st.spinner("Generating AI Future Forecast..."):
    # 1. Project price forward
    df_forecast = ml_engine.forecast_future(df_analyzed, days=7)
    
    if not df_forecast.empty:
        # 2. Combine with enough history to recalculate indicators (e.g., last 200 days)
        # We use df_analyzed as history
        df_combined = pd.concat([df_analyzed, df_forecast], ignore_index=True)
        
        # 3. Recalculate indicators on combined data
        forecast_analyzer = TechnicalAnalyzer(df_combined)
        forecast_analyzer.add_moving_averages()
        forecast_analyzer.add_rsi()
        forecast_analyzer.add_atr()
        forecast_analyzer.determine_regime()
        df_combined = forecast_analyzer.add_derived_features()
        
        # 4. Run AI Predictions on the projected portion (with Forward Confirmation)
        df_combined = ml_engine.add_predictions_to_df(df_combined, use_forward_confirmation=True)
        
        # 5. Extract only the forecast portion
        df_forecast_enriched = df_combined[df_combined["is_forecast"] == True].copy()
        
        # 6. FORCE SINGLE BEST SIGNAL (Quantitative Mode)
        # We want to show ONLY the most probable peak and dip in the forecast
        # Reset any threshold-based signals first
        df_forecast_enriched["AI_Dip"] = 0
        df_forecast_enriched["AI_Peak"] = 0
        
        # Pick the single highest probability dates
        if not df_forecast_enriched.empty:
            max_dip_idx = df_forecast_enriched["AI_Dip_Prob"].idxmax()
            max_peak_idx = df_forecast_enriched["AI_Peak_Prob"].idxmax()
            
            # Enforce minimal confidence just in case (e.g. > 0.05) to avoid noise if prob is 0.01
            if df_forecast_enriched.loc[max_dip_idx, "AI_Dip_Prob"] > 0.05:
                df_forecast_enriched.loc[max_dip_idx, "AI_Dip"] = 1
                
            if df_forecast_enriched.loc[max_peak_idx, "AI_Peak_Prob"] > 0.05:
                df_forecast_enriched.loc[max_peak_idx, "AI_Peak"] = 1
        
        # Update window for display
        df_window = pd.concat([df_window, df_forecast_enriched], ignore_index=True)


# =====================================================
# UI RENDERING (Simplified)
# =====================================================
render_header()

# CRITICAL: We separate HISTORY from FORECAST for current status analysis
if "is_forecast" in df_window.columns:
    # History rows will be NaN or False, Forecast rows will be True
    df_history_only = df_window[df_window["is_forecast"].fillna(False) == False]
else:
    df_history_only = df_window

# 2. Get probabilities for specifically the "Selected Date" (The present moment)
# This is for the Gauges and Insights
try:
    # Use the last row of df_history_only which corresponds to the selected_date
    target_row = df_history_only.iloc[-1]
    dip_prob, peak_prob = ml_engine.predict_probs(target_row)
except:
    dip_prob, peak_prob = 0.0, 0.0

# KPI Summary
render_kpi_metrics(df_history_only, dip_prob, peak_prob, 
                   metrics['dip'].get('accuracy', 0), 
                   metrics['peak'].get('accuracy', 0))

# Full-width Chart with AI Predictions
st.subheader("📊 BIST100 + AI Predictions")
render_plotly_chart(df_window, DATE_COL, selected_date=selected_date)

# AI Insights (The colored warning boxes)
render_ai_insights(df_history_only, dip_prob, peak_prob)

# =====================================================
# FUTURE PREDICTIONS LIST
# =====================================================
st.divider()
st.subheader("🔮 Gelecek Tahminleri")
st.caption(f"Model **{selected_date.strftime('%d/%m/%Y')}** tarihine kadar olan veriler ile eğitildi.")

# Get future data (after selected_date)
future_df = df_window[df_window[DATE_COL] > selected_date].copy()

if future_df.empty:
    st.info("Seçilen tarihten sonra veri bulunmuyor.")
else:
    # Find the SINGLE most probable Dip and Peak in the future
    # We already forced 'AI_Dip' and 'AI_Peak' to be 1 ONLY for the single best candidate in Step 6.
    # So we simply retrieve those rows.
    
    # Filter for the forced signals
    dip_rows = future_df[future_df["AI_Dip"] == 1]
    peak_rows = future_df[future_df["AI_Peak"] == 1]
    
    best_dip = dip_rows.iloc[0] if not dip_rows.empty else None
    best_peak = peak_rows.iloc[0] if not peak_rows.empty else None
    
    signals_list = []
    
    if best_dip is not None and best_dip["AI_Dip_Prob"] > 0.05: # Minimal threshold to show anything
        # Mark it on the dataframe for Chart visualization (if not already marked by threshold)
        # Note: df_window is already built, but we can ensure the chart matches the table.
        # But for the TABLE, we just add this row.
        signals_list.append({
            "Tarih": best_dip[DATE_COL].strftime("%d/%m/%Y"),
            "Sinyal": "🟢 TAHMİNİ DİP (En Güçlü)",
            "Fiyat": f"{best_dip['price']:,.2f}",
            "Olasılık": f"%{best_dip['AI_Dip_Prob']*100:.0f}"
        })
        
    if best_peak is not None and best_peak["AI_Peak_Prob"] > 0.05:
        signals_list.append({
            "Tarih": best_peak[DATE_COL].strftime("%d/%m/%Y"),
            "Sinyal": "🔴 POTANSİYEL ZİRVE (5 Gün İçinde Hedef)",
            "Fiyat": f"{best_peak['price']:,.2f}",
            "Olasılık": f"%{best_peak['AI_Peak_Prob']*100:.0f}"
        })
    if not signals_list:
        st.warning("Gelecek dönemde kayda değer bir sinyal bulunamadı.")
    else:
        # Sort by date
        signals_df = pd.DataFrame(signals_list).sort_values("Tarih")
        st.dataframe(signals_df, hide_index=True, use_container_width=True)
        
        # Construct summary string dynamically
        summary_parts = []
        if best_dip is not None:
             summary_parts.append(f"DIP: {best_dip[DATE_COL].strftime('%d/%m')} (Conf: %{best_dip['AI_Dip_Prob']*100:.0f})")
        if best_peak is not None:
             summary_parts.append(f"PEAK: {best_peak[DATE_COL].strftime('%d/%m')} (Conf: %{best_peak['AI_Peak_Prob']*100:.0f})")
             
        st.success(f"Gelecek 7 Gün İçin En Güçlü Tahminler:\n" + ", ".join(summary_parts))

# =====================================================
# MODEL INSIGHTS (Feature Importance & Accuracy)
# =====================================================
st.divider()
st.subheader("🧠 Model Insights & Accuracy")

importances = ml_engine.get_feature_importance()
tab1, tab2 = st.tabs(["Feature Importance", "Detailed Performance"])

with tab1:
    st.markdown("**Which features drive the model's decisions? (Feature Importance)**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Dip Model Top Features")
        top_dip = importances.sort_values("Dip_Importance", ascending=False).head(10)
        st.bar_chart(top_dip["Dip_Importance"], color="#2E8B57") # SeaGreen
        
    with col2:
        st.markdown("### Peak Model Top Features")
        top_peak = importances.sort_values("Peak_Importance", ascending=False).head(10)
        st.bar_chart(top_peak["Peak_Importance"], color="#CD5C5C") # IndianRed

with tab2:
    st.markdown("**Model Performance on Unseen Data (Cross-Validation Results)**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Dip Model")
        st.metric("Dip Model Accuracy", f"%{metrics['dip'].get('accuracy', 0)*100:.1f}")
        st.metric("Dip Precision", f"%{metrics['dip'].get('precision', 0)*100:.1f}")
        st.metric("Dip Recall", f"%{metrics['dip'].get('recall', 0)*100:.1f}")
        
    with col2:
        st.markdown("#### Peak Model")
        st.metric("Peak Model Accuracy", f"%{metrics['peak'].get('accuracy', 0)*100:.1f}")
        st.metric("Peak Precision", f"%{metrics['peak'].get('precision', 0)*100:.1f}")
        st.metric("Peak Recall", f"%{metrics['peak'].get('recall', 0)*100:.1f}")
