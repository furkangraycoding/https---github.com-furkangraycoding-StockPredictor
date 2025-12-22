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
# CONFIG & SETUP
# =====================================================
st.set_page_config(
    page_title="BIST100 Decision Support",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# DATA LOADING
# =====================================================
FILE_PATH = "BIST100_ALL_WITH_STRATEGY_AND_ML.csv"
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
    
    # Add Base Indicators
    analyzer.add_moving_averages()
    analyzer.add_rsi()
    analyzer.add_atr()
    
    # Add Advanced Features
    analyzer.determine_regime()
    analyzer.add_local_extrema() # Historical Labels
    df_full_enriched = analyzer.add_derived_features()

# =====================================================
# DATA SLICING (SIMULATE PAST)
# =====================================================
# For Training and "Current Status", we must pretend we are at 'selected_date'
# We mask out the future observations.
mask_past = df_full_enriched[DATE_COL] <= selected_date
df_analyzed = df_full_enriched[mask_past].copy()

if df_analyzed.empty:
    st.error("Selected date is before the start of data. Please pick a later date.")
    st.stop()

# =====================================================
# MACHINE LEARNING ENGINE
# =====================================================
@st.cache_resource
@st.cache_resource
def get_trained_model(data):
    # Train on available history
    engine = MLEngine(data)
    metrics, backtest_df = engine.train()
    return engine, metrics, backtest_df

with st.spinner("Training AI Prediction Models (Dips & Peaks)..."):
    ml_engine, metrics, backtest_df = get_trained_model(df_analyzed)
    
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
# Window: [selected_date - months_lookback] TO [selected_date + 30 days]
window_start = selected_date - pd.DateOffset(months=months_lookback)

# Base window from existing data
df_window = df_full_enriched[
    (df_full_enriched[DATE_COL] >= window_start) & 
    (df_full_enriched[DATE_COL] <= selected_date)
].copy()

# FUTURE PROJECTION: If we are at the end or want to see what's next
with st.spinner("Generating AI Future Forecast..."):
    # 1. Project price forward
    df_forecast = ml_engine.forecast_future(df_analyzed, days=30)
    
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
        
        # 4. Run AI Predictions on the projected portion
        df_combined = ml_engine.add_predictions_to_df(df_combined)
        
        # 5. Extract only the forecast portion
        df_forecast_enriched = df_combined[df_combined["is_forecast"] == True].copy()
        
        # 6. FORCE AT LEAST ONE SIGNAL (as requested by user)
        # If no dip or no peak found in the 30-day forecast, pick the highest probability dates
        if (df_forecast_enriched["AI_Dip"] == 0).all():
            max_dip_idx = df_forecast_enriched["AI_Dip_Prob"].idxmax()
            df_forecast_enriched.loc[max_dip_idx, "AI_Dip"] = 1
            
        if (df_forecast_enriched["AI_Peak"] == 0).all():
            max_peak_idx = df_forecast_enriched["AI_Peak_Prob"].idxmax()
            df_forecast_enriched.loc[max_peak_idx, "AI_Peak"] = 1
        
        # Update window for display
        df_window = pd.concat([df_window, df_forecast_enriched], ignore_index=True)

# 2. Get probabilities for specifically the "Selected Date" (The present moment)
# This is for the Gauges and Insights
try:
    target_row = df_analyzed.iloc[-1]
    dip_prob, peak_prob = ml_engine.predict_probs(target_row)
except:
    dip_prob, peak_prob = 0.0, 0.0

# =====================================================
# UI RENDERING (Simplified)
# =====================================================
render_header()

# Full-width Chart with AI Predictions
st.subheader("📊 BIST100 + AI Predictions")
render_plotly_chart(df_window, DATE_COL, selected_date=selected_date)

# Current Status (Simple)
if dip_prob > 0.50:
    st.success(f"🟢 **AI BUY Signal Today**: Dip Probability = %{dip_prob*100:.0f}")
elif peak_prob > 0.50:
    st.error(f"🔴 **AI SELL Signal Today**: Peak Probability = %{peak_prob*100:.0f}")
else:
    st.info(f"⚪ **No Strong Signal**: Dip=%{dip_prob*100:.0f}, Peak=%{peak_prob*100:.0f}")

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
    # Filter only rows with signals
    dip_signals = future_df[future_df["AI_Dip"] == 1]
    peak_signals = future_df[future_df["AI_Peak"] == 1]
    
    if dip_signals.empty and peak_signals.empty:
        st.warning("Gelecek dönemde güçlü bir sinyal yok.")
    else:
        # Display signals in a table
        signals_list = []
        
        for _, row in dip_signals.iterrows():
            is_forecast = row.get("is_forecast", False)
            signals_list.append({
                "Tarih": row[DATE_COL].strftime("%d/%m/%Y"),
                "Sinyal": "🟢 DIP (AL)" + (" (Tahmin)" if is_forecast else ""),
                "Fiyat": f"{row['price']:,.2f}",
                "Olasılık": f"%{row['AI_Dip_Prob']*100:.0f}"
            })
        
        for _, row in peak_signals.iterrows():
            is_forecast = row.get("is_forecast", False)
            signals_list.append({
                "Tarih": row[DATE_COL].strftime("%d/%m/%Y"),
                "Sinyal": "🔴 PEAK (SAT)" + (" (Tahmin)" if is_forecast else ""),
                "Fiyat": f"{row['price']:,.2f}",
                "Olasılık": f"%{row['AI_Peak_Prob']*100:.0f}"
            })
        
        # Sort by date
        signals_df = pd.DataFrame(signals_list)
        st.dataframe(signals_df, hide_index=True, use_container_width=True)
        
        st.success(f"Toplam: {len(dip_signals)} DIP sinyali, {len(peak_signals)} PEAK sinyali")
