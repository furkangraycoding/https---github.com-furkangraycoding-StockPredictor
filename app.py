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

# Extract metrics for UI
dip_prec = metrics.get("dip", {}).get("precision", 0.0)
peak_prec = metrics.get("peak", {}).get("precision", 0.0)
dip_recall = metrics.get("dip", {}).get("recall", 0.0)
peak_recall = metrics.get("peak", {}).get("recall", 0.0)

st.sidebar.metric("Dip Precision", f"%{dip_prec*100:.0f}")
st.sidebar.metric("Peak Precision", f"%{peak_prec*100:.0f}")
st.sidebar.metric("Dip Recall", f"%{dip_recall*100:.0f}")
st.sidebar.metric("Peak Recall", f"%{peak_recall*100:.0f}")

# =====================================================
# DATA WINDOWING (EXTENDED WINDOW: PAST + FUTURE)
# =====================================================
# The user wants to see history AND future predictions on the SAME chart.
# Window: [selected_date - months_lookback] TO [selected_date + 30 days]
window_start = selected_date - pd.DateOffset(months=months_lookback)
window_end = selected_date + pd.DateOffset(days=30) # Peek 1 month into the future

df_window = df_full_enriched[
    (df_full_enriched[DATE_COL] >= window_start) & 
    (df_full_enriched[DATE_COL] <= window_end)
].copy()

# 1. Run inference on this entire window (optimized thresholds: Dip=0.50, Peak=0.60)
df_window = ml_engine.add_predictions_to_df(df_window)

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
            signals_list.append({
                "Tarih": row[DATE_COL].strftime("%d/%m/%Y"),
                "Sinyal": "🟢 DIP (AL)",
                "Fiyat": f"{row['price']:,.2f}",
                "Olasılık": f"%{row['AI_Dip_Prob']*100:.0f}"
            })
        
        for _, row in peak_signals.iterrows():
            signals_list.append({
                "Tarih": row[DATE_COL].strftime("%d/%m/%Y"),
                "Sinyal": "🔴 PEAK (SAT)",
                "Fiyat": f"{row['price']:,.2f}",
                "Olasılık": f"%{row['AI_Peak_Prob']*100:.0f}"
            })
        
        # Sort by date
        signals_df = pd.DataFrame(signals_list)
        st.dataframe(signals_df, hide_index=True, use_container_width=True)
        
        st.success(f"Toplam: {len(dip_signals)} DIP sinyali, {len(peak_signals)} PEAK sinyali")
