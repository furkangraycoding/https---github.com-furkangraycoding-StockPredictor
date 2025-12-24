import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def render_header():
    """Renders the main page header and description."""
    st.title("📈 BIST100 Karar Destek Sistemi")
    st.markdown(
        """
        **Yapay Zeka Destekli Piyasa Rejimi ve Trend Analizi**  
        Bu panel, kritik dönüş noktalarını, piyasa rejimini ve veriye dayalı öngörüleri sağlar.
        """
    )
    st.divider()

def render_kpi_metrics(df_window: pd.DataFrame, 
                       dip_prob: float = 0.0, peak_prob: float = 0.0,
                       dip_acc: float = 0.0, peak_acc: float = 0.0):
    """
    Renders the KPI summary strip with Predictive Analytics.
    """
    st.subheader("📌 Piyasa Durumu ve AI Öngörüleri")

    if df_window.empty:
        st.warning("Görüntülenecek geçmiş veri bulunamadı.")
        return

    # Safe column retrieval
    row = df_window.iloc[-1]
    last_price = row["price"]
    regime = row.get("Regime", "Unknown")
    rsi = row.get("RSI", row.get("rsi_14", 50))
    
    # Translate Regime
    regime_tr = {"Bullish": "Boğa (Yükseliş)", "Bearish": "Ayı (Düşüş)", "Recovery": "Toparlanma", "Warning": "Zayıflama"}.get(regime, regime)
    
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    kpi1.metric("Son Fiyat", f"{last_price:,.2f}")
    kpi2.metric("Piyasa Rejimi", regime_tr)
    kpi3.metric("RSI (14)", f"{rsi:.1f}")

    kpi4.metric(
        "🔮 Dip Olasılığı", 
        f"%{dip_prob*100:.1f}", 
        delta=f"Başarı: %{dip_acc*100:.0f}" if dip_prob > 0.5 else None,
        delta_color="normal"
    )

    kpi5.metric(
        "🔮 Zirve Olasılığı", 
        f"%{peak_prob*100:.1f}", 
        delta=f"Başarı: %{peak_acc*100:.0f}" if peak_prob > 0.5 else None,  
        delta_color="inverse"
    )
    
    neutral_prob = max(0.0, 1.0 - (dip_prob + peak_prob))
    kpi6.metric(
        "➡️ Trend Devamı",
        f"%{neutral_prob*100:.1f}",
        delta="Yatay/Trend"
    )
    
    st.divider()
    col_info, col_help = st.columns([4, 1])
    with col_help:
        st.caption("ℹ️ **Olasılık Hakkında Not**:\n'Dip' ve 'Zirve' dönüş sinyalleridir. Toplamları %100 etmeyebilir, çünkü piyasa genellikle **Trend Devamı** (Nötr) aşamasındadır.")
    
    with col_info:
        # NEW: Signal Purity Logic (User's Formula)
        # Gap = Peak% - Dip% 
        # Check if Gap > 48% of RSI
        # Peak Purity Logic
        gap = (peak_prob - dip_prob) * 100
        threshold = rsi * 0.48
        
        # Dip Purity Logic (Hybrid: 85%|1.1x OR 72%|1.3x)
        dip_gap = (dip_prob - peak_prob) * 100
        is_pure_dip = ((dip_prob >= 0.85 and dip_gap > rsi * 1.1) or 
                       (dip_prob >= 0.72 and dip_gap > rsi * 1.3))
        
        if is_pure_dip:
            st.success(f"🟢 **SAF DİP SİNYALİ**: Model %{dip_prob*100:.0f} güvenle tertemiz bir dip bölgesi öngörüyor. (Fark: {dip_gap:.1f})")
        elif dip_prob >= 0.72:
            st.info(f"⚪ **DİP DOYGUNLUĞU**: Dip olasılığı (%{dip_prob*100:.0f}) yüksek ancak saflık barajı geçilemedi.")
        elif dip_prob > 0.60:
            st.success(f"🟢 **DİP BÖLGESİ**: Model %{dip_prob*100:.0f} ihtimalle bir dip bölgesinde olduğumuzu öngörüyor.")
        elif peak_prob >= 0.85:
             if gap > threshold:
                 st.error(f"🔴 **SAF ZİRVE SİNYALİ**: Model %{peak_prob*100:.0f} güvenle tertemiz bir zirve sinyali üretiyor. (Fark: {gap:.1f} > Baraj: {threshold:.1f})")
             else:
                 st.info(f"⚪ **ZİRVE DOYGUNLUĞU**: Zirve olasılığı yüksek (%{peak_prob*100:.0f}) ancak Dip emaresi de yükseldiği için sinyal saflığını yitirdi. Baraj ({threshold:.1f}) geçilemedi.")
        elif peak_prob > 0.70:
            st.error(f"🟠 **OLASI ZİRVE**: Model %{peak_prob*100:.0f} ihtimalle zirve uyarısı veriyor.")
        else:
            st.info(f"👉 **TREND DEVAMI**: Piyasa şu anki yolunda devam etme eğiliminde.")

def render_plotly_chart(df_window: pd.DataFrame, date_col: str, selected_date: pd.Timestamp = None):
    """
    Chart showing: Price + Actual Dip/Peak + AI Predictions
    """
    fig = go.Figure()

    # Price Line
    fig.add_trace(go.Scatter(
        x=df_window[date_col], 
        y=df_window["price"],
        mode='lines',
        name='BIST100',
        line=dict(color='white', width=2)
    ))

    # Vertical line for Selected Date
    if selected_date:
        fig.add_vline(x=selected_date, line_width=2, line_dash="dash", line_color="yellow")
        fig.add_annotation(x=selected_date, y=1.02, yref="paper", text="📍", showarrow=False, font=dict(size=16))

    # -------------------------------------------------------------
    # ZIGZAG STRUCTURE (Connect Peaks and Dips)
    # -------------------------------------------------------------
    # Combine Tepe and Dip into a single series for the line
    zigzag_points = []
    if "Tepe" in df_window.columns and "Dip" in df_window.columns:
        # Extract potential points
        peaks = df_window.dropna(subset=["Tepe"])[["Tepe"]]
        peaks["type"] = "peak"
        peaks["val"] = peaks["Tepe"]
        
        dips = df_window.dropna(subset=["Dip"])[["Dip"]]
        dips["type"] = "dip"
        dips["val"] = dips["Dip"]
        
        # Combine and sort index
        zigzag_df = pd.concat([peaks, dips]).sort_index()
        
        if not zigzag_df.empty:
            fig.add_trace(go.Scatter(
                x=df_window.loc[zigzag_df.index, date_col], 
                y=zigzag_df["val"],
                mode='lines', 
                name='ZigZag Structure',
                line=dict(color='gray', width=1, dash='dash'),
                opacity=0.5
            ))

    # -------------------------------------------------------------
    # ACTUAL DIP/PEAK POINTS (Confirmed by ZigZag)
    # -------------------------------------------------------------
    # These are the "Official" retrospective labels
    if "Tepe" in df_window.columns:
        actual_peaks = df_window.dropna(subset=["Tepe"])
        if not actual_peaks.empty:
            fig.add_trace(go.Scatter(
                x=actual_peaks[date_col], y=actual_peaks["Tepe"],
                mode='markers', name='✅ Onaylı Zirve',
                marker=dict(color='red', symbol='triangle-down', size=12, 
                           line=dict(width=1, color='white'))
            ))
    
    if "Dip" in df_window.columns:
        actual_dips = df_window.dropna(subset=["Dip"])
        if not actual_dips.empty:
            fig.add_trace(go.Scatter(
                x=actual_dips[date_col], y=actual_dips["Dip"],
                mode='markers', name='✅ Onaylı Dip',
                marker=dict(color='lime', symbol='triangle-up', size=12,
                           line=dict(width=1, color='white'))
            ))

    # -------------------------------------------------------------
    # DETECTED TURNING POINTS (Based on State Change)
    # -------------------------------------------------------------
    # If official label isn't there yet, but State changed, mark it as "Detected"
    if "Last_Signal" in df_window.columns:
        # Find where state changes
        sig_change = df_window["Last_Signal"].diff()
        
        # State changed to 1 (Dip Detected)
        # The Dip itself was likely a few days ago (local min). 
        # But for visualization, let's mark the "Detection Day" or find the local min.
        dips_detected_mask = (sig_change == 2) | (sig_change == 1) # -1->1 or 0->1
        dips_detected = df_window[dips_detected_mask]
        
        if not dips_detected.empty:
             # Find actual local min in the 5 days prior to detection for better visual
             # This is purely cosmetic to place the marker on the candle wick
             real_dip_vals = []
             real_dip_dates = []
             for idx, row in dips_detected.iterrows():
                 # Look back 5 days
                 loc_idx = df_window.index.get_loc(idx)
                 start_loc = max(0, loc_idx - 5)
                 window_slice = df_window.iloc[start_loc:loc_idx+1]
                 min_row = window_slice.loc[window_slice["low"].idxmin()]
                 real_dip_vals.append(min_row["low"])
                 real_dip_dates.append(min_row[date_col])
                 
             fig.add_trace(go.Scatter(
                x=real_dip_dates, y=real_dip_vals,
                mode='markers', name='🔎 Tespit Edilen Dip',
                marker=dict(color='green', symbol='circle-open', size=10,
                           line=dict(width=2))
            ))

        # State changed to -1 (Peak Detected)
        peaks_detected_mask = (sig_change == -2) | (sig_change == -1) # 1->-1 or 0->-1
        peaks_detected = df_window[peaks_detected_mask]
        
        if not peaks_detected.empty:
             real_peak_vals = []
             real_peak_dates = []
             for idx, row in peaks_detected.iterrows():
                 loc_idx = df_window.index.get_loc(idx)
                 start_loc = max(0, loc_idx - 5)
                 window_slice = df_window.iloc[start_loc:loc_idx+1]
                 max_row = window_slice.loc[window_slice["high"].idxmax()]
                 real_peak_vals.append(max_row["high"])
                 real_peak_dates.append(max_row[date_col])

             fig.add_trace(go.Scatter(
                x=real_peak_dates, y=real_peak_vals,
                mode='markers', name='🔎 Tespit Edilen Zirve',
                marker=dict(color='red', symbol='circle-open', size=10,
                           line=dict(width=2))
            ))

    # -------------------------------------------------------------
    # AI PREDICTIONS
    # -------------------------------------------------------------
    if "AI_Dip" in df_window.columns:
        # Split into historical and forecast
        hist_dips = df_window[(df_window["AI_Dip"] == 1) & (df_window.get("is_forecast", False) == False)]
        fore_dips = df_window[(df_window["AI_Dip"] == 1) & (df_window.get("is_forecast", False) == True)]
        
        if not hist_dips.empty:
            fig.add_trace(go.Scatter(
                x=hist_dips[date_col], y=hist_dips["price"] * 0.97,
                mode='markers+text', name='🟢 AI BUY Sinyali',
                text=["BUY"] * len(hist_dips), textposition="bottom center",
                textfont=dict(size=9, color='lime'),
                marker=dict(color='lime', symbol='triangle-up', size=12)
            ))
            
        if not fore_dips.empty:
            fig.add_trace(go.Scatter(
                x=fore_dips[date_col], y=fore_dips["price"] * 0.97,
                mode='markers+text', name='🌵 AI BUY (Projeksiyon)',
                text=["PROJ"] * len(fore_dips), textposition="bottom center",
                textfont=dict(size=9, color='#00FF00'),
                marker=dict(color='#00FF00', symbol='triangle-up-open', size=12, line=dict(width=2))
            ))
        
    if "AI_Peak" in df_window.columns:
        hist_peaks = df_window[(df_window["AI_Peak"] == 1) & (df_window.get("is_forecast", False) == False)]
        fore_peaks = df_window[(df_window["AI_Peak"] == 1) & (df_window.get("is_forecast", False) == True)]
        
        if not hist_peaks.empty:
            fig.add_trace(go.Scatter(
                x=hist_peaks[date_col], y=hist_peaks["price"] * 1.03,
                mode='markers+text', name='🔴 AI SELL Sinyali',
                text=["SELL"] * len(hist_peaks), textposition="top center",
                textfont=dict(size=9, color='red'),
                marker=dict(color='red', symbol='triangle-down', size=12)
            ))
            
        if not fore_peaks.empty:
            fig.add_trace(go.Scatter(
                x=fore_peaks[date_col], y=fore_peaks["price"] * 1.03,
                mode='markers+text', name='🌵 AI SELL (Projeksiyon)',
                text=["PROJ"] * len(fore_peaks), textposition="top center",
                textfont=dict(size=9, color='#FF4500'),
                marker=dict(color='#FF4500', symbol='triangle-down-open', size=12, line=dict(width=2))
            ))

    fig.update_layout(
        template="plotly_dark",
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.03)'),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)



def render_ai_insights(df_window: pd.DataFrame, dip_prob: float = 0.0, peak_prob: float = 0.0):
    """
    Kritik uyarılar ve trend analizi.
    """
    st.subheader("🤖 AI Insight Engine")
    
    if df_window.empty:
        st.write("Analiz edilecek veri bulunamadı.")
        return
        
    last_row = df_window.iloc[-1]
    rsi = last_row.get("RSI", last_row.get("rsi_14", 50))
    regime = last_row.get("Regime", "Unknown")
    
    # Purity Check - PEAK (User's Formula: Gap > RSI * 0.48)
    gap = (peak_prob - dip_prob) * 100
    threshold = rsi * 0.48
    is_pure = (peak_prob >= 0.85) and (gap > threshold)

    # Purity Check - DIP (Hybrid Logic: 85%|1.1x OR 72%|1.3x)
    dip_gap = (dip_prob - peak_prob) * 100
    is_pure_dip = ((dip_prob >= 0.85 and dip_gap > rsi * 1.1) or 
                   (dip_prob >= 0.72 and dip_gap > rsi * 1.3))
    
    # Show CURRENT Pure Status as a Priority Box
    if is_pure:
        st.error(f"🔴 **SAF ZİRVE SİNYALİ TESPİT EDİLDİ**: Mevcut teknik veriler %{peak_prob*100:.0f} güvenle tertemiz bir zirveyi işaret ediyor. (Purity Gap: {gap:.1f} > Baraj: {threshold:.1f})")
    elif is_pure_dip:
        st.success(f"🟢 **SAF DİP SİNYALİ TESPİT EDİLDİ**: Mevcut teknik veriler %{dip_prob*100:.0f} güvenle tertemiz bir dibi işaret ediyor. (Fark: {dip_gap:.1f})")
    else:
        st.info("ℹ️ **PİYASA GÖRÜNÜMÜ**: Şu an için 'Saf' bir dönüş sinyali veya aşırı doygunluk emaresi saptanmadı. Piyasa mevcut trendini koruyor.")
    
    # Check Persistence (Last 5 Days)
    if "AI_Peak_Prob" in df_window.columns:
        last_5_days = df_window.tail(5)
        
        red_signals = (last_5_days["AI_Peak_Prob"] > 0.70).sum()
        yellow_signals = ((last_5_days["AI_Peak_Prob"] > 0.50) & 
                        (last_5_days["AI_Peak_Prob"] <= 0.70)).sum()
        
        green_signals = (last_5_days["AI_Dip_Prob"] > 0.70).sum()
        light_green_signals = ((last_5_days["AI_Dip_Prob"] > 0.50) & 
                            (last_5_days["AI_Dip_Prob"] <= 0.70)).sum()
        
        # Rule 1: 3/3 RED (Hyper-Critical)
        if len(last_5_days) >= 3:
            last_3_peak = last_5_days["AI_Peak_Prob"].iloc[-3:]
            if (last_3_peak > 0.70).all():
                if is_pure:
                    if peak_prob > 0.90 and dip_prob > 0.45:
                        st.error("🌪️ **PİYASA KLİMAKSI (Final Zirve)**: Zirve sinyali doygunluğa ulaştı ve model artık karşı yönde emareler görüyor. Dönüş an meselesi! (Güven: %97+)")
                    else:
                        st.error("🚨 **HİPER-KRİTİK ZİRVE UYARISI**: Model son 3 gündür ARALIKSIZ yüksek güvenli (%70+) sinyal üretiyor.")
                else:
                    st.info(f"🌫️ **SİNYAL DOYGUNLUĞU**: Israrlı sinyaller mevcut ancak saflık barajı ({threshold:.1f}) geçilemediği için alarm susturuldu.")
            
            # Rule 1-Dip: 3/3 GREEN (Hyper-Dip)
            last_3_dip = last_5_days["AI_Dip_Prob"].iloc[-3:]
            if (last_3_dip > 0.70).all():
                if is_pure_dip:
                    st.success(f"💎 **HİPER-KRİTİK DİP UYARISI**: Model son 3 gündür ARALIKSIZ tertemiz dip sinyali üretiyor.")
                else:
                    st.info(f"🌫️ **DİP DOYGUNLUĞU**: Israrlı dip sinyalleri var ancak saflık barajı geçilemedi.")
        
        # Rule 2: 2 Red + max 2 Yellow (Critical Peak)
        is_bullseye_peak = (last_5_days["AI_Peak_Prob"] > 0.85).any()
        if ((red_signals >= 2 and yellow_signals <= 2) or is_bullseye_peak) and is_pure:
            st.warning(f"🛑 **KRİTİK ZİRVE UYARISI**: Son 5 günde {red_signals} yüksek güvenli zirve sinyali algılandı.")

        # Rule 2-Dip: 2 Green + max 2 Light Green (Critical Dip)
        is_bullseye_dip = (last_5_days["AI_Dip_Prob"] > 0.85).any()
        if ((green_signals >= 2 and light_green_signals <= 2) or is_bullseye_dip) and is_pure_dip:
            st.success(f"🛡️ **GÜVENLİ ALIM BÖLGESİ**: Son 5 günde {green_signals} yüksek güvenli dip sinyali algılandı.")

        # Audit Log
        with st.expander("🔍 Sinyal Detayları (Son 5 Gün)"):
            for idx, row in last_5_days.iterrows():
                d_val = row.get("date", row.get("Date", None))
                d_str = d_val.strftime('%d.%m.%Y') if d_val else "Gün"
                p_peak = row["AI_Peak_Prob"]
                p_dip = row.get("AI_Dip_Prob", 0.0)
                
                if p_peak > 0.70:
                    msg = f"🔴 **ZİRVE** (%{p_peak*100:.0f}) | Dip: %{p_dip*100:.0f}"
                elif p_dip > 0.70:
                    msg = f"🟢 **DİP** (%{p_dip*100:.0f}) | Zirve: %{p_peak*100:.0f}"
                elif p_peak > 0.50:
                    msg = f"🟠 Olası Zirve (%{p_peak*100:.0f})"
                elif p_dip > 0.50:
                    msg = f"🔵 Olası Dip (%{p_dip*100:.0f})"
                else:
                    msg = f"⚪ Nötr (Z:%{p_peak*100:.0f} D:%{p_dip*100:.0f})"
                st.write(f"{d_str}: {msg}")

    # RSI Insights
    if rsi > 70:
        st.info(f"⚠️ **RSI Aşırı Alım ({rsi:.1f})**: Momentum çok şişti. Düzeltme riski yüksek.")
    elif rsi < 30:
        st.info(f"✅ **RSI Aşırı Satım ({rsi:.1f})**: Momentum çok düştü. Tepki alımı gelebilir.")
        
    # Regime Insights
    if regime == "Bullish":
        st.info("🐂 **Ana Trend: Yükseliş (Boğa)**. Düşüşler alım fırsatı olabilir.")
    elif regime == "Bearish":
        st.info("🐻 **Ana Trend: Düşüş (Ayı)**. Yükselişler satış fırsatı olabilir.")

def render_scenario_simulator(ml_engine, current_row: pd.Series):
    """
    AI-Powered what-if analysis. 
    Predicts how AI probabilities change based on simulated price moves.
    """
    st.subheader("🎲 AI Scenario Simulator")
    st.caption("What happens to AI sentiment if the price moves today?")
    
    current_price = current_row["price"]
    shock_pct = st.slider("Simulated Move (%)", -10.0, 10.0, 0.0, 0.5)
    
    sim_price = current_price * (1 + shock_pct/100)
    
    # Create a simulated row
    sim_row = current_row.copy()
    sim_row["price"] = sim_price
    # Update Dist_SMA if possible (simplified)
    if "SMA_50" in sim_row and sim_row["SMA_50"] != 0:
        sim_row["Dist_SMA50"] = (sim_price - sim_row["SMA_50"]) / sim_row["SMA_50"]
    
    # Re-predict using ML Engine
    with st.spinner("Calculating AI Response..."):
        sim_dip, sim_peak = ml_engine.predict_probs(sim_row)
    
    # Display Results
    c1, c2, c3 = st.columns(3)
    c1.metric("Simulated Price", f"{sim_price:,.2f}", f"{shock_pct}%")
    c2.metric("New Dip Prob", f"%{sim_dip*100:.1f}")
    c3.metric("New Peak Prob", f"%{sim_peak*100:.1f}")
    
    if sim_dip > 0.60:
        st.success("🎯 This move would trigger a STRONG BUY signal.")
    elif sim_peak > 0.60:
        st.warning("⚠️ This move would trigger a STRONG SELL signal.")
    else:
        st.info("Neutral: This move doesn't change the core trend prediction significantly.")

def render_backtest_chart(backtest_df: pd.DataFrame, date_col: str):
    """
    Renders a chart comparing Model Predictions vs Actual Extrema for the backtest period.
    """
    if backtest_df.empty:
        st.warning("No backtest data available.")
        return
        
    st.subheader("🔙 Visual Backtest (Prediction vs Reality)")
    st.markdown("This chart shows where the AI **would have traded** (Predictions) vs what actually happened (Reality) on unseen data.")
    
    fig = go.Figure()

    # Debug Info (Helper for user to see why it might be empty)
    total_preds = backtest_df["Pred_Dip"].sum() + backtest_df["Pred_Peak"].sum()
    if total_preds == 0:
        st.warning("⚠️ The AI has no 'High Confidence' trades in this specific period. Try lowering the 'Confidence Threshold' in the sidebar to [0.45 - 0.50].")
    
    # ACTUAL DIP (Green Triangle)
    actual_dips = backtest_df[backtest_df["Label_Dip"] == 1]
    fig.add_trace(go.Scatter(
        x=actual_dips[date_col], y=actual_dips["price"],
        mode='markers', name='Actual Dip (Labels)',
        marker=dict(color='lime', symbol='triangle-up', size=12, line=dict(width=2, color='white'))
    ))
    
    # PREDICTED DIP (Cyan Dot)
    # Check if we have Pred_Dip. If not, check if we have Prob_Dip to show 'Potential' signals
    pred_dips = backtest_df[backtest_df["Pred_Dip"] == 1]
    
    if pred_dips.empty and "Prob_Dip" in backtest_df.columns:
        # Fallback: Show everything above 50% as a "Low Confidence" ghost signal
        pred_dips = backtest_df[backtest_df["Prob_Dip"] > 0.45]
        name_dip = "AI Dip (Low Conf > 45%)"
    else:
        name_dip = "AI Predicted Dip"

    fig.add_trace(go.Scatter(
        x=pred_dips[date_col], y=pred_dips["price"] * 0.98,
        mode='markers', name=name_dip,
        marker=dict(color='cyan', symbol='circle', size=8, opacity=0.7)
    ))
    
    # ACTUAL PEAK (Red Triangle)
    actual_peaks = backtest_df[backtest_df["Label_Peak"] == 1]
    fig.add_trace(go.Scatter(
        x=actual_peaks[date_col], y=actual_peaks["price"],
        mode='markers', name='Actual Peak (Labels)',
        marker=dict(color='magenta', symbol='triangle-down', size=12, line=dict(width=2, color='white'))
    ))
    
    # PREDICTED PEAK (Orange Dot)
    pred_peaks = backtest_df[backtest_df["Pred_Peak"] == 1]
    
    if pred_peaks.empty and "Prob_Peak" in backtest_df.columns:
        pred_peaks = backtest_df[backtest_df["Prob_Peak"] > 0.45]
        name_peak = "AI Peak (Low Conf > 45%)"
    else:
        name_peak = "AI Predicted Peak"

    fig.add_trace(go.Scatter(
        x=pred_peaks[date_col], y=pred_peaks["price"] * 1.02,
        mode='markers', name=name_peak,
        marker=dict(color='orange', symbol='circle', size=8, opacity=0.7)
    ))

    fig.update_layout(
        template="plotly_dark",
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", y=1, x=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
