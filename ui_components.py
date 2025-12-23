import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def render_header():
    """Renders the main page header and description."""
    st.title("📈 BIST100 Decision Support (Professional)")
    st.markdown(
        """
        **AI-Powered Market Regime & Trend Analysis**  
        This dashboard identifies pivotal market levels, regime states, and provides actionable data-driven insights.
        """
    )
    st.divider()

def render_kpi_metrics(df_window: pd.DataFrame, 
                       dip_prob: float = 0.0, peak_prob: float = 0.0,
                       dip_acc: float = 0.0, peak_acc: float = 0.0):
    """
    Renders the KPI summary strip with Predictive Analytics.
    """
    st.subheader("📌 Market Position & AI Predictions")

    if "price" not in df_window.columns:
        st.warning("Price data missing for KPI calculation.")
        return

    last_price = df_window.iloc[-1]["price"]
    regime = df_window.iloc[-1].get("Regime", "Unknown")
    rsi = df_window.iloc[-1].get("RSI", 50)
    
    # Layout: Price | Regime | RSI | Dip Prediction | Peak Prediction | Trend Prediction
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    
    # 1. Price
    kpi1.metric("Last Price", f"{last_price:,.2f}")
    
    # 2. Regime
    kpi2.metric("Regime", regime)

    # 3. RSI
    kpi3.metric("RSI (14)", f"{rsi:.1f}")

    # 4. DIP Prediction
    # Showing Probability AND Model Precision (Accuracy confidence)
    dip_delta = "High Conf." if dip_acc > 0.6 else "Mod. Conf."
    kpi4.metric(
        "🔮 Dip Probability", 
        f"%{dip_prob*100:.1f}", 
        delta=f"Acc: %{dip_acc*100:.0f}" if dip_prob > 0.5 else None,
        delta_color="normal"
    )

    # 5. PEAK Prediction
    peak_delta = "High Conf." if peak_acc > 0.6 else "Mod. Conf."
    kpi5.metric(
        "🔮 Peak Probability", 
        f"%{peak_prob*100:.1f}", 
        delta=f"Acc: %{peak_acc*100:.0f}" if peak_prob > 0.5 else None,  
        delta_color="inverse"
    )
    
    # 6. Trend / Neutral (The "Missing" %)
    # Since models are independent, we estimate neutral as roughly the remainder.
    # We clip to 0 to avoid negative numbers in edge cases.
    neutral_prob = max(0.0, 1.0 - (dip_prob + peak_prob))
    kpi6.metric(
        "➡️ Trend Prob",
        f"%{neutral_prob*100:.1f}",
        delta="Continuation"
    )
    
    # Signal Interpretation Banner
    st.divider()
    
    col_info, col_help = st.columns([4, 1])
    with col_help:
        st.caption("ℹ️ **Note on Probabilities**:\n'Dip' & 'Peak' are specific reversal events. They don't sum to 100% because the market is usually in **Trend Continuation** (Neutral).")
    
    with col_info:
        if dip_prob > 0.65:
            st.success(f"🟢 **POSSIBLE BOTTOM**: Model detects %{dip_prob*100:.0f} chance of a swing low. (Model Precision: %{dip_acc*100:.0f})")
        elif peak_prob > 0.65:
            st.error(f"🔴 **POSSIBLE TOP**: Model detects %{peak_prob*100:.0f} chance of a swing high. (Model Precision: %{peak_acc*100:.0f})")
        else:
            st.info(f"👉 **TREND CONTINUATION**: The market is likely to continue its current path (Neutral/Trend Probability: %{neutral_prob*100:.1f}). No strong reversal imminent.")

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
    # ACTUAL DIP/PEAK POINTS (Markers)
    # -------------------------------------------------------------
    if "Tepe" in df_window.columns:
        actual_peaks = df_window.dropna(subset=["Tepe"])
        if not actual_peaks.empty:
            fig.add_trace(go.Scatter(
                x=actual_peaks[date_col], y=actual_peaks["Tepe"],
                mode='markers', name='⚪ Gerçek Zirve',
                marker=dict(color='white', symbol='circle', size=8, 
                           line=dict(width=1, color='red'))
            ))
    
    if "Dip" in df_window.columns:
        actual_dips = df_window.dropna(subset=["Dip"])
        if not actual_dips.empty:
            fig.add_trace(go.Scatter(
                x=actual_dips[date_col], y=actual_dips["Dip"],
                mode='markers', name='⚪ Gerçek Dip',
                marker=dict(color='white', symbol='circle', size=8,
                           line=dict(width=1, color='lime'))
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
    Generates natural language insights based on technical state.
    """
    st.subheader("🤖 AI Insight Engine")
    
    last_row = df_window.iloc[-1]
    rsi = last_row.get("RSI", 50)
    regime = last_row.get("Regime", "Unknown")
    
    insights = []
    
    # ML Logic (Priority)
    if dip_prob > 0.60:
        insights.append(f"🟢 **POSSIBLE BUY SIGNAL**: AI Model detects a local **Dip** with {dip_prob:.0%} confidence.")
    elif peak_prob > 0.60:
        insights.append(f"🔴 **POSSIBLE SELL SIGNAL**: AI Model detects a local **Peak** with {peak_prob:.0%} confidence.")

    # RSI Logic
    if rsi > 70:
        insights.append(f"⚠️ **RSI Overbought ({rsi:.1f})**: Momentum is stretched. Reversal risk high.")
    elif rsi < 30:
        insights.append(f"✅ **RSI Oversold ({rsi:.1f})**: Potential mean reversion. Watch for stabilization.")
        
    # Regime Logic
    if regime == "Bullish":
        insights.append("🐂 **Primary Trend**: Uptrend. Look for confirmed dips to enter.")
    elif regime == "Bearish":
        insights.append("🐻 **Primary Trend**: Downtrend. Rallies are likely selling opportunities.")
        
    # Display
    for insight in insights:
        st.info(insight)
        
    if not insights:
        st.write("Market is behaving within normal parameters. No strong signals.")

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
