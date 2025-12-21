import pandas as pd
import streamlit as st

@st.cache_data
def load_data(file_path: str):
    """
    Loads data from a CSV file, handles flexible date column identification,
    and returns a sorted DataFrame.
    """
    df = pd.read_csv(file_path)

    # Automatically find the date column
    date_col = None
    for col in df.columns:
        if col.lower() in ["date", "datetime", "time"]:
            date_col = col
            break

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
    else:
        # If no explicit column, try the index
        try:
            df.index = pd.to_datetime(df.index)
            df["Date"] = df.index
            date_col = "Date"
        except Exception as e:
            st.error(f"Could not identify a date column or parse index as date. Error: {e}")
            return pd.DataFrame(), None

    # Sort by date to ensure time-series integrity
    df = df.sort_values(date_col)
    
    # Clean numeric columns (handle "1,149.03" strings)
    # We target common price/volume columns regardless of case
    target_cols = ["price", "open", "high", "low", "close", "adj close", "vol.", "vol", "volume"]
    
    for col in df.columns:
        if col.lower() in target_cols or col.lower() in [c.lower() for c in target_cols]:
            if df[col].dtype == 'object':
                try:
                    # Remove commas and convert to float
                    df[col] = df[col].astype(str).str.replace(',', '').astype(float)
                except Exception:
                    pass # unexpected format, leave as is or let downstream fail gracefully

    return df, date_col
