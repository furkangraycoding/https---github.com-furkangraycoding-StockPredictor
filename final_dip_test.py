import pandas as pd
import numpy as np
from ml_engine import MLEngine
from data_loader import load_data
from indicators import TechnicalAnalyzer

def test_final_dip_logic():
    # 1. Load data
    df, date_col = load_data('BIST100_PREDICTION_READY.csv')
    analyzer = TechnicalAnalyzer(df)
    analyzer.add_rsi()
    analyzer.add_zigzag_labels()
    df = analyzer.add_derived_features()
    
    engine = MLEngine(df)
    engine.train()
    df_all = engine.add_predictions_to_df(engine.df.copy())
    
    df_all['gap'] = (df_all['AI_Dip_Prob'] - df_all['AI_Peak_Prob']) * 100
    df_all['is_exact_dip'] = df_all['Dip'].notna()

    # Final Hybrid Logic (0.72)
    def is_final_pure(row):
        prob = row['AI_Dip_Prob']
        gap = row['gap']
        rsi = row['RSI']
        
        # Condition A: 85% Prob + 1.1x RSI Gap
        cond_a = (prob >= 0.85) and (gap > rsi * 1.1)
        # Condition B: 72% Prob + 1.6x RSI Gap
        cond_b = (prob >= 0.72) and (gap > rsi * 1.6)
        
        return cond_a or cond_b

    df_all['is_pure_final'] = df_all.apply(is_final_pure, axis=1)

    def calculate_metrics(col_name):
        subset = df_all[df_all[col_name] == True]
        total = len(subset)
        if total == 0: return 0, 0, 0, 0
        
        exact = df_all[(df_all[col_name] == True) & (df_all['is_exact_dip'] == True)].shape[0]
        
        near1 = 0
        for idx in subset.index:
            loc = df_all.index.get_loc(idx)
            window = df_all.iloc[max(0, loc-1) : min(len(df_all), loc+2)]
            if window['is_exact_dip'].any():
                near1 += 1
                
        near3 = 0
        for idx in subset.index:
            loc = df_all.index.get_loc(idx)
            window = df_all.iloc[max(0, loc-3) : min(len(df_all), loc+4)]
            if window['is_exact_dip'].any():
                near3 += 1
        
        return total, (exact/total*100), (near1/total*100), (near3/total*100)

    print("\n" + "="*75)
    print("🏆 SAF DİP SİNYALİ (FİNAL HİBRİT %72) PERFORMANSI")
    print("="*75)
    
    c, e, n1, n3 = calculate_metrics('is_pure_final')
    print(f"Toplam Üretilen Sinyal   : {c} Gün")
    print("-" * 75)
    print(f"📍 TAM ÜSTÜNE (Exact Day)       : %{e:.2f}")
    print(f"🎯 ÇOK YAKIN (+/- 1 GÜN)        : %{n1:.2f}")
    print(f"🛡️ GÜVENLİ BÖLGE (+/- 3 GÜN)    : %{n3:.2f}")
    print("-" * 75)
    
    # Check 17 Oct specifically
    oct17_row = df_all[df_all[date_col].astype(str).str.contains('2024-10-17')]
    if not oct17_row.empty:
        status = is_final_pure(oct17_row.iloc[0])
        print(f"DEBUG: 17 Ekim 2024 Sinyal Durumu: {'YANDI ✅' if status else 'YANMADI ❌'}")
        print(f"DEBUG: 17 Ekim Verileri -> Prob: %{oct17_row.iloc[0]['AI_Dip_Prob']*100:.1f}, Gap: {oct17_row.iloc[0]['gap']:.1f}, Eşik (1.6xRSI): {oct17_row.iloc[0]['RSI']*1.6:.1f}")

if __name__ == "__main__":
    test_final_dip_logic()
