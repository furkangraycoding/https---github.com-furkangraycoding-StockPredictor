import pandas as pd
import numpy as np
from ml_engine import MLEngine
from data_loader import load_data
from indicators import TechnicalAnalyzer

def verify_momentum_composite():
    # 1. Veriyi yükle
    df, date_col = load_data('BIST100_PREDICTION_READY.csv')
    
    # [KRİTİK] Teknik özellikleri baştan hesapla
    analyzer = TechnicalAnalyzer(df)
    analyzer.add_rsi()
    analyzer.add_zigzag_labels() # 'Dip' ve 'Tepe' için gerekli
    df = analyzer.add_derived_features()
    
    # 2. Modeli eğit
    engine = MLEngine(df)
    engine.train()
    
    # Tahminleri ekle
    df_all = engine.add_predictions_to_df(engine.df.copy())
    
    # 3. Kolon isimlerini eşitle
    rsi_col = 'RSI' if 'RSI' in df_all.columns else 'rsi_14'
    rsi_diff_col = 'RSI_Diff_1D'
    
    if rsi_diff_col not in df_all.columns:
        print(f"HATA: {rsi_diff_col} bulunamadı.")
        return

    # 4. Saf Sinyal Kuralı
    df_all['gap'] = (df_all['AI_Peak_Prob'] - df_all['AI_Dip_Prob']) * 100
    df_all['purity_threshold'] = df_all[rsi_col] * 0.45
    df_all['is_pure'] = (df_all['AI_Peak_Prob'] > 0.85) & (df_all['gap'] > df_all['purity_threshold'])
    
    # 5. Momentum Filtresi (RSI Aşağı Dönüş)
    df_all['is_turning_down'] = df_all[rsi_diff_col] < 0
    
    # 6. KOMPOZİT SİNYAL
    df_all['momentum_composite'] = df_all['is_pure'] & df_all['is_turning_down']
    
    # Analiz
    pure_total = df_all['is_pure'].sum()
    composite_total = df_all['momentum_composite'].sum()
    
    pure_correct = df_all[df_all['is_pure'] & (df_all['Label_Peak'] == 1)].shape[0]
    composite_correct = df_all[df_all['momentum_composite'] & (df_all['Label_Peak'] == 1)].shape[0]
    
    pure_precision = (pure_correct / pure_total * 100) if pure_total > 0 else 0
    composite_precision = (composite_correct / composite_total * 100) if composite_total > 0 else 0
    
    print("\n=== MOMENTUM KOMPOZİT SİNYAL ANALİZİ ===")
    print(f"Sadece Saf Sinyal Günü: {pure_total} -> Hassasiyet: %{pure_precision:.2f}")
    print(f"Momentum Onaylı Gün: {composite_total} -> Hassasiyet: %{composite_precision:.2f}")
    print("-" * 30)
    print(f"Hassasiyet Artışı: %{composite_precision - pure_precision:.2f}")

if __name__ == "__main__":
    verify_momentum_composite()
