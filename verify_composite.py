import pandas as pd
import numpy as np
from ml_engine import MLEngine
from data_loader import load_data

def verify_composite_logic():
    # 1. Veriyi yükle ve tahminleri al
    df, date_col = load_data('BIST100_PREDICTION_READY.csv')
    engine = MLEngine(df)
    engine.train()
    
    # Tüm tabloya olasılıkları ekle
    df_all = engine.add_predictions_to_df(engine.df.copy())
    
    # 2. Parametreleri tanımla
    # Saf Sinyal Kuralı: Gap > RSI * 0.45
    df_all['gap'] = (df_all['AI_Peak_Prob'] - df_all['AI_Dip_Prob']) * 100
    df_all['purity_threshold'] = df_all['rsi_14'] * 0.45
    df_all['is_pure'] = (df_all['AI_Peak_Prob'] > 0.85) & (df_all['gap'] > df_all['purity_threshold'])
    
    # Israr Kuralı: Son 5 günün 3'ünde Peak_Prob > 0.70 olması
    df_all['high_conf_signal'] = (df_all['AI_Peak_Prob'] > 0.70).astype(int)
    df_all['persistence_count'] = df_all['high_conf_signal'].rolling(window=5).sum()
    
    # 3. KOMPOZİT SİNYAL: Hem SAF olacak, hem de SON 5 GÜNDE EN AZ 3 SİNYAL (Persistence) olacak
    df_all['composite_signal'] = df_all['is_pure'] & (df_all['persistence_count'] >= 3)
    
    # Analiz
    pure_total = df_all['is_pure'].sum()
    composite_total = df_all['composite_signal'].sum()
    
    # Doğru sinyaller (Label_Peak == 1 olanlar)
    pure_correct = df_all[df_all['is_pure'] & (df_all['Label_Peak'] == 1)].shape[0]
    composite_correct = df_all[df_all['composite_signal'] & (df_all['Label_Peak'] == 1)].shape[0]
    
    pure_precision = (pure_correct / pure_total * 100) if pure_total > 0 else 0
    composite_precision = (composite_correct / composite_total * 100) if composite_total > 0 else 0
    
    print("=== KOMPOZİT SİNYAL ANALİZİ ===")
    print(f"Sadece Saf Sinyal Sayısı: {pure_total}")
    print(f"Kompozit Sinyal Sayısı: {composite_total}")
    print("-" * 30)
    print(f"Saf Sinyal Hassasiyeti (Precision): %{pure_precision:.2f}")
    print(f"Kompozit Sinyal Hassasiyeti (Precision): %{composite_precision:.2f}")
    print("-" * 30)
    print(f"İyileşme Oranı: %{composite_precision - pure_precision:.2f}")

if __name__ == "__main__":
    verify_composite_logic()
