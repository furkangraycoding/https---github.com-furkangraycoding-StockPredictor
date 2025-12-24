"""
Forward Testing: Gerçek Dip/Zirve Noktaları için Saf Sinyal Başarısı
Her gerçek nokta için, N gün öncesinden model eğitip saf sinyal üretip üretmediğini test eder.
"""
import pandas as pd
import numpy as np
from data_loader import load_data
from indicators import TechnicalAnalyzer
from ml_engine import MLEngine
import warnings
warnings.filterwarnings('ignore')

def find_real_pivots(df, date_col):
    """Gerçek dip ve zirve noktalarını bulur (ZigZag ile belirlenmiş)"""
    # ZigZag ile pivot noktalarını bul
    analyzer = TechnicalAnalyzer(df)
    analyzer.add_atr(14)
    analyzer.add_zigzag_labels(threshold_pct=0.05)
    df_with_pivots = analyzer.get_df()
    
    # Dip noktaları
    dip_points = df_with_pivots[df_with_pivots["Dip"].notna()].copy()
    dip_points["type"] = "Dip"
    dip_points["pivot_price"] = dip_points["Dip"]
    
    # Zirve noktaları
    peak_points = df_with_pivots[df_with_pivots["Tepe"].notna()].copy()
    peak_points["type"] = "Peak"
    peak_points["pivot_price"] = peak_points["Tepe"]
    
    # Birleştir ve sırala
    all_pivots = pd.concat([dip_points, peak_points]).sort_values(date_col)
    
    return all_pivots, df_with_pivots

def train_and_check_pure_signal(df, df_analyzed, date_col, target_date, lookback_days, pivot_type):
    """
    Belirli bir tarihten N gün öncesine kadar veriyle model eğitir
    ve o tarihte saf sinyal üretip üretmediğini kontrol eder.
    
    Args:
        df: Tüm veri
        date_col: Tarih kolonu
        target_date: Test edilecek tarih (gerçek pivot noktası)
        lookback_days: Kaç gün geriye gidilecek
        pivot_type: "Dip" veya "Peak"
    
    Returns:
        dict: Sonuçlar (pure_signal, dip_prob, peak_prob, etc.)
    """
    try:
        # Target date'i pandas Timestamp'e çevir
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        
        # df_analyzed'de target date'in pozisyonunu bul
        target_mask = df_analyzed[date_col].dt.date == target_date.date()
        target_indices = df_analyzed[target_mask].index
        
        if len(target_indices) == 0:
            # Alternatif: en yakın tarihi bul
            date_diff = (df_analyzed[date_col].dt.date - target_date.date()).abs()
            closest_idx = date_diff.idxmin()
            target_pos = df_analyzed.index.get_loc(closest_idx)
        else:
            target_idx = target_indices[0]
            target_pos = df_analyzed.index.get_loc(target_idx)
        
        # Lookback: target_date'ten N gün öncesine kadar olan TÜM veriyi kullan
        # lookback_days=0 ise, maksimum veri kullan (target_date'e kadar, target hariç)
        if lookback_days == 0:
            # Maksimum veri kullan (target_date'e kadar, target hariç)
            start_pos = 0  # En baştan başla
            end_pos = target_pos  # Target'a kadar (target hariç)
        else:
            # N gün öncesine kadar: target_date'ten N gün öncesine kadar olan tüm veri
            # Önce N gün öncesini bul
            target_date_obj = df_analyzed.iloc[target_pos][date_col]
            lookback_date = target_date_obj - pd.Timedelta(days=lookback_days)
            
            # Lookback date'ten önceki tüm veriyi kullan
            lookback_mask = df_analyzed[date_col] <= lookback_date
            if lookback_mask.sum() > 0:
                start_pos = 0  # En baştan başla
                # Lookback date'e kadar olan son satır
                lookback_indices = df_analyzed[lookback_mask].index
                end_pos = df_analyzed.index.get_loc(lookback_indices[-1]) + 1
            else:
                # Eğer lookback date çok erken ise, sadece target'tan N gün öncesi
                start_pos = max(0, target_pos - lookback_days)
                end_pos = target_pos
        
        train_data = df_analyzed.iloc[start_pos:end_pos].copy()
        
        # Minimum veri kontrolü (lookback'e göre esnek)
        # Küçük lookback'ler için daha az veri yeterli
        if lookback_days == 0:
            min_required = 100
        elif lookback_days <= 2:
            min_required = 20  # 1-2 gün için minimum
        else:
            min_required = 30  # 3-5 gün için minimum
        
        if len(train_data) < min_required:
            if lookback_days == 5:  # Sadece ilk lookback'te log
                print(f"     → Yetersiz veri: {len(train_data)} satır (min {min_required} gerekli)")
            return None
        
        # train_data zaten df_analyzed'den geldi, tüm feature'lar var
        # Sadece ZigZag label'larını yeniden hesapla (lookback window için)
        # Çünkü ZigZag label'ları tüm veri üzerinde hesaplanmış olabilir
        train_analyzer = TechnicalAnalyzer(train_data)
        if "ATR" not in train_data.columns:
            train_analyzer.add_atr(14)
        train_analyzer.add_zigzag_labels(threshold_pct=0.05)
        train_data = train_analyzer.get_df()
        
        # Eksik feature'ları ekle
        if "Volatility_20" not in train_data.columns:
            train_analyzer.add_rolling_volatility()
        if "Drawdown_Pct" not in train_data.columns:
            train_analyzer.add_drawdown_features()
        train_data = train_analyzer.add_derived_features()
        
        engine = MLEngine(train_data)
        metrics, _ = engine.train(optimize=False)  # Hızlı eğitim
        
        # Target date ve sonraki 3 günü içeren window oluştur
        # Forward confirmation için sonraki günler gerekli
        target_date_analyzed = df_analyzed[df_analyzed[date_col].dt.date == target_date.date()]
        if len(target_date_analyzed) == 0:
            # En yakın tarihi bul
            date_diff = (df_analyzed[date_col].dt.date - target_date.date()).abs()
            closest_idx = date_diff.idxmin()
            target_pos_in_analyzed = df_analyzed.index.get_loc(closest_idx)
        else:
            target_pos_in_analyzed = df_analyzed.index.get_loc(target_date_analyzed.index[0])
        
        # Target date'ten sonraki 3 günü de al (forward confirmation için)
        end_pos = min(len(df_analyzed), target_pos_in_analyzed + 4)  # Target + 3 gün sonrası
        prediction_window = df_analyzed.iloc[max(0, target_pos_in_analyzed - 4):end_pos].copy()
        
        if len(prediction_window) < 5:  # En az target + 1 gün sonrası gerekli
            if lookback_days == 5:
                print(f"     → Forward confirmation için yetersiz veri")
            return None
        
        # Forward confirmation ile saf sinyal kontrolü
        # MLEngine'in add_predictions_to_df metodunu kullan (forward confirmation dahil)
        df_with_predictions = engine.add_predictions_to_df(
            prediction_window.copy(), 
            use_forward_confirmation=True
        )
        
        # Target günün sinyalini kontrol et
        target_mask = df_with_predictions[date_col].dt.date == target_date.date()
        if target_mask.sum() == 0:
            # En yakın tarihi bul
            date_diff = (df_with_predictions[date_col].dt.date - target_date.date()).abs()
            closest_idx = date_diff.idxmin()
            target_row_pred = df_with_predictions.loc[closest_idx]
        else:
            target_row_pred = df_with_predictions[target_mask].iloc[0]
        
        # Sonuç
        if pivot_type == "Dip":
            pure_signal = target_row_pred["AI_Dip"] == 1
            signal_prob = target_row_pred["AI_Dip_Prob"]
            dip_prob = target_row_pred["AI_Dip_Prob"]
            peak_prob = target_row_pred["AI_Peak_Prob"]
        else:  # Peak
            pure_signal = target_row_pred["AI_Peak"] == 1
            signal_prob = target_row_pred["AI_Peak_Prob"]
            dip_prob = target_row_pred["AI_Dip_Prob"]
            peak_prob = target_row_pred["AI_Peak_Prob"]
        
        # Gap değerleri (raporlama için)
        peak_gap = (target_row_pred["AI_Peak_Prob"] - target_row_pred["AI_Dip_Prob"]) * 100
        dip_gap = (target_row_pred["AI_Dip_Prob"] - target_row_pred["AI_Peak_Prob"]) * 100
        
        # RSI değeri
        rsi = target_row_pred.get("RSI", target_row_pred.get("rsi_14", 50))
        if pd.isna(rsi):
            rsi = 50  # Default
        peak_threshold = rsi * 0.48
        
        return {
            "lookback_days": lookback_days,
            "pure_signal": pure_signal,
            "dip_prob": dip_prob,
            "peak_prob": peak_prob,
            "signal_prob": signal_prob,
            "rsi": rsi,
            "train_size": len(train_data),
            "peak_gap": peak_gap,
            "dip_gap": dip_gap,
            "peak_threshold": peak_threshold
        }
    except Exception as e:
        import traceback
        # İlk pivot'ta detaylı hata göster
        if lookback_days == 5:
            error_msg = str(e)[:100]  # İlk 100 karakter
            print(f"  ⚠️ Hata (lookback={lookback_days}): {type(e).__name__}: {error_msg}")
            if "Last_Signal" in str(e) or "Label" in str(e):
                print(f"     → Muhtemelen ZigZag label'ları eksik")
            elif "KeyError" in str(e):
                print(f"     → Eksik kolon hatası")
        return None

def forward_test_pure_signals():
    """Ana test fonksiyonu"""
    print("=" * 80)
    print("FORWARD TEST: Gerçek Pivot Noktaları için Saf Sinyal Başarısı")
    print("=" * 80)
    
    # Veri yükle
    df, date_col = load_data('BIST100_PREDICTION_READY.csv')
    print(f"\n✓ Veri yüklendi: {len(df)} satır")
    
    # Tarih kolonunu kontrol et
    if date_col not in df.columns:
        print(f"❌ Tarih kolonu bulunamadı: {date_col}")
        return
    
    df[date_col] = pd.to_datetime(df[date_col])
    
    # 2025 yılı pivot noktalarını bulmak için tüm veriyi kullan
    # Ama sadece 2025 yılındaki pivot'ları test edeceğiz
    print(f"✓ Tüm veri kullanılacak (2025 pivot'ları için): {len(df)} satır")
    
    # Feature'ları hazırla (tüm veri üzerinde)
    print("\n📊 Feature'lar hazırlanıyor...")
    analyzer = TechnicalAnalyzer(df)
    analyzer.add_moving_averages()
    analyzer.add_rsi()
    analyzer.add_atr()
    analyzer.determine_regime()
    analyzer.add_zigzag_labels(threshold_pct=0.05)
    analyzer.add_rolling_volatility()
    analyzer.add_drawdown_features()
    df_analyzed = analyzer.add_derived_features()
    
    # 2025 yılı pivot noktalarını bul
    print("\n🔍 Gerçek pivot noktaları bulunuyor...")
    all_pivots, df_with_pivots = find_real_pivots(df_analyzed, date_col)
    
    # 2025 yılı pivot'ları
    pivots_2025 = all_pivots[all_pivots[date_col].dt.year == 2025].copy()
    
    if len(pivots_2025) == 0:
        print("❌ 2025 yılında pivot noktası bulunamadı")
        return
    
    print(f"✓ 2025 yılında {len(pivots_2025)} pivot noktası bulundu")
    print(f"  - Dip: {len(pivots_2025[pivots_2025['type'] == 'Dip'])}")
    print(f"  - Peak: {len(pivots_2025[pivots_2025['type'] == 'Peak'])}")
    
    # Her pivot için test
    results = []
    lookback_periods = [5, 4, 3, 2, 1, 0]  # 0 = tam gün
    
    print("\n" + "=" * 80)
    print("TEST BAŞLIYOR...")
    print("=" * 80)
    
    for idx, pivot_row in pivots_2025.iterrows():
        pivot_date = pivot_row[date_col]
        pivot_type = pivot_row["type"]
        pivot_price = pivot_row["pivot_price"]
        
        print(f"\n📍 {pivot_type} - {pivot_date.strftime('%Y-%m-%d')} @ {pivot_price:.2f}")
        
        # Her lookback period için test
        for lookback in lookback_periods:
            result = train_and_check_pure_signal(
                df_analyzed, df_analyzed, date_col, pivot_date, lookback, pivot_type
            )
            
            if result:
                result["pivot_date"] = pivot_date
                result["pivot_type"] = pivot_type
                result["pivot_price"] = pivot_price
                results.append(result)
                
                status = "✅" if result["pure_signal"] else "❌"
                print(f"  {status} Lookback {lookback:2d}g: Pure={result['pure_signal']}, "
                      f"Prob={result['signal_prob']:.2f}, RSI={result['rsi']:.1f}")
            else:
                print(f"  ⚠️  Lookback {lookback:2d}g: Yetersiz veri veya hata")
    
    # Sonuçları analiz et
    print("\n" + "=" * 80)
    print("SONUÇLAR")
    print("=" * 80)
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("❌ Sonuç bulunamadı")
        return
    
    # Pivot tipine göre grupla
    for pivot_type in ["Dip", "Peak"]:
        type_results = results_df[results_df["pivot_type"] == pivot_type]
        if len(type_results) == 0:
            continue
        
        print(f"\n📊 {pivot_type} Noktaları:")
        print("-" * 80)
        
        # Lookback period'a göre başarı oranı
        for lookback in lookback_periods:
            lookback_results = type_results[type_results["lookback_days"] == lookback]
            if len(lookback_results) == 0:
                continue
            
            total = len(lookback_results)
            success = lookback_results["pure_signal"].sum()
            success_rate = (success / total * 100) if total > 0 else 0
            
            avg_prob = lookback_results["signal_prob"].mean()
            avg_rsi = lookback_results["rsi"].mean()
            
            print(f"  Lookback {lookback:2d}g: {success:2d}/{total:2d} başarılı "
                  f"(%{success_rate:5.1f}) | Avg Prob: {avg_prob:.3f} | Avg RSI: {avg_rsi:.1f}")
        
        # Genel istatistikler
        total_pivots = type_results["pivot_date"].nunique()
        overall_success = type_results.groupby("pivot_date")["pure_signal"].any().sum()
        overall_rate = (overall_success / total_pivots * 100) if total_pivots > 0 else 0
        
        print(f"\n  📈 Genel: {overall_success}/{total_pivots} pivot'ta en az 1 lookback'te saf sinyal "
              f"(%{overall_rate:.1f})")
    
    # En iyi lookback period
    print("\n" + "=" * 80)
    print("EN İYİ LOOKBACK PERIOD")
    print("=" * 80)
    
    for pivot_type in ["Dip", "Peak"]:
        type_results = results_df[results_df["pivot_type"] == pivot_type]
        if len(type_results) == 0:
            continue
        
        best_lookback = None
        best_rate = 0
        
        for lookback in lookback_periods:
            lookback_results = type_results[type_results["lookback_days"] == lookback]
            if len(lookback_results) == 0:
                continue
            
            success_rate = (lookback_results["pure_signal"].sum() / len(lookback_results) * 100)
            if success_rate > best_rate:
                best_rate = success_rate
                best_lookback = lookback
        
        if best_lookback is not None:
            print(f"{pivot_type}: En iyi lookback = {best_lookback}g (%{best_rate:.1f} başarı)")
    
    # Detaylı sonuçları CSV'ye kaydet
    output_file = "forward_test_pure_signals_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Detaylı sonuçlar kaydedildi: {output_file}")
    
    return results_df

if __name__ == "__main__":
    results = forward_test_pure_signals()

