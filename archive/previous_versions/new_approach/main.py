import os
import sys

# Ensure src modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_and_aggregate_sku
from src.anomaly_detector import detect_anomalies
from src.feature_engineering import build_features, get_feature_columns
from src.backtester import run_backtesting

def main():
    excel_path = os.path.join(os.path.dirname(__file__), '..', 'Rimmel Brand Products Sales - 1 Jan 2025 to 31st July 2026.xlsx')
    output_dir = os.path.join(os.path.dirname(__file__), 'output_results')
    os.makedirs(output_dir, exist_ok=True)
    
    sku_code = 'RIM-MSC-E3DL-003'
    print(f"Loading and processing SKU: {sku_code}...")
    
    raw_df = load_and_aggregate_sku(excel_path, sku_code)
    anomaly_df = detect_anomalies(raw_df)
    feature_df = build_features(anomaly_df)
    
    feature_cols = get_feature_columns()
    clean_data = feature_df.dropna(subset=feature_cols).copy().reset_index(drop=True)
    
    windows = [
        {'name': 'Window 1: May 2026', 'test_start': '2026-05-01', 'test_end': '2026-05-30'},
        {'name': 'Window 2: June 2026', 'test_start': '2026-06-01', 'test_end': '2026-06-30'},
        {'name': 'Window 3: July 2026', 'test_start': '2026-07-01', 'test_end': '2026-07-30'}
    ]
    
    print("Running multi-window backtesting...")
    metrics_df, feature_imp_df = run_backtesting(clean_data, feature_cols, windows, output_dir, sku_code)
    
    print("\n" + "="*70)
    print("BACKTESTING METRICS SUMMARY:")
    print("="*70)
    print(metrics_df.to_string(index=False))
    
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE SUMMARY:")
    print("="*70)
    print(feature_imp_df.to_string(index=False))
    
    print(f"\nCompleted! Output artifacts saved to: {output_dir}")

if __name__ == '__main__':
    main()
