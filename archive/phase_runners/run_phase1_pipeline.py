"""
PHASE 1.1 PIPELINE RUNNER & ORCHESTRATOR
========================================
Executes end-to-end data audit, targeted corrections, and exports the two workbooks:
1. data/processed/rimmel_full_normalized.xlsx (101,085 transaction rows)
2. data/processed/rimmel_ml_ready_normalized.xlsx (573,678 daily observation rows + 61,511 transaction rows)
"""
import sys
import os
import time
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.abspath('.'))

from src.data_ingestion import load_raw_dataset, RAW_SOURCE_FILE, BACKUP_COPY_FILE
from src.platform_mapping import build_platform_mapping_table
from src.sku_mapping import build_sku_master
from src.data_cleaning import audit_and_classify_data_issues, build_transformation_log
from src.observation_engine import build_observation_states_table
from src.normalization import normalize_transactions, build_daily_sku_platform_layer
from src.reconciliation import run_reconciliation
from src.export_excel import export_full_workbook, export_ml_ready_workbook

OUTPUT_FULL_FILE = os.path.join('data', 'processed', 'rimmel_full_normalized.xlsx')
OUTPUT_ML_READY_FILE = os.path.join('data', 'processed', 'rimmel_ml_ready_normalized.xlsx')

def run_phase1_1():
    start_time = time.time()
    print("=" * 80)
    print("STARTING PHASE 1.1: TARGETED DATA CORRECTION & DAILY OBSERVATION LAYER")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 1. Ingest Raw Dataset
    df_raw, raw_metadata = load_raw_dataset()
    
    # 2. Normalize Transaction Layer (Full 101,085 rows)
    print("\n[PIPELINE] Normalizing transaction records with resolved identifiers & child_asin fix...")
    df_transactions_full = normalize_transactions(df_raw)
    
    # Check non-Amazon child_asin
    non_amz_asin = df_transactions_full[df_transactions_full['platform_group'] != 'Amazon']['child_asin'].notnull().sum()
    print(f"[VERIFICATION] Non-Amazon child_asin count: {non_amz_asin} (MUST BE 0)")
    assert non_amz_asin == 0, f"Error: {non_amz_asin} non-Amazon child_asin values found!"
    
    # 3. Partition ML Period Transactions (61,511 rows)
    df_transactions_ml = df_transactions_full[df_transactions_full['date'] >= '2025-08-01'].copy()
    print(f"[PIPELINE] Partitioned ML Transactions: {len(df_transactions_ml):,} rows (strictly >= 2025-08-01)")
    
    # 4. Build True Daily SKU x Platform Observation Layer
    print("\n[PIPELINE] Building True Daily SKU x Platform Observation Layer (DATE x PLATFORM x CANONICAL_SKU)...")
    df_daily_obs = build_daily_sku_platform_layer(df_transactions_ml)
    print(f"[PIPELINE] Generated Daily Observation Layer: {len(df_daily_obs):,} rows across 406 calendar days")
    print(f"[PIPELINE] Daily Observation State Distribution:\n{df_daily_obs['observation_state'].value_counts().to_string()}")
    
    # 5. Build Master Catalog & Reference Tables
    print("\n[PIPELINE] Building platform mapping master table...")
    df_platform_mapping = build_platform_mapping_table(df_raw)
    
    print("[PIPELINE] Building SKU catalog master table...")
    df_sku_master = build_sku_master(df_raw)
    df_sku_master_ml = build_sku_master(df_raw[df_raw['date'] >= '2025-08-01'])
    
    print("[PIPELINE] Running data quality audit & classifying anomalies...")
    df_data_quality = audit_and_classify_data_issues(df_raw)
    
    print("[PIPELINE] Generating transformation audit log...")
    df_transformation_log = build_transformation_log()
    
    print("[PIPELINE] Constructing observation states reference table...")
    df_observation_states = build_observation_states_table(df_raw)
    
    # Update observation states table with daily layer empirical counts
    daily_state_counts = df_daily_obs['observation_state'].value_counts()
    for idx, row in df_observation_states.iterrows():
        s_name = row['state_name']
        if s_name in daily_state_counts:
            df_observation_states.at[idx, 'count_in_dataset'] = int(daily_state_counts[s_name])
            df_observation_states.at[idx, 'percentage'] = f"{(daily_state_counts[s_name] / len(df_daily_obs)) * 100:.2f}%"
            
    # 6. Run Reconciliation
    print("\n[PIPELINE] Computing exact unit, order, SKU, and platform reconciliation...")
    df_reconciliation = run_reconciliation(df_raw, df_transactions_full, df_transactions_ml)
    print(df_reconciliation[['metric', 'raw_dataset', 'full_normalized', 'ml_eligible', 'reconciliation_status']].to_string(index=False))
    
    # Assert reconciliation invariants
    assert df_transactions_full['observed_units_sold'].sum() == df_raw['units_sold'].sum() == 273999
    assert df_transactions_full['orders_count'].sum() == df_raw['orders_count'].sum() == 231252
    assert len(df_transactions_full) == len(df_raw) == 101085
    assert len(df_transactions_ml) == 61511
    assert df_transactions_ml['observed_units_sold'].sum() == 164786
    assert df_transactions_ml['orders_count'].sum() == 139605
    
    # Assert daily layer unit conservation
    daily_units = df_daily_obs['observed_units_sold'].sum()
    print(f"[VERIFICATION] Daily Layer Total Observed Units: {daily_units:,} (Target: 164,786)")
    assert daily_units == 164786, f"Daily units mismatch: expected 164,786, got {daily_units}"
    print("[VERIFICATION] 100% Mathematical Reconciliation Verified across all layers!")
    
    # 7. Export Workbooks
    print("\n[PIPELINE] Exporting Workbook 1: Full Normalized Dataset...")
    export_full_workbook(
        filepath=OUTPUT_FULL_FILE,
        df_transactions=df_transactions_full,
        df_sku_master=df_sku_master,
        df_platform_mapping=df_platform_mapping,
        df_observation_states=df_observation_states,
        df_data_quality=df_data_quality,
        df_reconciliation=df_reconciliation,
        df_transformation_log=df_transformation_log
    )
    
    print("\n[PIPELINE] Exporting Workbook 2: ML-Ready Normalized Dataset...")
    export_ml_ready_workbook(
        filepath=OUTPUT_ML_READY_FILE,
        df_daily_obs=df_daily_obs,
        df_transactions_ml=df_transactions_ml,
        df_sku_master=df_sku_master_ml,
        df_platform_mapping=build_platform_mapping_table(df_raw[df_raw['date'] >= '2025-08-01']),
        df_observation_states=df_observation_states,
        df_data_quality=df_data_quality,
        df_reconciliation=df_reconciliation,
        df_transformation_log=df_transformation_log
    )
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"PHASE 1.1 TARGETED CORRECTIONS COMPLETE in {elapsed:.1f} seconds!")
    print(f"Full Workbook:     {OUTPUT_FULL_FILE} ({os.path.getsize(OUTPUT_FULL_FILE):,} bytes)")
    print(f"ML-Ready Workbook: {OUTPUT_ML_READY_FILE} ({os.path.getsize(OUTPUT_ML_READY_FILE):,} bytes)")
    print("=" * 80)

if __name__ == '__main__':
    run_phase1_1()
