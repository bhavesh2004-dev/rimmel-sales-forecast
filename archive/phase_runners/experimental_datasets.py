"""
EXPERIMENTAL ML DATASETS MODULE (ZERO VERSION VS. AVERAGE VERSION)
==================================================================
Constructs two production ML-ready experimental datasets from the normalized ML dataset:
1. data/processed/rimmel_ml_zero_version.xlsx
   - Missing sales dates filled with units_sold = 0.
2. data/processed/rimmel_ml_average_version.xlsx
   - Missing sales dates filled using strictly causal, leakage-safe historical averages
     (SKU x Platform expanding mean -> Platform x Category fallback -> Platform macro fallback).

Both workbooks contain:
- MODEL_INPUT: Clean, stripped-down 11-column training sheet ready for ML feature engineering.
- daily_sku_platform_data: Full continuous daily observation layer with audit & treatment lineage.
- imputation_comparison: Summary analytics and sample side-by-side comparison slice.
- normalized_transactions: Untouched transaction records (61,511 rows).
- sku_master, platform_mapping, data_dictionary.
"""
import sys
import os
import time
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath('.'))

ML_READY_FILE = os.path.join('data', 'processed', 'rimmel_ml_ready_normalized.xlsx')
OUTPUT_ZERO_FILE = os.path.join('data', 'processed', 'rimmel_ml_zero_version.xlsx')
OUTPUT_AVG_FILE = os.path.join('data', 'processed', 'rimmel_ml_average_version.xlsx')

def generate_experimental_datasets():
    start_time = time.time()
    print("=" * 80)
    print("STARTING EXPERIMENTAL ML DATASET GENERATION (ZERO VS. AVERAGE)")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 1. Load source sheets from rimmel_ml_ready_normalized.xlsx
    print(f"\n[1/6] Loading source sheets from: {ML_READY_FILE}...")
    df_daily_raw = pd.read_excel(ML_READY_FILE, sheet_name='daily_sku_platform_obs')
    df_transactions = pd.read_excel(ML_READY_FILE, sheet_name='normalized_transactions')
    df_sku_master = pd.read_excel(ML_READY_FILE, sheet_name='sku_master')
    df_platform_mapping = pd.read_excel(ML_READY_FILE, sheet_name='platform_mapping')
    df_dict = pd.read_excel(ML_READY_FILE, sheet_name='data_dictionary')
    
    df_daily = df_daily_raw.copy()
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily = df_daily.sort_values(by=['platform_group', 'canonical_sku', 'date']).reset_index(drop=True)
    
    print(f"Loaded {len(df_daily):,} daily observation rows and {len(df_transactions):,} transaction rows.")
    
    # 2. Identify recorded sales vs missing observations
    # In daily layer, recorded sales have observation_state == 'OBSERVED_SALE' or (stockout with units > 0)
    is_recorded_sale = (df_daily['observation_state'] == 'OBSERVED_SALE') | (
        (df_daily['observation_state'] == 'STOCKOUT_DEMAND_CENSORED') & (df_daily['observed_units_sold'] > 0)
    )
    df_daily['recorded_units'] = np.where(is_recorded_sale, df_daily['observed_units_sold'], np.nan)
    
    recorded_sales_units = df_daily['recorded_units'].sum()
    print(f"Total recorded sales units (ground truth): {recorded_sales_units:,.1f} (Target: 164,786.0)")
    assert abs(recorded_sales_units - 164786.0) < 1e-4, f"Mismatch in recorded sales: {recorded_sales_units}"
    
    total_missing_obs = (~is_recorded_sale).sum()
    print(f"Total missing / non-sales daily observations to treat: {total_missing_obs:,} rows ({(total_missing_obs/len(df_daily))*100:.2f}%)")
    
    # 3. Compute Leakage-Safe Historical Averages for Average Version
    print("\n[2/6] Computing strictly causal expanding historical averages (Zero Future Leakage)...")
    
    # Level 1: Primary SKU x Platform expanding historical average
    grp_sku_p = df_daily.groupby(['platform_group', 'canonical_sku'])['recorded_units']
    sku_p_exp_sum = grp_sku_p.apply(lambda s: s.shift(1).fillna(0).cumsum()).reset_index(level=[0,1], drop=True)
    sku_p_exp_cnt = grp_sku_p.apply(lambda s: (~s.shift(1).isna()).cumsum()).reset_index(level=[0,1], drop=True)
    sku_p_avg = sku_p_exp_sum / sku_p_exp_cnt.replace(0, np.nan)
    
    df_daily['sku_p_exp_cnt'] = sku_p_exp_cnt
    df_daily['sku_p_avg'] = sku_p_avg
    
    # Level 2: Fallback Category x Platform expanding historical average
    all_dates = pd.date_range('2025-08-01', '2026-09-10', freq='D')
    cat_daily = df_daily.dropna(subset=['recorded_units']).groupby(['date', 'platform_group', 'category'])['recorded_units'].mean().reset_index()
    cat_pairs = df_daily[['platform_group', 'category']].drop_duplicates()
    cat_grid = []
    for _, r in cat_pairs.iterrows():
        for d in all_dates:
            cat_grid.append((d, r['platform_group'], r['category']))
    cat_grid_df = pd.DataFrame(cat_grid, columns=['date', 'platform_group', 'category']).sort_values(by=['platform_group', 'category', 'date']).reset_index(drop=True)
    cat_grid_df = cat_grid_df.merge(cat_daily, on=['date', 'platform_group', 'category'], how='left')
    grp_cat_p = cat_grid_df.groupby(['platform_group', 'category'])['recorded_units']
    cat_p_exp_sum = grp_cat_p.apply(lambda s: s.shift(1).fillna(0).cumsum()).reset_index(level=[0,1], drop=True)
    cat_p_exp_cnt = grp_cat_p.apply(lambda s: (~s.shift(1).isna()).cumsum()).reset_index(level=[0,1], drop=True)
    cat_grid_df['cat_p_avg'] = cat_p_exp_sum / cat_p_exp_cnt.replace(0, np.nan)
    df_daily = df_daily.merge(cat_grid_df[['date', 'platform_group', 'category', 'cat_p_avg']], on=['date', 'platform_group', 'category'], how='left')
    
    # Level 3: Fallback Platform Macro expanding historical average
    plat_daily = df_daily.dropna(subset=['recorded_units']).groupby(['date', 'platform_group'])['recorded_units'].mean().reset_index()
    plat_grid = []
    for p in df_daily['platform_group'].unique():
        for d in all_dates:
            plat_grid.append((d, p))
    plat_grid_df = pd.DataFrame(plat_grid, columns=['date', 'platform_group']).sort_values(by=['platform_group', 'date']).reset_index(drop=True)
    plat_grid_df = plat_grid_df.merge(plat_daily, on=['date', 'platform_group'], how='left')
    grp_plat = plat_grid_df.groupby('platform_group')['recorded_units']
    plat_p_exp_sum = grp_plat.apply(lambda s: s.shift(1).fillna(0).cumsum()).reset_index(level=0, drop=True)
    plat_p_exp_cnt = grp_plat.apply(lambda s: (~s.shift(1).isna()).cumsum()).reset_index(level=0, drop=True)
    plat_grid_df['plat_macro_avg'] = plat_p_exp_sum / plat_p_exp_cnt.replace(0, np.nan)
    df_daily = df_daily.merge(plat_grid_df[['date', 'platform_group', 'plat_macro_avg']], on=['date', 'platform_group'], how='left')
    
    # Apply hierarchy selection
    cond_primary = (df_daily['sku_p_exp_cnt'] >= 3) & df_daily['sku_p_avg'].notnull()
    cond_cat = df_daily['cat_p_avg'].notnull()
    cond_macro = df_daily['plat_macro_avg'].notnull()
    
    df_daily['avg_imputed_units'] = np.where(
        cond_primary,
        df_daily['sku_p_avg'],
        np.where(cond_cat, df_daily['cat_p_avg'], np.where(cond_macro, df_daily['plat_macro_avg'], 1.0))
    )
    
    df_daily['average_source_level'] = np.where(
        is_recorded_sale,
        'OBSERVED_SALE',
        np.where(
            cond_primary,
            'SKU_PLATFORM_HISTORICAL_AVG',
            np.where(cond_cat, 'PLATFORM_CATEGORY_FALLBACK', 'PLATFORM_MACRO_FALLBACK')
        )
    )
    
    # Fill any remaining Day-1 edge cases cleanly with 1.0
    df_daily['avg_imputed_units'] = df_daily['avg_imputed_units'].fillna(1.0).round(4)
    
    # Check average level distribution
    print("Average source level breakdown across all rows:")
    print(df_daily['average_source_level'].value_counts().to_string())
    
    # 4. Construct Datasets
    print("\n[3/6] Constructing Zero Version and Average Version dataframes...")
    
    # Shared base columns
    df_daily['date_str'] = df_daily['date'].dt.strftime('%Y-%m-%d')
    df_daily['in_stock_flag'] = np.where(df_daily['current_stock'] > 0, 1, 0)
    
    # ── ZERO VERSION ───────────────────────────────────────────────────────────
    df_zero = df_daily.copy()
    df_zero['units_sold'] = np.where(is_recorded_sale, df_zero['observed_units_sold'], 0.0)
    df_zero['orders_count'] = np.where(is_recorded_sale, df_zero['orders_count'], 0.0)
    df_zero['is_imputed'] = np.where(is_recorded_sale, 0, 1)
    df_zero['treatment_applied'] = np.where(is_recorded_sale, 'AS_OBSERVED', 'ZERO_IMPUTATION')
    df_zero['treatment_value'] = np.where(is_recorded_sale, df_zero['observed_units_sold'], 0.0)
    
    # MODEL_INPUT for Zero Version
    model_cols = [
        'date_str', 'platform_group', 'canonical_sku', 'category',
        'units_sold', 'orders_count', 'current_stock', 'in_stock_flag',
        'selling_price', 'pack_multiplier', 'is_imputed'
    ]
    df_zero_model_input = df_zero[model_cols].rename(columns={'date_str': 'date'}).copy()
    
    # Detailed daily data for Zero Version
    daily_cols_zero = [
        'date_str', 'platform_group', 'canonical_sku', 'canonical_sku_source',
        'resolved_parent_id', 'category', 'pack_multiplier',
        'units_sold', 'orders_count', 'selling_price', 'current_stock',
        'has_inventory_signal', 'stockout_flag', 'observation_state',
        'is_imputed', 'treatment_applied', 'treatment_value'
    ]
    df_zero_daily_full = df_zero[daily_cols_zero].rename(columns={'date_str': 'date'}).copy()
    
    # ── AVERAGE VERSION ────────────────────────────────────────────────────────
    df_avg = df_daily.copy()
    df_avg['units_sold'] = np.where(is_recorded_sale, df_avg['observed_units_sold'], df_avg['avg_imputed_units'])
    df_avg['orders_count'] = np.where(
        is_recorded_sale,
        df_avg['orders_count'],
        (df_avg['avg_imputed_units'] / df_avg['pack_multiplier']).round(4)
    )
    df_avg['is_imputed'] = np.where(is_recorded_sale, 0, 1)
    df_avg['treatment_applied'] = np.where(is_recorded_sale, 'AS_OBSERVED', 'LEAKAGE_SAFE_HISTORICAL_AVG')
    df_avg['treatment_value'] = np.where(is_recorded_sale, df_avg['observed_units_sold'], df_avg['avg_imputed_units'])
    
    df_avg_model_input = df_avg[model_cols].rename(columns={'date_str': 'date'}).copy()
    
    daily_cols_avg = [
        'date_str', 'platform_group', 'canonical_sku', 'canonical_sku_source',
        'resolved_parent_id', 'category', 'pack_multiplier',
        'units_sold', 'orders_count', 'selling_price', 'current_stock',
        'has_inventory_signal', 'stockout_flag', 'observation_state',
        'is_imputed', 'treatment_applied', 'treatment_value', 'average_source_level'
    ]
    df_avg_daily_full = df_avg[daily_cols_avg].rename(columns={'date_str': 'date'}).copy()
    
    # 5. Build Comparison Sheet
    print("\n[4/6] Constructing imputation comparison analytics & sample slice...")
    
    # Summary by Platform
    plat_comp = []
    for p in df_daily['platform_group'].unique():
        p_slice = df_daily[df_daily['platform_group'] == p]
        rec_cnt = (p_slice['recorded_units'].notnull()).sum()
        imp_cnt = (p_slice['recorded_units'].isnull()).sum()
        rec_u = p_slice['recorded_units'].sum()
        avg_u_added = p_slice[p_slice['recorded_units'].isnull()]['avg_imputed_units'].sum()
        avg_mean_val = p_slice[p_slice['recorded_units'].isnull()]['avg_imputed_units'].mean()
        sku_p_cnt = (p_slice['average_source_level'] == 'SKU_PLATFORM_HISTORICAL_AVG').sum()
        cat_cnt = (p_slice['average_source_level'] == 'PLATFORM_CATEGORY_FALLBACK').sum()
        macro_cnt = (p_slice['average_source_level'] == 'PLATFORM_MACRO_FALLBACK').sum()
        
        plat_comp.append({
            'platform_group': p,
            'total_observations': len(p_slice),
            'recorded_sales_days': rec_cnt,
            'missing_days_imputed': imp_cnt,
            'recorded_sales_units': round(rec_u, 1),
            'zero_version_imputed_units': 0.0,
            'zero_version_imputed_mean': 0.0,
            'avg_version_imputed_units': round(avg_u_added, 1),
            'avg_version_imputed_mean': round(avg_mean_val, 4),
            'sku_platform_level_pct': round((sku_p_cnt / imp_cnt) * 100, 2) if imp_cnt > 0 else 0.0,
            'category_fallback_pct': round((cat_cnt / imp_cnt) * 100, 2) if imp_cnt > 0 else 0.0,
            'platform_macro_fallback_pct': round((macro_cnt / imp_cnt) * 100, 2) if imp_cnt > 0 else 0.0
        })
    df_plat_comp = pd.DataFrame(plat_comp)
    
    # Representative side-by-side sample slice (200 rows showing mix of observed sales & imputed dates across top SKUs)
    top_skus = df_daily.groupby('canonical_sku')['recorded_units'].sum().sort_values(ascending=False).head(5).index.tolist()
    sample_mask = df_daily['canonical_sku'].isin(top_skus) & (df_daily['date'] >= '2025-08-01') & (df_daily['date'] <= '2025-08-31')
    sample_df = df_daily[sample_mask].copy()
    
    sample_slice = pd.DataFrame({
        'date': sample_df['date_str'],
        'platform_group': sample_df['platform_group'],
        'canonical_sku': sample_df['canonical_sku'],
        'category': sample_df['category'],
        'observation_state': sample_df['observation_state'],
        'is_recorded_sale': np.where(sample_df['recorded_units'].notnull(), 1, 0),
        'recorded_units_sold': sample_df['recorded_units'].fillna(0.0),
        'zero_version_units': np.where(sample_df['recorded_units'].notnull(), sample_df['recorded_units'], 0.0),
        'avg_version_units': np.where(sample_df['recorded_units'].notnull(), sample_df['recorded_units'], sample_df['avg_imputed_units']),
        'average_imputation_source': sample_df['average_source_level']
    }).sort_values(by=['canonical_sku', 'platform_group', 'date']).head(250)
    
    # 6. Export Workbooks
    print(f"\n[5/6] Exporting Zero Version Workbook: {OUTPUT_ZERO_FILE}...")
    with pd.ExcelWriter(OUTPUT_ZERO_FILE, engine='openpyxl') as writer:
        df_zero_model_input.to_excel(writer, sheet_name='MODEL_INPUT', index=False)
        df_zero_daily_full.to_excel(writer, sheet_name='daily_sku_platform_data', index=False)
        df_plat_comp.to_excel(writer, sheet_name='imputation_comparison_summary', index=False)
        sample_slice.to_excel(writer, sheet_name='imputation_comparison_samples', index=False)
        df_transactions.to_excel(writer, sheet_name='normalized_transactions', index=False)
        df_sku_master.to_excel(writer, sheet_name='sku_master', index=False)
        df_platform_mapping.to_excel(writer, sheet_name='platform_mapping', index=False)
        df_dict.to_excel(writer, sheet_name='data_dictionary', index=False)
    print(f"Successfully wrote {OUTPUT_ZERO_FILE} ({os.path.getsize(OUTPUT_ZERO_FILE):,} bytes)")
    
    print(f"\n[6/6] Exporting Average Version Workbook: {OUTPUT_AVG_FILE}...")
    with pd.ExcelWriter(OUTPUT_AVG_FILE, engine='openpyxl') as writer:
        df_avg_model_input.to_excel(writer, sheet_name='MODEL_INPUT', index=False)
        df_avg_daily_full.to_excel(writer, sheet_name='daily_sku_platform_data', index=False)
        df_plat_comp.to_excel(writer, sheet_name='imputation_comparison_summary', index=False)
        sample_slice.to_excel(writer, sheet_name='imputation_comparison_samples', index=False)
        df_transactions.to_excel(writer, sheet_name='normalized_transactions', index=False)
        df_sku_master.to_excel(writer, sheet_name='sku_master', index=False)
        df_platform_mapping.to_excel(writer, sheet_name='platform_mapping', index=False)
        df_dict.to_excel(writer, sheet_name='data_dictionary', index=False)
    print(f"Successfully wrote {OUTPUT_AVG_FILE} ({os.path.getsize(OUTPUT_AVG_FILE):,} bytes)")
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"EXPERIMENTAL DATASET GENERATION COMPLETE in {elapsed:.1f} seconds!")
    print("=" * 80)

if __name__ == '__main__':
    generate_experimental_datasets()
