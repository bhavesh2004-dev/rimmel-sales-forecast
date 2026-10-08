"""
Comprehensive Diagnostic Script: Zero-Forecast & Date Grain Investigation
=========================================================================
"""
import openpyxl
import pandas as pd
import numpy as np
import sqlite3
import json
import pickle

# 1. Inspect Current Excel Workbooks
print("=" * 80)
print("1. CURRENT EXCEL GRAIN & ROW DUPLICATION ANALYSIS")
print("=" * 80)
wb_v = openpyxl.load_workbook('reports/validation_report_sep_01_to_10_2026.xlsx', data_only=True)
df_v = pd.DataFrame(wb_v['Daily SKU Validation'].values)
df_v.columns = df_v.iloc[0]
df_v = df_v[1:].reset_index(drop=True)

wb_f = openpyxl.load_workbook('reports/production_forecast_sep_11_to_20_2026.xlsx', data_only=True)
df_f = pd.DataFrame(wb_f['Daily SKU Forecast'].values)
df_f.columns = df_f.iloc[0]
df_f = df_f[1:].reset_index(drop=True)

print(f"Validation Sheet Shape: {df_v.shape}")
print(f"Forecast Sheet Shape:   {df_f.shape}")

v_dups = df_v.duplicated(subset=['Date', 'SKU']).sum()
f_dups = df_f.duplicated(subset=['Date', 'SKU']).sum()
print(f"Duplicate (Date, SKU) in Validation: {v_dups}")
print(f"Duplicate (Date, SKU) in Forecast:   {f_dups}")

sku_f_counts = df_f.groupby('SKU').agg(
    row_count=('Date', 'count'),
    unique_dates=('Date', 'nunique')
).reset_index()
sku_f_counts['duplicate_dates'] = sku_f_counts['row_count'] - sku_f_counts['unique_dates']
print(f"SKUs with duplicate dates: {(sku_f_counts['duplicate_dates'] > 0).sum()}")
print(f"Rows per SKU distribution:\n{sku_f_counts['row_count'].value_counts()}")

print("\n--- WHY THE USER SAW REPEATED DATES ---")
print("Top 15 rows of production_forecast_sep_11_to_20_2026.xlsx:")
print(df_f[['Date', 'SKU', 'Amazon Predicted Units', 'eBay Predicted Units', 'Total Predicted Units']].head(15))

print("\nNotice: The table is ordered by DATE then SKU:")
print(f"Dates on row 1 to row 674: All {df_f.iloc[0]['Date']}!")
print(f"Date on row 675: {df_f.iloc[674]['Date']}!")

# 2. Trace 20 SKUs from Raw Model Output to Excel Output
print("\n" + "=" * 80)
print("2. TRACING 20 SAMPLE SKUS: RAW MODEL -> PLATFORM -> SKU -> EXCEL")
print("=" * 80)

# Load raw database and model
conn = sqlite3.connect('data/rimmel_clean.db')
df_raw = pd.read_sql_query("SELECT * FROM ml_features_zero WHERE date = '2026-09-10'", conn)
conn.close()

with open('models/production_features.json', 'r') as f:
    meta = json.load(f)
feature_cols = meta['feature_list']
for c in meta['categorical_features']:
    if c in feature_cols:
        df_raw[c] = df_raw[c].astype('category')

with open('models/production_lgbm_model.pkl', 'rb') as f:
    prod_model = pickle.load(f)

# Pick 20 diverse SKUs: 5 high volume, 5 moderate, 5 intermittent, 5 zero
# Top 5 by historical v90
top_skus = df_raw.groupby('canonical_sku')['v90'].sum().sort_values(ascending=False).head(5).index.tolist()
# Moderate
mid_skus = df_raw.groupby('canonical_sku')['v90'].sum().sort_values(ascending=False).iloc[100:105].index.tolist()
# Low/intermittent
low_skus = df_raw.groupby('canonical_sku')['v90'].sum().sort_values(ascending=False).iloc[250:255].index.tolist()
# Quiet/zero
quiet_skus = df_raw.groupby('canonical_sku')['v90'].sum().sort_values(ascending=False).tail(5).index.tolist()

sample_20 = top_skus + mid_skus + low_skus + quiet_skus

print(f"Selected 20 SKUs:\n{sample_20}\n")

# Trace predictions on 2026-09-11
day_feat = df_raw[feature_cols].copy()
for col in day_feat.columns:
    if col not in meta['categorical_features']:
        day_feat[col] = pd.to_numeric(day_feat[col], errors='coerce').fillna(0.0)
day_feat['day_of_week'] = 4 # Friday
day_feat['is_weekend'] = 0
raw_p = np.clip(prod_model.predict(day_feat), 0, None)

# Exp6 calibration
z_mask = (df_raw['v7'] == 0) & (df_raw['v14'] == 0) & (df_raw['v30'] == 0)
ev_mask = (df_raw['promo_days_30'] > 0) | (df_raw['amazon_sessions_momentum'] > 1.25)
calib_z = (z_mask & (~ev_mask)).values
calib_stk = ((df_raw['in_stock_flag'] == 0) & (df_raw['has_inventory_signal'] == 1)).values

calib_p = raw_p.copy()
calib_p[calib_z] *= 0.10
calib_p[calib_stk] *= 0.10

df_raw['raw_lgbm_pred'] = raw_p
df_raw['calibrated_pred'] = calib_p

trace_rows = []
for sku in sample_20:
    sub = df_raw[df_raw['canonical_sku'] == sku]
    # Check in Excel for 2026-09-11
    excel_sub = df_f[(df_f['SKU'] == sku) & (df_f['Date'] == '11-Sep-2026')]
    
    amz_sub = sub[sub['platform_group'] == 'Amazon']
    ebay_sub = sub[sub['platform_group'] == 'eBay']
    web_sub = sub[sub['platform_group'] == 'Website']
    oth_sub = sub[sub['platform_group'] == 'Other']

    raw_amz = amz_sub['raw_lgbm_pred'].values[0] if len(amz_sub) > 0 else 0.0
    cal_amz = amz_sub['calibrated_pred'].values[0] if len(amz_sub) > 0 else 0.0
    
    raw_ebay = ebay_sub['raw_lgbm_pred'].values[0] if len(ebay_sub) > 0 else 0.0
    cal_ebay = ebay_sub['calibrated_pred'].values[0] if len(ebay_sub) > 0 else 0.0

    raw_web = web_sub['raw_lgbm_pred'].values[0] if len(web_sub) > 0 else 0.0
    cal_web = web_sub['calibrated_pred'].values[0] if len(web_sub) > 0 else 0.0

    raw_oth = oth_sub['raw_lgbm_pred'].values[0] if len(oth_sub) > 0 else 0.0
    cal_oth = oth_sub['calibrated_pred'].values[0] if len(oth_sub) > 0 else 0.0

    raw_tot = raw_amz + raw_ebay + raw_web + raw_oth
    cal_tot = cal_amz + cal_ebay + cal_web + cal_oth

    excel_amz = excel_sub['Amazon Predicted Units'].values[0] if len(excel_sub) > 0 else 0
    excel_ebay = excel_sub['eBay Predicted Units'].values[0] if len(excel_sub) > 0 else 0
    excel_tot = excel_sub['Total Predicted Units'].values[0] if len(excel_sub) > 0 else 0
    excel_reason = excel_sub['Reason'].values[0] if len(excel_sub) > 0 else ''

    trace_rows.append({
        'SKU': sku,
        'Platforms': len(sub),
        'Raw_LGBM_Sum': round(raw_tot, 4),
        'Calib_Exp6_Sum': round(cal_tot, 4),
        'Excel_Amz_Int': excel_amz,
        'Excel_eBay_Int': excel_ebay,
        'Excel_Total_Int': excel_tot,
        'Reason': excel_reason[:30] + '...'
    })

trace_df = pd.DataFrame(trace_rows)
print(trace_df.to_string(index=False))

# 3. Catalog-Wide Zero Rate Analysis: Continuous Decimal vs Integer Rounding
print("\n" + "=" * 80)
print("3. CATALOG-WIDE ZERO RATE AUDIT")
print("=" * 80)

# Check all 1,413 series on 2026-09-10
print(f"Total active series on 2026-09-10: {len(df_raw)}")
print(f"Series with Raw LightGBM < 0.5: {(df_raw['raw_lgbm_pred'] < 0.5).sum()} ({(df_raw['raw_lgbm_pred'] < 0.5).sum()/len(df_raw)*100:.1f}%)")
print(f"Series with Calibrated Exp6 < 0.5: {(df_raw['calibrated_pred'] < 0.5).sum()} ({(df_raw['calibrated_pred'] < 0.5).sum()/len(df_raw)*100:.1f}%)")
print(f"Series with Calibrated Exp6 == 0.0: {(df_raw['calibrated_pred'] == 0.0).sum()}")

# Check SKU level (summing platforms)
sku_daily_sum = df_raw.groupby('canonical_sku')['calibrated_pred'].sum().reset_index()
print(f"\nSKUs with 1-day Total Decimal Demand < 0.5: {(sku_daily_sum['calibrated_pred'] < 0.5).sum()} out of {len(sku_daily_sum)} ({(sku_daily_sum['calibrated_pred'] < 0.5).sum()/len(sku_daily_sum)*100:.1f}%)")
print(f"SKUs with 1-day Total Decimal Demand == 0.0: {(sku_daily_sum['calibrated_pred'] == 0.0).sum()}")

# Check 10-day sum
sku_10d_sum = sku_daily_sum.copy()
sku_10d_sum['10d_decimal'] = sku_10d_sum['calibrated_pred'] * 10.0
print(f"SKUs with 10-Day Total Decimal Demand < 0.5: {(sku_10d_sum['10d_decimal'] < 0.5).sum()} out of {len(sku_10d_sum)} ({(sku_10d_sum['10d_decimal'] < 0.5).sum()/len(sku_10d_sum)*100:.1f}%)")
print(f"SKUs with 10-Day Total Decimal Demand >= 0.5: {(sku_10d_sum['10d_decimal'] >= 0.5).sum()} out of {len(sku_10d_sum)} ({(sku_10d_sum['10d_decimal'] >= 0.5).sum()/len(sku_10d_sum)*100:.1f}%)")

# Compare total units before vs after integer rounding
tot_pred_decimal_10d = df_raw['calibrated_pred'].sum() * 10.0
tot_pred_excel_10d = df_f['Total Predicted Units'].sum()
print(f"\nTotal 10-Day Forecast Units Before Rounding (Continuous Decimal): {tot_pred_decimal_10d:.1f} units")
print(f"Total 10-Day Forecast Units in Excel (Integer Rounded):            {tot_pred_excel_10d} units")
print(f"Rounding Difference:                                            {tot_pred_excel_10d - tot_pred_decimal_10d:.1f} units")

# Validation totals
wb_val = openpyxl.load_workbook('reports/validation_report_sep_01_to_10_2026.xlsx', data_only=True)
ws_v_summary = wb_val['Summary']
print("\nValidation Totals in Summary Sheet:")
for row in ws_v_summary.iter_rows(values_only=True):
    if row[1] in ['Validation Period', 'Actual Physical Units', 'Predicted Units (Model)', 'Catalog WAPE', 'Net Forecast Bias']:
        print(f"  {row[1]}: {row[2]}")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
