"""
DETAILED METRIC EXTRACTION SCRIPT (ROBUST TYPES)
================================================
"""
import os, sys
import pandas as pd
import numpy as np

EXCEL_PATH = r'C:\Users\bhave\Desktop\ml_project\data\Rimmel Brand Sales Data - 1 Jan 2025 to 10 Sep 2026.xlsx'

df = pd.read_excel(EXCEL_PATH)

print(f"Total Rows: {len(df):,}")
print(f"Total Columns: {len(df.columns)}")
print(f"Columns: {list(df.columns)}\n")

col_summary = []
for c in df.columns:
    s = df[c]
    t_cnt = len(s)
    not_null = int(s.notnull().sum())
    null_cnt = int(s.isnull().sum())
    null_pct = round((null_cnt / t_cnt) * 100.0, 2)
    uniq = int(s.nunique(dropna=True))
    
    zeros = 0
    negs = 0
    min_v = 'N/A'
    max_v = 'N/A'
    mean_v = 'N/A'
    median_v = 'N/A'
    
    # Try numeric conversion
    s_num = pd.to_numeric(s, errors='coerce')
    if s_num.notnull().sum() > 0.5 * not_null:
        zeros = int((s_num == 0).sum())
        negs = int((s_num < 0).sum())
        min_v = round(float(s_num.min()), 2)
        max_v = round(float(s_num.max()), 2)
        mean_v = round(float(s_num.mean()), 2)
        median_v = round(float(s_num.median()), 2)
    elif pd.api.types.is_datetime64_any_dtype(s) or 'date' in c.lower():
        s_dt = pd.to_datetime(s, errors='coerce')
        if s_dt.notnull().sum() > 0:
            min_v = s_dt.min().strftime('%Y-%m-%d')
            max_v = s_dt.max().strftime('%Y-%m-%d')
    else:
        sample_non_null = s.dropna().astype(str)
        if len(sample_non_null) > 0:
            min_v = str(sample_non_null.iloc[0])
            max_v = str(sample_non_null.iloc[-1])
            
    col_summary.append({
        'Column': c,
        'Dtype': str(s.dtype),
        'Total Rows': t_cnt,
        'Non-Null': not_null,
        'Null Count': null_cnt,
        'Null %': null_pct,
        'Unique': uniq,
        'Zeros': zeros,
        'Negatives': negs,
        'Min': min_v,
        'Max': max_v,
        'Mean': mean_v,
        'Median': median_v
    })

df_col_sum = pd.DataFrame(col_summary)
print("="*120)
print("EXACT COLUMN-BY-COLUMN SUMMARY TABLE:")
print("="*120)
print(df_col_sum.to_string(index=False))

# Date parsing
df['date_parsed'] = pd.to_datetime(df['date'])
print(f"\nDate Range: {df['date_parsed'].min().strftime('%Y-%m-%d')} to {df['date_parsed'].max().strftime('%Y-%m-%d')} ({df['date_parsed'].nunique()} unique dates)")

# Channel Breakdown
print("\n" + "="*120)
print("CHANNEL / PLATFORM BREAKDOWN:")
print("="*120)
ch_df = df.groupby('channel').agg(
    Rows=('channel', 'count'),
    Units=('units_sold', 'sum'),
    Orders=('orders_count', 'sum'),
    SKUs=('sku', 'nunique'),
    Min_Date=('date_parsed', 'min'),
    Max_Date=('date_parsed', 'max')
).reset_index()
ch_df['Unit_%'] = round((ch_df['Units'] / ch_df['Units'].sum()) * 100.0, 2)
ch_df['Row_%'] = round((ch_df['Rows'] / len(df)) * 100.0, 2)
print(ch_df.to_string(index=False))

# Platform Specific Fields Breakdown
print("\n" + "="*120)
print("PLATFORM-SPECIFIC FIELDS AUDIT:")
print("="*120)
for pf_col in ['child_asin', 'buy_box_percentage', 'amazon_sessions', 'fulfillment_type', 'ebay_promoted_flag']:
    print(f"\nField: '{pf_col}'")
    for ch, grp in df.groupby('channel'):
        non_null_cnt = grp[pf_col].notnull().sum()
        pct = (non_null_cnt / len(grp)) * 100.0
        print(f"  Channel: {ch:28s} | Populated: {non_null_cnt:,} / {len(grp):,} ({pct:6.2f}%)")

# Identifiers Breakdown
print("\n" + "="*120)
print("IDENTIFIERS RELATIONSHIPS:")
print("="*120)
print(f"Total Unique 'sku': {df['sku'].nunique():,}")
print(f"Total Unique 'actual_sku': {df['actual_sku'].nunique(dropna=True):,}")
print(f"Total Unique 'listing_id': {df['listing_id'].nunique(dropna=True):,}")
print(f"Total Unique 'child_asin': {df['child_asin'].nunique(dropna=True):,}")
print(f"Total Unique 'parent_id': {df['parent_id'].nunique(dropna=True):,}")

# Pack Multipliers
print("\n" + "="*120)
print("PACK MULTIPLIER DISTRIBUTION:")
print("="*120)
print(df['pack_multiplier'].value_counts(dropna=False))

# Check units vs orders vs pack_multiplier
print("\n" + "="*120)
print("UNITS vs ORDERS vs PACK MULTIPLIER MATH:")
print("="*120)
df['implied_units'] = df['orders_count'] * df['pack_multiplier']
print(f"Rows where units_sold == orders_count: {(df['units_sold'] == df['orders_count']).sum():,} ({(df['units_sold'] == df['orders_count']).mean()*100:.2f}%)")
print(f"Rows where units_sold == implied_units (orders * pack_multiplier): {(df['units_sold'] == df['implied_units']).sum():,} ({(df['units_sold'] == df['implied_units']).mean()*100:.2f}%)")
print(f"Rows where units_sold != implied_units: {(df['units_sold'] != df['implied_units']).sum():,}")
diff_sample = df[df['units_sold'] != df['implied_units']][['sku', 'pack_multiplier', 'orders_count', 'units_sold', 'selling_price', 'channel']].head(10)
print(f"Sample of units != orders * pack_multiplier:\n{diff_sample.to_string(index=False)}")

# Check duplicate business keys
print("\n" + "="*120)
print("DUPLICATES AUDIT:")
print("="*120)
print(f"Exact Duplicate Rows: {df.duplicated().sum():,}")
print(f"Duplicate (date, sku, channel): {df.duplicated(subset=['date', 'sku', 'channel']).sum():,}")
print(f"Duplicate (date, listing_id): {df.dropna(subset=['listing_id']).duplicated(subset=['date', 'listing_id']).sum():,}")

# Category and Launch Date consistency
print("\n" + "="*120)
print("CATEGORY & LAUNCH DATE CONSISTENCY:")
print("="*120)
cat_per_sku = df.groupby('sku')['category'].nunique()
print(f"SKUs with multiple categories: {(cat_per_sku > 1).sum():,}")
launch_per_sku = df.groupby('sku')['launch_date'].nunique()
print(f"SKUs with multiple launch dates: {(launch_per_sku > 1).sum():,}")
if (launch_per_sku > 1).sum() > 0:
    print(f"Sample SKUs with multiple launch dates: {launch_per_sku[launch_per_sku > 1].head(5).to_dict()}")

# Restock Date Audit
print("\n" + "="*120)
print("RESTOCK DATE AUDIT:")
print("="*120)
print(f"Populated restock dates: {df['restock_date'].notnull().sum():,} ({df['restock_date'].notnull().mean()*100:.2f}%)")
print(f"Unique restock dates: {df['restock_date'].nunique():,}")
print(f"Sample restock dates: {df['restock_date'].dropna().unique()}")
