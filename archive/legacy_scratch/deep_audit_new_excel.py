"""
DEEP AUDIT SCRIPT FOR: Rimmel Brand Sales Data - 1 Jan 2025 to 10 Sep 2026.xlsx
=============================================================================
Calculates exact facts, distributions, missing counts, mappings, and anomalies.
"""
import os, sys
import pandas as pd
import numpy as np
import openpyxl

EXCEL_PATH = r'C:\Users\bhave\Desktop\ml_project\data\Rimmel Brand Sales Data - 1 Jan 2025 to 10 Sep 2026.xlsx'

print("="*100)
print("1. WORKBOOK STRUCTURE & SHEETS")
print("="*100)
wb = openpyxl.load_workbook(EXCEL_PATH, read_only=True)
sheet_names = wb.sheetnames
print(f"Sheet Names ({len(sheet_names)}): {sheet_names}")
wb.close()

# Load main sheet(s)
for sname in sheet_names:
    print(f"\nLoading Sheet: {sname}...")
    df = pd.read_excel(EXCEL_PATH, sheet_name=sname)
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"Data Types:\n{df.dtypes}\n")

    print("="*100)
    print(f"2. COLUMN-BY-COLUMN AUDIT ({sname})")
    print("="*100)
    
    for col in df.columns:
        s = df[col]
        total_rows = len(s)
        non_null_cnt = s.notnull().sum()
        null_cnt = s.isnull().sum()
        null_pct = (null_cnt / total_rows) * 100.0
        unique_cnt = s.nunique(dropna=True)
        
        # Numbers
        zeros_cnt = 0
        neg_cnt = 0
        min_val = None
        max_val = None
        mean_val = None
        median_val = None
        
        if pd.api.types.is_numeric_dtype(s):
            zeros_cnt = (s == 0).sum()
            neg_cnt = (s < 0).sum()
            min_val = s.min()
            max_val = s.max()
            mean_val = s.mean()
            median_val = s.median()
        elif pd.api.types.is_datetime64_any_dtype(s):
            min_val = s.min()
            max_val = s.max()
        else:
            # Check if text contains dates or numbers
            try:
                s_dt = pd.to_datetime(s.dropna(), errors='coerce')
                if s_dt.notnull().sum() > 0.8 * len(s.dropna()):
                    min_val = s_dt.min()
                    max_val = s_dt.max()
            except:
                pass
                
        # Sample non-null values
        sample_vals = s.dropna().unique()[:5].tolist()
        
        print(f"\n--- Column: '{col}' ---")
        print(f"  Type: {s.dtype} | Total Rows: {total_rows:,} | Non-Null: {non_null_cnt:,} ({100-null_pct:.2f}%) | Null: {null_cnt:,} ({null_pct:.2f}%)")
        print(f"  Unique Values: {unique_cnt:,}")
        if min_val is not None:
            print(f"  Min: {min_val} | Max: {max_val}")
        if mean_val is not None:
            print(f"  Mean: {mean_val:.4f} | Median: {median_val:.4f} | Zeros: {zeros_cnt:,} | Negatives: {neg_cnt:,}")
        print(f"  Sample Values: {sample_vals}")

    print("\n" + "="*100)
    print("3. DATE & TIME-SERIES COVERAGE AUDIT")
    print("="*100)
    if 'date' in [c.lower() for c in df.columns]:
        date_col = [c for c in df.columns if c.lower() == 'date'][0]
        df['parsed_date'] = pd.to_datetime(df[date_col], errors='coerce')
        print(f"Date Column: '{date_col}'")
        print(f"Min Date: {df['parsed_date'].min()} | Max Date: {df['parsed_date'].max()}")
        print(f"Total Unique Dates: {df['parsed_date'].nunique():,}")
        
        date_counts = df['parsed_date'].value_counts().sort_index()
        print(f"Daily Rows Range: Min={date_counts.min()} rows/day, Max={date_counts.max()} rows/day, Mean={date_counts.mean():.1f} rows/day")
        
        # Check full calendar range
        full_cal = pd.date_range(df['parsed_date'].min(), df['parsed_date'].max(), freq='D')
        missing_cal_dates = set(full_cal) - set(df['parsed_date'].dropna().unique())
        print(f"Missing Calendar Dates in timeline: {len(missing_cal_dates)}")
        if missing_cal_dates:
            print(f"  Missing dates sample: {sorted(list(missing_cal_dates))[:10]}")

    print("\n" + "="*100)
    print("4. CHANNEL / PLATFORM ANALYSIS")
    print("="*100)
    channel_col = None
    for c in df.columns:
        if c.lower() in ['channel', 'platform', 'saleschannel', 'channels']:
            channel_col = c
            break
            
    units_col = None
    for c in df.columns:
        if c.lower() in ['units_sold', 'solde_quantity', 'sold_qty', 'units', 'quantity']:
            units_col = c
            break
            
    sku_col = None
    for c in df.columns:
        if c.lower() in ['sku', 'order_item_sku', 'sellersku']:
            sku_col = c
            break

    if channel_col:
        print(f"Channel Column: '{channel_col}'")
        ch_summary = df.groupby(channel_col).agg(
            row_count=(channel_col, 'count'),
            total_units=(units_col, 'sum') if units_col else (channel_col, 'count'),
            unique_skus=(sku_col, 'nunique') if sku_col else (channel_col, 'count'),
            min_date=('parsed_date', 'min') if 'parsed_date' in df.columns else (channel_col, 'min'),
            max_date=('parsed_date', 'max') if 'parsed_date' in df.columns else (channel_col, 'max')
        ).reset_index()
        ch_summary['unit_pct'] = (ch_summary['total_units'] / ch_summary['total_units'].sum()) * 100.0 if units_col else 0
        ch_summary['row_pct'] = (ch_summary['row_count'] / len(df)) * 100.0
        print(ch_summary.to_string(index=False))

    print("\n" + "="*100)
    print("5. IDENTIFIER RELATIONSHIP ANALYSIS (SKU vs actual_sku vs ASIN vs Parent)")
    print("="*100)
    # Check for actual_sku, listing_id, child_asin, parent_id
    cols_lower = {c.lower(): c for c in df.columns}
    print("Detected Identifier Columns:")
    for k in ['sku', 'actual_sku', 'listing_id', 'child_asin', 'parent_asin', 'parent_id', 'product_id', 'ean']:
        if k in cols_lower:
            print(f"  - {k}: '{cols_lower[k]}' (Unique: {df[cols_lower[k]].nunique():,}, Nulls: {df[cols_lower[k]].isnull().sum():,})")

    if 'sku' in cols_lower and 'actual_sku' in cols_lower:
        sku_c = cols_lower['sku']
        act_c = cols_lower['actual_sku']
        sku_to_act = df.groupby(sku_c)[act_c].nunique()
        act_to_sku = df.groupby(act_c)[sku_c].nunique()
        print(f"\nSKU -> actual_sku mapping:")
        print(f"  SKUs mapping to >1 actual_sku: {(sku_to_act > 1).sum()}")
        print(f"  actual_skus mapped by >1 SKU: {(act_to_sku > 1).sum()}")
        
        # Sample of multi-mappings
        multi_skus = act_to_sku[act_to_sku > 1].head(5).index.tolist()
        print(f"  Sample actual_skus with multiple SKUs: {multi_skus}")
        for m in multi_skus:
            print(f"    actual_sku '{m}' has SKUs: {df[df[act_c] == m][sku_c].unique().tolist()}")

    print("\n" + "="*100)
    print("6. INVENTORY / CURRENT_STOCK AUDIT")
    print("="*100)
    stock_col = None
    for c in df.columns:
        if 'stock' in c.lower() or 'inventory' in c.lower():
            stock_col = c
            break
            
    if stock_col:
        print(f"Stock Column: '{stock_col}'")
        print(f"Total rows with stock notnull: {df[stock_col].notnull().sum():,} ({df[stock_col].notnull().mean()*100:.2f}%)")
        print(f"Stock Missing Count: {df[stock_col].isnull().sum():,} ({df[stock_col].isnull().mean()*100:.2f}%)")
        print(f"Stock Min: {df[stock_col].min()} | Max: {df[stock_col].max()} | Zeros: {(df[stock_col] == 0).sum():,}")
        
        if 'parsed_date' in df.columns:
            stock_by_date = df.groupby(df['parsed_date'].dt.to_period('M'))[stock_col].agg(['count', lambda x: x.isnull().sum(), lambda x: x.notnull().mean()*100])
            stock_by_date.columns = ['Total_Rows', 'Missing_Stock_Rows', 'Populated_Stock_Pct']
            print("\nStock Availability by Month:")
            print(stock_by_date.to_string())
            
        if channel_col:
            stock_by_ch = df.groupby(channel_col)[stock_col].agg(['count', lambda x: x.isnull().sum(), lambda x: x.notnull().mean()*100])
            stock_by_ch.columns = ['Total_Rows', 'Missing_Stock_Rows', 'Populated_Stock_Pct']
            print("\nStock Availability by Channel:")
            print(stock_by_ch.to_string())

    print("\n" + "="*100)
    print("7. PACK MULTIPLIER & BUNDLE AUDIT")
    print("="*100)
    pack_col = None
    for c in df.columns:
        if 'pack' in c.lower() or 'multiplier' in c.lower():
            pack_col = c
            break
    if pack_col:
        print(f"Pack Multiplier Column: '{pack_col}'")
        print(df[pack_col].value_counts(dropna=False).to_string())
    else:
        print("No dedicated 'pack_multiplier' column found in raw Excel headers. Checking SKU text for '-2PK', '-3PK', 'Pack of'...")
        if sku_col:
            pack_skus = df[df[sku_col].astype(str).str.contains(r'2PK|3PK|4PK|PACK|BUNDLE', case=False, na=False)][sku_col].unique()
            print(f"Found {len(pack_skus)} SKUs with bundle/pack keywords in SKU name:")
            print(f"  Sample: {list(pack_skus)[:10]}")

    print("\n" + "="*100)
    print("8. CATEGORY & LAUNCH DATE CONSISTENCY")
    print("="*100)
    cat_col = None
    for c in df.columns:
        if 'cat' in c.lower():
            cat_col = c
            break
    if cat_col and sku_col:
        print(f"Category Column: '{cat_col}'")
        cat_by_sku = df.groupby(sku_col)[cat_col].nunique(dropna=True)
        print(f"SKUs with >1 Category string: {(cat_by_sku > 1).sum():,}")
        if (cat_by_sku > 1).sum() > 0:
            sample_cat_conflicts = cat_by_sku[cat_by_sku > 1].head(5).index.tolist()
            for sc in sample_cat_conflicts:
                print(f"  SKU '{sc}' has categories: {df[df[sku_col] == sc][cat_col].unique().tolist()}")

    print("\n" + "="*100)
    print("9. RESTOCK DATE AUDIT")
    print("="*100)
    restock_col = None
    for c in df.columns:
        if 'restock' in c.lower():
            restock_col = c
            break
    if restock_col:
        print(f"Restock Column: '{restock_col}'")
        print(f"Non-null count: {df[restock_col].notnull().sum():,} ({df[restock_col].notnull().mean()*100:.2f}%)")
        print(f"Unique values: {df[restock_col].nunique():,}")
        print(f"Sample values: {df[restock_col].dropna().unique()[:10]}")
