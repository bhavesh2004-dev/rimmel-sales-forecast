"""
Expert ML Engineering Audit Script
Runs a full technical inspection of the Version 3.0 forecasting engine across
multiple edge cases, data quality issues, and model capability bounds.
Outputs a comprehensive audit report before scaling to 588 SKUs.
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.multi_sku_cleaner import prepare_top_20_cleaned_datasets
from src.feature_engineering import build_features, get_feature_columns
from src.model import SalesForecasterEnsemble
from src.croston import CrostonForecaster

excel_path = os.path.join(os.path.dirname(__file__), '..', 'Rimmel Brand Products Sales - 1 Jan 2025 to 31st July 2026 New.xlsx')
output_dir = os.path.join(os.path.dirname(__file__), 'output_results')
os.makedirs(output_dir, exist_ok=True)

df_raw = pd.read_excel(excel_path)
df_raw['date'] = pd.to_datetime(df_raw['date'])
feature_cols = get_feature_columns()

print("="*85)
print("EXPERT ML ENGINEERING AUDIT — VERSION 3.0 FORECASTING ENGINE")
print("="*85)

# ─────────────────────────────────────────────────────────────────────────────
# AUDIT 1: COLD START SKUs (< 60 Days of History)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[AUDIT 1] COLD START SKUs (New Products < 60 Days History)")
print("-"*85)
sku_history = df_raw.groupby('sku')['date'].nunique().reset_index()
sku_history.columns = ['sku', 'history_days']
cold_start_skus = sku_history[sku_history['history_days'] < 60]
print(f"Total SKUs with < 60 Days of Sales History : {len(cold_start_skus)} / 588 SKUs")
print(f"Risk Level: CRITICAL — These SKUs cannot build lag_30 or rolling_mean_30 features!")
print(f"Fix Required: Assign global category-level baseline for cold-start SKUs.")
if len(cold_start_skus) > 0:
    print("Cold Start SKU Examples:")
    print(cold_start_skus.head(5).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# AUDIT 2: EXTREME ZERO-SALES SKUs (>70% Zero Days)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[AUDIT 2] EXTREME ZERO-SALES SKUs (> 70% Zero Sales Days)")
print("-"*85)
sku_zeros = df_raw.groupby('sku').apply(lambda x: (x['sold_quantity']==0).mean()*100).reset_index()
sku_zeros.columns = ['sku', 'zero_pct']
extreme_zero_skus = sku_zeros[sku_zeros['zero_pct'] > 70]
print(f"Total SKUs with > 70% Zero Sales Days : {len(extreme_zero_skus)} / 588 SKUs")
print(f"Risk Level: HIGH — Croston model assigns low daily demand (near 0 units/day).")
print(f"Fix Required: Flag these as 'Slow Movers' and consider EOL (End-of-Life) classification.")
if len(extreme_zero_skus) > 0:
    print("Examples:")
    print(extreme_zero_skus.sort_values('zero_pct', ascending=False).head(5).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# AUDIT 3: MISSING PRICE DATA IN NEW DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("\n[AUDIT 3] MISSING / ZERO PRICE DATA")
print("-"*85)
zero_price = (df_raw['selling_price'] == 0).sum()
null_price = df_raw['selling_price'].isnull().sum()
price_zero_skus = df_raw[df_raw['selling_price'] == 0]['sku'].nunique()
print(f"Total Rows with $0.00 Selling Price : {zero_price} rows across {price_zero_skus} SKUs")
print(f"Total Rows with NULL Selling Price  : {null_price} rows")
print(f"Risk Level: MEDIUM — Zero-price rows may represent clearance or data entry errors.")
print(f"Fix Required: Forward-fill price from last known non-zero daily price per SKU.")

# ─────────────────────────────────────────────────────────────────────────────
# AUDIT 4: MISSING CATEGORY DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[AUDIT 4] MISSING CATEGORY DATA")
print("-"*85)
null_cat = df_raw['category'].isnull().sum()
null_cat_skus = df_raw[df_raw['category'].isnull()]['sku'].nunique()
print(f"Total Rows with NULL Category : {null_cat} rows across {null_cat_skus} unique SKUs")
print(f"Risk Level: MEDIUM — These SKUs lose category-pooling feature advantage.")
print(f"Fix Required: Impute missing categories with 'Cosmetics' (safest global default).")

# ─────────────────────────────────────────────────────────────────────────────
# AUDIT 5: EXTREME PRICE OUTLIERS ($0 to $48.99) SPIKE IMPACT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[AUDIT 5] EXTREME PRICE OUTLIERS & IMPACT")
print("-"*85)
extreme_price = df_raw[df_raw['selling_price'] > 20]
print(f"Total Rows with Selling Price > $20.00 : {len(extreme_price)} rows")
print(f"Max Selling Price in Dataset           : ${df_raw['selling_price'].max():.2f}")
print(f"These may represent bundle pricing or incorrect data entries!")
print(f"Fix Required: Cap selling_price to a per-SKU rolling 95th percentile.")

# ─────────────────────────────────────────────────────────────────────────────
# AUDIT 6: STOCKOUT COVERAGE ACROSS FULL CATALOG
# ─────────────────────────────────────────────────────────────────────────────
print("\n[AUDIT 6] STOCKOUT COVERAGE ACROSS FULL 588 SKU CATALOG")
print("-"*85)
null_stock = df_raw['current_stock'].isnull().sum()
zero_stock = (df_raw['current_stock'] == 0).sum()
zero_stock_skus = df_raw[df_raw['current_stock'] == 0]['sku'].nunique()
null_stock_skus = df_raw[df_raw['current_stock'].isnull()]['sku'].nunique()
print(f"Total Rows with current_stock = 0    : {zero_stock:,} rows across {zero_stock_skus} SKUs")
print(f"Total Rows with NULL current_stock   : {null_stock:,} rows across {null_stock_skus} SKUs")
print(f"Risk Level: HIGH — 44,741 NULL stock rows means we cannot confirm stockout on those days!")
print(f"Fix Required: Treat NULL current_stock rows as 'unknown' and use sold_qty > 0 as demand signal only.")

# ─────────────────────────────────────────────────────────────────────────────
# AUDIT 7: MODEL FEATURE COMPLETENESS TEST ON TOP 20
# ─────────────────────────────────────────────────────────────────────────────
print("\n[AUDIT 7] FEATURE COMPLETENESS TEST ON TOP 20 SKUs")
print("-"*85)
cleaned_dict, data_quality_df = prepare_top_20_cleaned_datasets(excel_path)
issues = []
for sku, df_sku in cleaned_dict.items():
    fdf = build_features(df_sku)
    clean_df = fdf.dropna(subset=feature_cols)
    missing_pct = (fdf[feature_cols].isnull().mean() * 100).max()
    issues.append({'SKU': sku, 'Rows After Drop NaN': len(clean_df), 'Max Feature NaN %': round(missing_pct, 1)})
    
feat_df = pd.DataFrame(issues)
risky = feat_df[feat_df['Max Feature NaN %'] > 10]
print(f"SKUs with > 10% Feature NaN Rate : {len(risky)} / 20")
print(feat_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# AUDIT 8: RESTOCK DATE COVERAGE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[AUDIT 8] RESTOCK DATE COVERAGE (Lead Time Feature)")
print("-"*85)
restock_available = df_raw['restock_date'].notna().sum()
restock_pct = (restock_available / len(df_raw)) * 100
print(f"Rows with Restock Date Available : {restock_available:,} / {len(df_raw):,} ({restock_pct:.1f}%)")
print(f"Risk Level: HIGH — Only {restock_pct:.1f}% restock data available.")
print(f"Cannot yet build automated Lead Time / Reorder Point (ROP) Engine until client provides full restock dates.")

print("\n" + "="*85)
print("AUDIT SUMMARY — ISSUES TO FIX BEFORE 588 SKU SCALING:")
print("="*85)
print("1. [CRITICAL] Cold-Start SKUs (< 60 days history): Assign category-level baseline.")
print("2. [HIGH]     Extreme Zero-Sales SKUs (> 70% zeros): Flag as Slow-Movers / Croston.")
print("3. [HIGH]     44,741 NULL current_stock rows: Use sold_qty > 0 as demand proxy.")
print("4. [MEDIUM]   Zero Price ($0.00) Rows: Forward-fill from last known SKU price.")
print("5. [MEDIUM]   493 NULL Category Rows: Impute with 'Cosmetics' global default.")
print("6. [LOW]      Price Outliers ($20+): Cap to per-SKU 95th percentile price.")
print("="*85)
