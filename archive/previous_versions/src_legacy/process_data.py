"""
DATA PROCESSING PIPELINE
=========================
Reads raw Rimmel Excel (multiple rows per SKU per day = one per channel)
Aggregates all channel sales into one clean row per SKU per date
Saves to:
  1. SQLite database  -> data/rimmel_clean.db  (table: daily_sales)
  2. Clean Excel file -> data/rimmel_clean_processed.xlsx

Training window: 1 Aug 2025 to 31 Jul 2026 (client-confirmed period)
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = r'C:\Users\bhave\Desktop\ml_project'
RAW_EXCEL  = os.path.join(BASE_DIR, 'Rimmel Brand Products Sales - 1 Jan 2025 to 31st July 2026 New.xlsx')
DATA_DIR   = os.path.join(BASE_DIR, 'data')
DB_PATH    = os.path.join(DATA_DIR, 'rimmel_clean.db')
CLEAN_XL   = os.path.join(DATA_DIR, 'rimmel_clean_processed.xlsx')

os.makedirs(DATA_DIR, exist_ok=True)

# ── STEP 1: Load Raw Excel ─────────────────────────────────────────────────────
print("=" * 75)
print("STEP 1 — Loading raw Excel file...")
print("=" * 75)

df = pd.read_excel(RAW_EXCEL)
df['date']          = pd.to_datetime(df['date'])
df['sku']           = df['sku'].astype(str).str.strip()
df['title']         = df['title'].astype(str).str.strip()
df['brand']         = df['brand'].astype(str).str.strip()
df['category']      = df['category'].fillna('Uncategorised').astype(str).str.strip()
df['channel']       = df['channel'].astype(str).str.strip()
df['sold_quantity'] = pd.to_numeric(df['sold_quantity'], errors='coerce').fillna(0).clip(lower=0)
df['selling_price'] = pd.to_numeric(df['selling_price'], errors='coerce')
df['current_stock'] = pd.to_numeric(df['current_stock'], errors='coerce')
df['launch_date']   = pd.to_datetime(df['launch_date'], errors='coerce')

print(f"  Raw rows loaded       : {len(df):,}")
print(f"  Date range            : {df['date'].min().date()}  to  {df['date'].max().date()}")
print(f"  Unique SKUs           : {df['sku'].nunique()}")
print(f"  Unique channels       : {df['channel'].nunique()}")
print(f"  Sample channels       : {sorted(df['channel'].unique())[:6]}")

# ── STEP 2: Aggregate Channels → One Row Per (SKU, Date) ──────────────────────
print("\n" + "=" * 75)
print("STEP 2 — Aggregating all channels into one row per SKU per date...")
print("=" * 75)

# For each (sku, date):
#   total_sales  = SUM of sold_quantity across ALL channels
#   selling_price = MEDIAN price (handles different channel prices fairly)
#   channel_count = how many channels sold that day
#   channels      = comma-joined list of active channels
#   current_stock = LAST non-null stock value reported
#   category, title, brand, launch_date = first non-null value (same across channels)

df_clean = df.groupby(['sku', 'date']).agg(
    total_sales    = ('sold_quantity',  'sum'),
    selling_price  = ('selling_price',  'median'),
    channel_count  = ('channel',        'nunique'),
    channels       = ('channel',        lambda x: ', '.join(sorted(x.dropna().unique()))),
    current_stock  = ('current_stock',  lambda x: x.dropna().iloc[-1] if x.dropna().any() else np.nan),
    category       = ('category',       'first'),
    title          = ('title',          'first'),
    brand          = ('brand',          'first'),
    launch_date    = ('launch_date',    'first'),
).reset_index()

# Sort properly
df_clean = df_clean.sort_values(['sku', 'date']).reset_index(drop=True)

print(f"  Aggregated rows       : {len(df_clean):,}  (was {len(df):,} raw rows)")
print(f"  Unique (SKU, date)    : {len(df_clean):,}")
print(f"  SKUs present          : {df_clean['sku'].nunique()}")

# ── STEP 3: Fix Missing Prices (forward fill per SKU) ─────────────────────────
print("\n" + "=" * 75)
print("STEP 3 — Fixing missing / zero prices (forward-fill within SKU)...")
print("=" * 75)

df_clean['selling_price'] = df_clean['selling_price'].replace(0, np.nan)
df_clean['selling_price'] = df_clean.groupby('sku')['selling_price'].transform(
    lambda x: x.ffill().bfill()
)
global_median_price = df_clean['selling_price'].median()
df_clean['selling_price'] = df_clean['selling_price'].fillna(global_median_price)

print(f"  Null prices remaining : {df_clean['selling_price'].isna().sum()}")
print(f"  Price range           : ${df_clean['selling_price'].min():.2f}  to  ${df_clean['selling_price'].max():.2f}")

# ── STEP 4: Filter to Training Window ─────────────────────────────────────────
print("\n" + "=" * 75)
print("STEP 4 — Filtering to client training window (1 Aug 2025 - 31 Jul 2026)...")
print("=" * 75)

df_full = df_clean.copy()   # keep full history for reference

df_train = df_clean[
    (df_clean['date'] >= '2025-08-01') &
    (df_clean['date'] <= '2026-07-31')
].copy()

print(f"  Full dataset rows     : {len(df_full):,}")
print(f"  Training window rows  : {len(df_train):,}")
print(f"  SKUs in training      : {df_train['sku'].nunique()}")
print(f"  Total units (train)   : {df_train['total_sales'].sum():,.0f}")
print(f"  Zero-sales rows       : {(df_train['total_sales']==0).sum():,}  "
      f"({(df_train['total_sales']==0).mean()*100:.1f}%)")

# ── STEP 5: Quality Report ─────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("STEP 5 — Data quality report...")
print("=" * 75)

sku_summary = df_train.groupby('sku').agg(
    days_present    = ('date',        'count'),
    days_with_sales = ('total_sales', lambda x: (x > 0).sum()),
    total_units     = ('total_sales', 'sum'),
    avg_daily_sales = ('total_sales', 'mean'),
    max_daily_sales = ('total_sales', 'max'),
    avg_price       = ('selling_price','mean'),
    category        = ('category',    'first'),
    title           = ('title',       'first'),
).reset_index()

sku_summary['zero_pct'] = 1 - sku_summary['days_with_sales'] / sku_summary['days_present']
sku_summary['tier'] = pd.cut(
    sku_summary['total_units'],
    bins=[-1, 0, 100, 1000, 5000, float('inf')],
    labels=['Dead (0 units)', 'Slow (<100 units)', 'Low (100-1k)', 'Mid (1k-5k)', 'High (5k+)']
)

print("\nSKU Demand Tiers in Training Window:")
print(sku_summary['tier'].value_counts().to_string())

print(f"\nTop 15 SKUs by Total Sales:")
top15 = sku_summary.sort_values('total_units', ascending=False).head(15)
print(top15[['sku','total_units','days_with_sales','avg_daily_sales','zero_pct','category']].to_string(index=False))

# ── STEP 6: Save to SQLite ─────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("STEP 6 — Saving to SQLite database...")
print("=" * 75)

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)

# Table 1: full_history (Jan 2025 - Jul 2026)
df_full.to_sql('full_history', conn, if_exists='replace', index=False)

# Table 2: training_window (Aug 2025 - Jul 2026) — what model will use
df_train.to_sql('training_window', conn, if_exists='replace', index=False)

# Table 3: sku_summary — metadata + demand profile per SKU
sku_summary.to_sql('sku_summary', conn, if_exists='replace', index=False)

# Create useful indexes for fast querying
conn.execute("CREATE INDEX IF NOT EXISTS idx_train_sku_date ON training_window (sku, date)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_full_sku_date  ON full_history    (sku, date)")
conn.commit()

# Verify by reading back
check_train = pd.read_sql("SELECT COUNT(*) as rows, COUNT(DISTINCT sku) as skus FROM training_window", conn)
check_full  = pd.read_sql("SELECT COUNT(*) as rows, COUNT(DISTINCT sku) as skus FROM full_history",    conn)
check_smry  = pd.read_sql("SELECT COUNT(*) as skus FROM sku_summary",                                  conn)

print(f"  DB saved to           : {DB_PATH}")
print(f"  Table 'full_history'  : {check_full.iloc[0]['rows']:,} rows, {check_full.iloc[0]['skus']} SKUs")
print(f"  Table 'training_window': {check_train.iloc[0]['rows']:,} rows, {check_train.iloc[0]['skus']} SKUs")
print(f"  Table 'sku_summary'   : {check_smry.iloc[0]['skus']} SKUs")
conn.close()

# ── STEP 7: Save to Clean Excel ───────────────────────────────────────────────
print("\n" + "=" * 75)
print("STEP 7 — Saving clean processed Excel file...")
print("=" * 75)

with pd.ExcelWriter(CLEAN_XL, engine='xlsxwriter') as writer:
    # Sheet 1: Training Window (Aug 2025 - Jul 2026) — Main sheet for model
    df_train.to_excel(writer, sheet_name='Training_Window_Aug25_Jul26', index=False)

    # Sheet 2: SKU Summary
    sku_summary.to_excel(writer, sheet_name='SKU_Summary', index=False)

    # Sheet 3: Full History (Jan 2025 - Jul 2026)
    df_full.to_excel(writer, sheet_name='Full_History_Jan25_Jul26', index=False)

    # Format headers
    wb = writer.book
    hdr_fmt = wb.add_format({'bold': True, 'bg_color': '#1a73e8', 'font_color': '#ffffff', 'border': 1})
    for sheet_name in writer.sheets:
        ws = writer.sheets[sheet_name]
        if sheet_name == 'Training_Window_Aug25_Jul26':
            for i, col in enumerate(df_train.columns):
                ws.write(0, i, col, hdr_fmt)
            ws.set_column('A:A', 22)   # sku
            ws.set_column('B:B', 14)   # date
            ws.set_column('C:C', 14)   # total_sales
            ws.set_column('D:D', 14)   # selling_price
            ws.set_column('E:F', 14)
            ws.set_column('G:G', 40)   # channels
            ws.freeze_panes(1, 0)

print(f"  Excel saved to        : {CLEAN_XL}")
print(f"  Sheet 1 (model input) : Training_Window_Aug25_Jul26  ({len(df_train):,} rows)")
print(f"  Sheet 2               : SKU_Summary                  ({len(sku_summary)} SKUs)")
print(f"  Sheet 3               : Full_History_Jan25_Jul26     ({len(df_full):,} rows)")

# ── FINAL SUMMARY ──────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("DATA PROCESSING COMPLETE — READY FOR MODEL TRAINING")
print("=" * 75)
print(f"""
  Source file   : Rimmel Brand Products Sales - 1 Jan 2025 to 31st July 2026 New.xlsx
  Raw rows      : {len(df):,}  (multiple rows per SKU per day — one per channel)
  
  After processing:
    Clean rows (Aug 25-Jul 26) : {len(df_train):,}  (one row per SKU per date)
    Total SKUs                 : {df_train['sku'].nunique()}
    Date range                 : 2025-08-01 to 2026-07-31  (365 days)
    Total units in window      : {df_train['total_sales'].sum():,.0f}
    Column 'total_sales'       : SUM of all channel sales per SKU per day

  Outputs:
    SQLite DB     : data/rimmel_clean.db
    Clean Excel   : data/rimmel_clean_processed.xlsx

  Use 'training_window' table (or Sheet 1) to train the model.
""")
