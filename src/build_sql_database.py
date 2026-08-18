"""
Phase 1: Production SQLite Data Pipeline
=========================================
Filter Window: 1 August 2025 to 31 July 2026 (Client Clean Window)
Aggregates all channel sales into 'total_sales' per SKU per date.
Stores clean tables in data/rimmel_clean.db
"""
import os
import sqlite3
import pandas as pd
import numpy as np

BASE_DIR  = r'C:\Users\bhave\Desktop\ml_project'
RAW_EXCEL = os.path.join(BASE_DIR, 'Rimmel Brand Products Sales - 1 Jan 2025 to 31st July 2026.xlsx')
DATA_DIR  = os.path.join(BASE_DIR, 'data')
DB_PATH   = os.path.join(DATA_DIR, 'rimmel_clean.db')

os.makedirs(DATA_DIR, exist_ok=True)

print("="*85)
print("PHASE 1: SQL DATA INGESTION (1 Aug 2025 to 31 Jul 2026)")
print("="*85)

df = pd.read_excel(RAW_EXCEL)
df['date']          = pd.to_datetime(df['date'])
df['sku']           = df['sku'].astype(str).str.strip()
df['category']      = df['category'].fillna('Cosmetics').astype(str).str.strip()
df['sold_quantity'] = pd.to_numeric(df['sold_quantity'], errors='coerce').fillna(0).clip(lower=0)
df['selling_price'] = pd.to_numeric(df['selling_price'], errors='coerce')

# Aggregate across channels (Sum sales per SKU per day)
df_agg = df.groupby(['sku', 'date']).agg(
    total_sales   = ('sold_quantity', 'sum'),
    selling_price = ('selling_price', 'median'),
    category      = ('category', 'first')
).reset_index()

# Forward fill missing prices
df_agg['selling_price'] = df_agg.groupby('sku')['selling_price'].transform(lambda x: x.ffill().bfill())
df_agg['selling_price'] = df_agg['selling_price'].fillna(df_agg['selling_price'].median())

# Filter strictly to client clean window (1 Aug 2025 to 31 Jul 2026)
TRAIN_START = pd.to_datetime('2025-08-01')
TRAIN_END   = pd.to_datetime('2026-07-31')

df_train_win = df_agg[(df_agg['date'] >= TRAIN_START) & (df_agg['date'] <= TRAIN_END)].copy()

# Save to SQLite
if os.path.exists(DB_PATH):
    try:
        os.remove(DB_PATH)
    except PermissionError:
        pass

conn = sqlite3.connect(DB_PATH)
df_agg.to_sql('full_history', conn, if_exists='replace', index=False)
df_train_win.to_sql('training_window', conn, if_exists='replace', index=False)

conn.execute("CREATE INDEX IF NOT EXISTS idx_train_sku_date ON training_window (sku, date)")
conn.commit()
conn.close()

print(f"SQL Ingestion Complete!")
print(f"SQLite DB Location  : {DB_PATH}")
print(f"Training Window     : {TRAIN_START.date()} to {TRAIN_END.date()} ({df_train_win['date'].nunique()} days)")
print(f"Clean Daily Rows    : {len(df_train_win):,}")
print(f"Active SKUs Tracked : {df_train_win['sku'].nunique()}")
print(f"Total Sales Volume  : {df_train_win['total_sales'].sum():,.0f} units")
