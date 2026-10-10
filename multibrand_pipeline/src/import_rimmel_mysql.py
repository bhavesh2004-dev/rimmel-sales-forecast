"""
Traceable Rimmel Sales Importer for MySQL.
Reads authoritative raw Rimmel sales Excel file and loads records into raw_rimmel_sales_data.
Creates audit record in import_batches table.
"""

import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Optional
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Ensure import paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from multibrand_pipeline.src.db_manager import DBManager

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
RIMMEL_EXCEL_PATH = os.path.join(PROJECT_ROOT, "data", "Rimmel Brand Sales Data - 1 Jan 2025 to 10 Sep 2026.xlsx")
BATCH_ID = "BATCH-RIMMEL-RAW-001"

def import_rimmel_data(excel_path: Optional[str] = None, batch_id: str = BATCH_ID):
    if excel_path is None:
        excel_path = RIMMEL_EXCEL_PATH
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Rimmel sales file not found at {excel_path}")

    logger.info(f"Loading authoritative Rimmel Excel file from: {excel_path}")
    df = pd.read_excel(excel_path)
    logger.info(f"Loaded {len(df):,} raw rows from Excel.")

    # Validation checks
    assert len(df) == 101085, f"Expected exactly 101,085 rows, found {len(df):,}"
    
    # Check that all SKUs belong to Rimmel
    non_rimmel_skus = df[~df['sku'].astype(str).str.upper().str.startswith('RIM-')]
    if len(non_rimmel_skus) > 0:
        raise ValueError(f"Found {len(non_rimmel_skus)} non-Rimmel SKUs in Rimmel sales data!")

    logger.info("Validation passed: All 101,085 rows confirmed as Rimmel brand records.")

    # Prepare DataFrame for raw_rimmel_sales_data table
    # Expected columns: batch_id, date, sku, category, channel, listing_id, parent_id, actual_sku,
    # pack_multiplier, units_sold, orders_count, selling_price, launch_date, current_stock,
    # child_asin, buy_box_percentage, amazon_sessions, fulfillment_type, ebay_promoted_flag, restock_date
    df_clean = pd.DataFrame(index=df.index)
    df_clean['batch_id'] = batch_id
    df_clean['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    df_clean['sku'] = df['sku'].astype(str).str.strip()
    df_clean['category'] = df['category'].fillna('').astype(str)
    df_clean['channel'] = df['channel'].astype(str).str.strip()
    df_clean['listing_id'] = df['listing_id'].apply(lambda x: None if pd.isna(x) else str(x))
    df_clean['parent_id'] = df['parent_id'].apply(lambda x: None if pd.isna(x) else str(x))
    df_clean['actual_sku'] = df['actual_sku'].apply(lambda x: None if pd.isna(x) else str(x))
    df_clean['pack_multiplier'] = pd.to_numeric(df['pack_multiplier'], errors='coerce').fillna(1).astype(int)
    df_clean['units_sold'] = pd.to_numeric(df['units_sold'], errors='coerce').fillna(0).astype(int)
    df_clean['orders_count'] = pd.to_numeric(df['orders_count'], errors='coerce').fillna(1).astype(int)
    df_clean['selling_price'] = pd.to_numeric(df['selling_price'], errors='coerce').fillna(0.0).round(2)
    df_clean['launch_date'] = pd.to_datetime(df['launch_date'], errors='coerce').dt.strftime('%Y-%m-%d')
    df_clean['current_stock'] = pd.to_numeric(df['current_stock'], errors='coerce').apply(lambda x: None if pd.isna(x) else int(x))
    df_clean['child_asin'] = df['child_asin'].apply(lambda x: None if pd.isna(x) else str(x))
    df_clean['buy_box_percentage'] = pd.to_numeric(df['buy_box_percentage'], errors='coerce').apply(lambda x: None if pd.isna(x) else round(float(x), 4))
    df_clean['amazon_sessions'] = df['amazon_sessions'].apply(lambda x: None if pd.isna(x) else str(x))
    df_clean['fulfillment_type'] = df['fulfillment_type'].apply(lambda x: None if pd.isna(x) else str(x))
    df_clean['ebay_promoted_flag'] = df['ebay_promoted_flag'].apply(lambda x: '1' if str(x) in ['1', '1.0'] else ('0' if str(x) in ['0', '0.0'] else None))
    df_clean['restock_date'] = pd.to_datetime(df['restock_date'], errors='coerce').dt.strftime('%Y-%m-%d')

    min_date = df_clean['date'].min()
    max_date = df_clean['date'].max()

    db = DBManager()
    with db.engine.connect() as conn:
        # Check if already imported
        existing = conn.execute(text("SELECT COUNT(*) FROM raw_rimmel_sales_data WHERE batch_id = :b"), {"b": batch_id}).fetchone()[0]
        if existing == len(df_clean):
            logger.info(f"Batch {batch_id} already fully loaded and verified in raw_rimmel_sales_data ({existing:,} rows). Skipping redundant re-insert.")
            return existing
        elif existing > 0:
            logger.info(f"Batch {batch_id} partially exists ({existing:,} rows). Re-loading clean batch records...")
            conn.execute(text("DELETE FROM raw_rimmel_sales_data WHERE batch_id = :b"), {"b": batch_id})
            conn.commit()

        # Insert batch tracker
        conn.execute(text("""
            INSERT INTO import_batches (batch_id, source_name, source_type, brand_id, row_count, min_date, max_date, status)
            VALUES (:batch_id, :source_name, 'EXCEL', 'RIMMEL', :row_count, :min_date, :max_date, 'PENDING')
            ON DUPLICATE KEY UPDATE status = 'PENDING', row_count = :row_count, min_date = :min_date, max_date = :max_date
        """), {
            "batch_id": batch_id,
            "source_name": os.path.basename(excel_path),
            "row_count": len(df_clean),
            "min_date": min_date,
            "max_date": max_date
        })
        conn.commit()

    logger.info(f"Inserting {len(df_clean):,} records into raw_rimmel_sales_data in chunks of 10,000...")
    chunk_size = 10000
    total_chunks = (len(df_clean) + chunk_size - 1) // chunk_size

    for i in range(total_chunks):
        chunk = df_clean.iloc[i*chunk_size : (i+1)*chunk_size]
        chunk.to_sql('raw_rimmel_sales_data', con=db.engine, if_exists='append', index=False, method='multi')
        logger.info(f"  Inserted chunk {i+1}/{total_chunks} ({len(chunk):,} rows)")

    with db.engine.connect() as conn:
        conn.execute(text("UPDATE import_batches SET status = 'LOADED' WHERE batch_id = :b"), {"b": batch_id})
        conn.commit()
        cnt = conn.execute(text("SELECT COUNT(*) FROM raw_rimmel_sales_data WHERE batch_id = :b"), {"b": batch_id}).fetchone()[0]

    logger.info(f"Rimmel sales import completed successfully! Total rows in raw_rimmel_sales_data: {cnt:,}")
    return cnt

if __name__ == "__main__":
    import_rimmel_data()
