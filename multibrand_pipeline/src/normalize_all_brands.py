"""
Multi-Brand Normalization & Ingestion Orchestration for MySQL.
Reads from order_sales_data (6 brands) and raw_rimmel_sales_data (Rimmel),
normalizes records to standard unified schema, and populates:
- normalized_sales
- sku_master
- brand_registry
- import_batches
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from multibrand_pipeline.src.db_manager import DBManager
from multibrand_pipeline.src.normalization import (
    normalize_brand_name,
    map_platform,
    resolve_canonical_sku,
    clean_monetary_value
)

PLATFORM_MAPPING = {
    'Amazon': ['amazon', 'amz', 'fba', 'fbm', 'prime'],
    'eBay': ['ebay'],
    'Website': ['website', 'shopify', 'web', 'magento', 'direct', 'woo', 'store'],
    'Other': ['tiktok', 'b2b', 'wholesale', 'manual', 'other']
}

def normalize_and_load():
    db = DBManager()
    
    with db.engine.connect() as conn:
        norm_count = conn.execute(text("SELECT COUNT(*) FROM normalized_sales")).fetchone()[0]
        if norm_count == 134901:
            logger.info(f"normalized_sales already fully populated with {norm_count:,} rows. Skipping reload.")
            return norm_count
            
    # 1. Register Batch for existing order_sales_data
    with db.engine.connect() as conn:
        osd_stats = conn.execute(text("SELECT COUNT(*), MIN(`date`), MAX(`date`) FROM order_sales_data")).fetchone()
        conn.execute(text("""
            INSERT INTO import_batches (batch_id, source_name, source_type, brand_id, row_count, min_date, max_date, status)
            VALUES ('BATCH-OSD-DEV-001', 'order_sales_data', 'MYSQL_TABLE', 'MULTI_BRAND', :cnt, :min_d, :max_d, 'LOADED')
            ON DUPLICATE KEY UPDATE row_count = :cnt, min_date = :min_d, max_date = :max_d, status = 'LOADED'
        """), {"cnt": osd_stats[0], "min_d": str(osd_stats[1]), "max_d": str(osd_stats[2])})
        conn.commit()
    logger.info(f"Registered import batch BATCH-OSD-DEV-001 ({osd_stats[0]:,} rows).")

    # 2. Extract from order_sales_data
    logger.info("Reading records from order_sales_data...")
    with db.engine.connect() as conn:
        df_osd = pd.read_sql("SELECT * FROM order_sales_data", conn)
    logger.info(f"Loaded {len(df_osd):,} records from order_sales_data.")

    # 3. Extract from raw_rimmel_sales_data
    logger.info("Reading records from raw_rimmel_sales_data...")
    with db.engine.connect() as conn:
        df_rimmel = pd.read_sql("SELECT * FROM raw_rimmel_sales_data", conn)
    logger.info(f"Loaded {len(df_rimmel):,} records from raw_rimmel_sales_data.")

    # 4. Process order_sales_data
    rows_norm = []
    
    for _, r in df_osd.iterrows():
        b_id, disp_name = normalize_brand_name(r['brand'])
        c_sku = resolve_canonical_sku(str(r['sku']), r['actual_sku'] if pd.notna(r['actual_sku']) else None)
        plat = map_platform(r['channel'], PLATFORM_MAPPING)
        pack_mult = int(r['pack_multiplier']) if pd.notna(r['pack_multiplier']) and int(r['pack_multiplier']) > 0 else 1
        qty = int(r['quantity']) if pd.notna(r['quantity']) else 0
        units_sold = qty * pack_mult
        orders_cnt = int(r['orders_count']) if pd.notna(r['orders_count']) and int(r['orders_count']) > 0 else 1
        price = clean_monetary_value(r.get('avg_item_sales_price', 0.0))
        stock = int(r['current_stock']) if pd.notna(r['current_stock']) else None
        
        rows_norm.append({
            'source_table': 'order_sales_data',
            'source_row_id': int(r['id']),
            'batch_id': 'BATCH-OSD-DEV-001',
            'date': str(r['date']),
            'brand_id': b_id,
            'display_brand_name': disp_name,
            'canonical_sku': c_sku,
            'raw_sku': str(r['sku']).strip(),
            'platform_group': plat,
            'raw_channel': str(r['channel']).strip(),
            'units_sold': units_sold,
            'orders_count': orders_cnt,
            'pack_multiplier': pack_mult,
            'selling_price': price,
            'current_stock': stock,
            'category': str(r['category']) if pd.notna(r['category']) else 'Cosmetics General'
        })

    # 5. Process raw_rimmel_sales_data
    for _, r in df_rimmel.iterrows():
        b_id, disp_name = ('RIMMEL', 'Rimmel')
        c_sku = resolve_canonical_sku(str(r['sku']), r['actual_sku'] if pd.notna(r['actual_sku']) else None)
        plat = map_platform(r['channel'], PLATFORM_MAPPING)
        pack_mult = int(r['pack_multiplier']) if pd.notna(r['pack_multiplier']) and int(r['pack_multiplier']) > 0 else 1
        qty = int(r['units_sold']) if pd.notna(r['units_sold']) else 0
        units_sold = qty * pack_mult
        orders_cnt = int(r['orders_count']) if pd.notna(r['orders_count']) and int(r['orders_count']) > 0 else 1
        price = clean_monetary_value(r.get('selling_price', 0.0))
        stock = int(r['current_stock']) if pd.notna(r['current_stock']) else None

        rows_norm.append({
            'source_table': 'raw_rimmel_sales_data',
            'source_row_id': int(r['id']),
            'batch_id': str(r['batch_id']),
            'date': str(r['date']),
            'brand_id': b_id,
            'display_brand_name': disp_name,
            'canonical_sku': c_sku,
            'raw_sku': str(r['sku']).strip(),
            'platform_group': plat,
            'raw_channel': str(r['channel']).strip(),
            'units_sold': units_sold,
            'orders_count': orders_cnt,
            'pack_multiplier': pack_mult,
            'selling_price': price,
            'current_stock': stock,
            'category': str(r['category']) if pd.notna(r['category']) else 'Cosmetics General'
        })

    df_unified = pd.DataFrame(rows_norm)
    logger.info(f"Unified normalized dataset prepared: {len(df_unified):,} total records.")
    
    # Brand breakdown check
    brand_counts = df_unified['display_brand_name'].value_counts()
    logger.info("Unified Brand Distribution:\n" + brand_counts.to_string())

    # 6. Insert into normalized_sales in chunks
    with db.engine.connect() as conn:
        logger.info("Clearing existing records in normalized_sales...")
        conn.execute(text("DELETE FROM normalized_sales"))
        conn.commit()

    chunk_size = 15000
    total_chunks = (len(df_unified) + chunk_size - 1) // chunk_size
    logger.info(f"Loading {len(df_unified):,} records into normalized_sales in {total_chunks} chunks...")

    for i in range(total_chunks):
        chunk = df_unified.iloc[i*chunk_size : (i+1)*chunk_size]
        chunk.to_sql('normalized_sales', con=db.engine, if_exists='append', index=False, method='multi')
        logger.info(f"  Inserted chunk {i+1}/{total_chunks} ({len(chunk):,} rows)")

    # 7. Populate brand_registry
    logger.info("Populating brand_registry...")
    brand_summary = df_unified.groupby(['brand_id', 'display_brand_name']).agg(
        first_date=('date', 'min'),
        last_date=('date', 'max'),
        total_skus=('canonical_sku', 'nunique')
    ).reset_index()

    with db.engine.connect() as conn:
        for _, b in brand_summary.iterrows():
            status = 'ACTIVE' if b['total_skus'] >= 20 else 'COLD_START'
            conn.execute(text("""
                INSERT INTO brand_registry (brand_id, display_name, status, first_observed_date, last_observed_date, total_skus)
                VALUES (:bid, :dname, :stat, :fdate, :ldate, :skus)
                ON DUPLICATE KEY UPDATE
                    display_name = :dname,
                    status = :stat,
                    first_observed_date = :fdate,
                    last_observed_date = :ldate,
                    total_skus = :skus
            """), {
                "bid": b['brand_id'],
                "dname": b['display_brand_name'],
                "stat": status,
                "fdate": b['first_date'],
                "ldate": b['last_date'],
                "skus": int(b['total_skus'])
            })
        conn.commit()

    # 8. Populate sku_master
    logger.info("Populating sku_master...")
    sku_agg = df_unified.groupby('canonical_sku').agg(
        brand_id=('brand_id', 'first'),
        category=('category', 'first'),
        current_stock=('current_stock', 'last')
    ).reset_index()

    sku_records = []
    for _, s in sku_agg.iterrows():
        stock_val = int(s['current_stock']) if pd.notna(s['current_stock']) else 0
        sku_records.append({
            'canonical_sku': s['canonical_sku'],
            'brand_id': s['brand_id'],
            'product_title': s['canonical_sku'],
            'category': s['category'],
            'current_stock': stock_val,
            'is_active': 1
        })
    df_skus = pd.DataFrame(sku_records)
    
    with db.engine.connect() as conn:
        conn.execute(text("DELETE FROM sku_master"))
        conn.commit()

    df_skus.to_sql('sku_master', con=db.engine, if_exists='append', index=False, method='multi')
    logger.info(f"Loaded {len(df_skus):,} canonical SKUs into sku_master.")

    return len(df_unified)

if __name__ == "__main__":
    normalize_and_load()
