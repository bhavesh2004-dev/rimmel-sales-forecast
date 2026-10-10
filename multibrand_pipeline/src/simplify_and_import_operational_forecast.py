"""
Simplify MySQL Tables and Import Approved Operational Forecast
==============================================================
1. Creates the canonical table: operational_forecast_rop
2. Registers/updates the pipeline run in pipeline_runs
3. Imports the approved 1,433 rows from MULTIBRAND_10DAY_OPERATIONAL_FORECAST_COMBINED.xlsx
4. Verifies all 21 columns, row counts, brand distribution, and invariants
5. Safely removes redundant output tables after confirming backup integrity:
   - forecast_results
   - report_forecast_inventory
   - report_rop
   - inventory_results
   - replenishment_results
   - validation_results
6. Verifies the simplified database state:
   - Retained input/metadata tables untouched
   - Exactly one canonical operational forecast output table active
"""

import os
import sys
import logging
from datetime import datetime
import openpyxl
import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import text
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("simplify_and_import")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(PIPELINE_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

EXCEL_PATH = os.path.join(PIPELINE_ROOT, "reports", "MULTIBRAND_10DAY_OPERATIONAL_FORECAST_COMBINED.xlsx")
if not os.path.exists(EXCEL_PATH):
    EXCEL_PATH = os.path.join(PROJECT_ROOT, "artifacts", "reports", "MULTIBRAND_10DAY_OPERATIONAL_FORECAST_COMBINED.xlsx")

BACKUP_DIR = os.path.join(PROJECT_ROOT, "database_backups", "pre_simplification_20261010")

RUN_ID = "RUN-PROD-20261010-OPERATIONAL"
MODEL_ID = "global_lgbm_model.pkl"

def get_engine():
    user = os.getenv('MYSQL_USER', 'forecast_app')
    pwd = os.getenv('MYSQL_PASSWORD')
    host = os.getenv('MYSQL_HOST', '127.0.0.1')
    port = os.getenv('MYSQL_PORT', '3306')
    db = os.getenv('MYSQL_DATABASE', 'multibrand_forecasting_dev')
    url = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}?charset=utf8mb4"
    return sqlalchemy.create_engine(url, pool_recycle=3600)

def step1_verify_backups():
    logger.info("=== STEP 1: VERIFYING PRE-SIMPLIFICATION BACKUPS ===")
    required_backups = [
        "full_db_dump.sql",
        "forecast_results.sql", "forecast_results.csv",
        "report_forecast_inventory.sql", "report_forecast_inventory.csv",
        "report_rop.sql", "report_rop.csv",
        "inventory_results.sql", "inventory_results.csv",
        "replenishment_results.sql", "replenishment_results.csv",
        "validation_results.sql", "validation_results.csv"
    ]
    for b in required_backups:
        p = os.path.join(BACKUP_DIR, b)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            raise FileNotFoundError(f"Missing or empty required backup file: {p}")
        logger.info(f"  Backup OK: {b} ({os.path.getsize(p):,} bytes)")
    logger.info("All pre-simplification backups are intact and verified.")

def step2_load_and_validate_excel():
    logger.info(f"=== STEP 2: LOADING AND VALIDATING EXCEL SOURCE ===")
    logger.info(f"Source file: {EXCEL_PATH}")
    wb = openpyxl.load_workbook(EXCEL_PATH, data_only=True)
    if "Operational_Forecast_and_ROP" not in wb.sheetnames:
        raise ValueError("Worksheet 'Operational_Forecast_and_ROP' not found in workbook.")
    
    ws = wb["Operational_Forecast_and_ROP"]
    headers = [ws.cell(row=3, column=c).value for c in range(1, 22)]
    
    expected_cols = [
        'Brand', 'SKU', 'Product', 'Current Stock', 'Forecast Period',
        'Amazon Predicted', 'eBay Predicted', 'Website Predicted', 'Other Predicted',
        '10-Day Forecast', 'Days of Cover', 'Confidence', 'Risk',
        'Recommended Action', 'Reason', 'Lead Time (Days)', 'Avg Daily Usage',
        'Lead-Time Demand', 'Target Stock', 'Replenishment Qty', 'ROP Status'
    ]
    assert headers == expected_cols, f"Headers mismatch!\nExpected: {expected_cols}\nGot: {headers}"
    
    rows = []
    for r in range(4, ws.max_row + 1):
        brand = ws.cell(row=r, column=1).value
        sku = ws.cell(row=r, column=2).value
        if brand == "Total Portfolio" or sku is None:
            continue
        vals = [ws.cell(row=r, column=c).value for c in range(1, 22)]
        rows.append(vals)
        
    df = pd.DataFrame(rows, columns=headers)
    logger.info(f"Loaded {len(df):,} individual SKU rows.")
    assert len(df) == 1433, f"Expected exactly 1,433 SKU rows, got {len(df)}"
    
    # Check duplicate brand/SKU
    dups = df.duplicated(subset=['Brand', 'SKU']).sum()
    assert dups == 0, f"Found {dups} duplicate (Brand, SKU) pairs!"
    
    # Check nulls in required columns
    null_counts = df[['Brand', 'SKU', '10-Day Forecast', 'Current Stock', 'Replenishment Qty', 'ROP Status']].isna().sum().sum()
    assert null_counts == 0, f"Found {null_counts} nulls in required columns!"
    
    # Check Additive Channel Law
    ch_sum = df['Amazon Predicted'] + df['eBay Predicted'] + df['Website Predicted'] + df['Other Predicted']
    diff = (ch_sum - df['10-Day Forecast']).abs().sum()
    assert diff == 0, f"Additive channel law violated! Sum diff: {diff}"
    
    # Check Replenishment safety buffer
    buf = df['Current Stock'] + df['Replenishment Qty'] - df['Lead-Time Demand']
    assert buf.min() >= 6.0, f"Buffer violation: min={buf.min()}"
    
    logger.info(f"Excel validation passed 100%: 1,433 SKUs across {df['Brand'].nunique()} brands, 0 violations.")
    return df

def step3_create_canonical_table(engine):
    logger.info("=== STEP 3: CREATING CANONICAL TABLE operational_forecast_rop ===")
    ddl = """
    CREATE TABLE IF NOT EXISTS `operational_forecast_rop` (
        `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
        `run_id` VARCHAR(50) NOT NULL,
        `brand_id` VARCHAR(50) NOT NULL,
        `brand` VARCHAR(100) NOT NULL,
        `sku` VARCHAR(100) NOT NULL,
        `product` VARCHAR(255) NULL,
        `current_stock` INT NOT NULL DEFAULT 0,
        `forecast_period` VARCHAR(50) NOT NULL,
        `amazon_predicted` INT UNSIGNED NOT NULL DEFAULT 0,
        `ebay_predicted` INT UNSIGNED NOT NULL DEFAULT 0,
        `website_predicted` INT UNSIGNED NOT NULL DEFAULT 0,
        `other_predicted` INT UNSIGNED NOT NULL DEFAULT 0,
        `forecast_10d` INT UNSIGNED NOT NULL DEFAULT 0,
        `days_of_cover` VARCHAR(30) NOT NULL,
        `confidence` VARCHAR(20) NOT NULL,
        `risk` VARCHAR(50) NOT NULL,
        `recommended_action` VARCHAR(100) NOT NULL,
        `reason` TEXT NULL,
        `lead_time_days` INT UNSIGNED NOT NULL DEFAULT 10,
        `avg_daily_usage` DECIMAL(10,2) NOT NULL DEFAULT 0.00,
        `lead_time_demand` DECIMAL(10,2) NOT NULL DEFAULT 0.00,
        `target_stock` DECIMAL(10,2) NOT NULL DEFAULT 0.00,
        `replenishment_qty` INT UNSIGNED NOT NULL DEFAULT 0,
        `rop_status` VARCHAR(50) NOT NULL,
        `model_id` VARCHAR(100) NOT NULL DEFAULT 'global_lgbm_model.pkl',
        `data_cutoff_date` DATE NOT NULL,
        `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (`id`),
        UNIQUE KEY `uk_run_brand_sku` (`run_id`, `brand`, `sku`),
        KEY `idx_run_id` (`run_id`),
        KEY `idx_brand_id` (`brand_id`),
        KEY `idx_sku` (`sku`),
        KEY `idx_rop_status` (`rop_status`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
    logger.info("Table operational_forecast_rop created or verified successfully.")

def step4_import_operational_data(engine, df_xl):
    logger.info("=== STEP 4: IMPORTING APPROVED OPERATIONAL FORECAST ===")
    with engine.connect() as conn:
        df_b = pd.read_sql("SELECT brand_id, display_name, last_observed_date FROM brand_registry;", conn)
    
    brand_map = {}
    cutoff_map = {}
    for _, r in df_b.iterrows():
        b_disp = r['display_name']
        b_id = r['brand_id']
        c_date = str(r['last_observed_date'])
        brand_map[b_disp] = b_id
        cutoff_map[b_disp] = c_date
        
    logger.info(f"Brand metadata mapping loaded: {brand_map}")
    
    # Register/Update pipeline run in pipeline_runs
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO pipeline_runs (
                run_id, run_type, started_at, completed_at, status, model_id,
                dataset_records, forecast_start, forecast_end, notes
            ) VALUES (
                :run_id, 'FORECAST', NOW(), NOW(), 'COMPLETED', :model_id,
                134901, '2026-09-11', '2026-10-16',
                'Canonical 21-column operational forecast import for 1,433 SKUs'
            ) ON DUPLICATE KEY UPDATE
                status = 'COMPLETED',
                completed_at = NOW(),
                notes = 'Canonical 21-column operational forecast import for 1,433 SKUs';
        """), {"run_id": RUN_ID, "model_id": MODEL_ID})
    
    insert_sql = """
    INSERT INTO `operational_forecast_rop` (
        `run_id`, `brand_id`, `brand`, `sku`, `product`, `current_stock`,
        `forecast_period`, `amazon_predicted`, `ebay_predicted`, `website_predicted`,
        `other_predicted`, `forecast_10d`, `days_of_cover`, `confidence`,
        `risk`, `recommended_action`, `reason`, `lead_time_days`,
        `avg_daily_usage`, `lead_time_demand`, `target_stock`, `replenishment_qty`,
        `rop_status`, `model_id`, `data_cutoff_date`
    ) VALUES (
        :run_id, :brand_id, :brand, :sku, :product, :current_stock,
        :forecast_period, :amazon_predicted, :ebay_predicted, :website_predicted,
        :other_predicted, :forecast_10d, :days_of_cover, :confidence,
        :risk, :recommended_action, :reason, :lead_time_days,
        :avg_daily_usage, :lead_time_demand, :target_stock, :replenishment_qty,
        :rop_status, :model_id, :data_cutoff_date
    ) ON DUPLICATE KEY UPDATE
        `brand_id` = VALUES(`brand_id`),
        `product` = VALUES(`product`),
        `current_stock` = VALUES(`current_stock`),
        `forecast_period` = VALUES(`forecast_period`),
        `amazon_predicted` = VALUES(`amazon_predicted`),
        `ebay_predicted` = VALUES(`ebay_predicted`),
        `website_predicted` = VALUES(`website_predicted`),
        `other_predicted` = VALUES(`other_predicted`),
        `forecast_10d` = VALUES(`forecast_10d`),
        `days_of_cover` = VALUES(`days_of_cover`),
        `confidence` = VALUES(`confidence`),
        `risk` = VALUES(`risk`),
        `recommended_action` = VALUES(`recommended_action`),
        `reason` = VALUES(`reason`),
        `lead_time_days` = VALUES(`lead_time_days`),
        `avg_daily_usage` = VALUES(`avg_daily_usage`),
        `lead_time_demand` = VALUES(`lead_time_demand`),
        `target_stock` = VALUES(`target_stock`),
        `replenishment_qty` = VALUES(`replenishment_qty`),
        `rop_status` = VALUES(`rop_status`),
        `model_id` = VALUES(`model_id`),
        `data_cutoff_date` = VALUES(`data_cutoff_date`);
    """
    
    records = []
    for _, r in df_xl.iterrows():
        b_name = r['Brand']
        b_id = brand_map.get(b_name, b_name.upper().replace(' ', '_'))
        c_date = cutoff_map.get(b_name, '2026-10-06')
        
        rec = {
            'run_id': RUN_ID,
            'brand_id': b_id,
            'brand': str(b_name),
            'sku': str(r['SKU']),
            'product': str(r['Product']) if pd.notna(r['Product']) else None,
            'current_stock': int(r['Current Stock']),
            'forecast_period': str(r['Forecast Period']),
            'amazon_predicted': int(r['Amazon Predicted']),
            'ebay_predicted': int(r['eBay Predicted']),
            'website_predicted': int(r['Website Predicted']),
            'other_predicted': int(r['Other Predicted']),
            'forecast_10d': int(r['10-Day Forecast']),
            'days_of_cover': str(r['Days of Cover']),
            'confidence': str(r['Confidence']),
            'risk': str(r['Risk']),
            'recommended_action': str(r['Recommended Action']),
            'reason': str(r['Reason']) if pd.notna(r['Reason']) else None,
            'lead_time_days': int(r['Lead Time (Days)']),
            'avg_daily_usage': float(r['Avg Daily Usage']),
            'lead_time_demand': float(r['Lead-Time Demand']),
            'target_stock': float(r['Target Stock']),
            'replenishment_qty': int(r['Replenishment Qty']),
            'rop_status': str(r['ROP Status']),
            'model_id': MODEL_ID,
            'data_cutoff_date': c_date
        }
        records.append(rec)
        
    with engine.begin() as conn:
        conn.execute(text(insert_sql), records)
        
    logger.info(f"Successfully inserted/updated {len(records):,} records in operational_forecast_rop.")

def step5_verify_imported_data(engine):
    logger.info("=== STEP 5: VERIFYING IMPORTED DATA IN MYSQL ===")
    with engine.connect() as conn:
        cnt = conn.execute(text(f"SELECT COUNT(*) FROM operational_forecast_rop WHERE run_id = '{RUN_ID}';")).scalar()
        logger.info(f"Rows in operational_forecast_rop for run {RUN_ID}: {cnt:,}")
        assert cnt == 1433, f"Expected 1,433 rows, got {cnt}"
        
        # Check brand distribution
        df_b = pd.read_sql(f"SELECT brand, COUNT(*) as cnt FROM operational_forecast_rop WHERE run_id = '{RUN_ID}' GROUP BY brand ORDER BY cnt DESC;", conn)
        logger.info(f"Brand distribution:\n{df_b.to_string(index=False)}")
        assert len(df_b) == 7, f"Expected 7 brands, found {len(df_b)}"
        
        # Check channel sums vs forecast_10d
        res = conn.execute(text(f"""
            SELECT 
                SUM(amazon_predicted) as amz,
                SUM(ebay_predicted) as ebay,
                SUM(website_predicted) as web,
                SUM(other_predicted) as oth,
                SUM(forecast_10d) as tot_fwd,
                SUM(replenishment_qty) as tot_replenish,
                SUM(current_stock) as tot_stock
            FROM operational_forecast_rop
            WHERE run_id = '{RUN_ID}';
        """)).fetchone()
        logger.info(f"Totals: Amazon={res[0]:,}, eBay={res[1]:,}, Web={res[2]:,}, Other={res[3]:,}, Total 10d={res[4]:,}, Replenish={res[5]:,}, Stock={res[6]:,}")
        assert (res[0] + res[1] + res[2] + res[3]) == res[4], "Additive channel law failed in MySQL!"
        assert res[4] == 3047, f"Expected 3,047 forecast units, got {res[4]}"
        assert res[5] == 4101, f"Expected 4,101 replenishment units, got {res[5]}"
        assert res[6] == 280058, f"Expected 280,058 warehouse units, got {res[6]}"
        
        # Check buffer invariant
        viol = conn.execute(text(f"""
            SELECT COUNT(*) FROM operational_forecast_rop
            WHERE run_id = '{RUN_ID}' AND (current_stock + replenishment_qty - lead_time_demand) < 6.0;
        """)).scalar()
        assert viol == 0, f"Safety buffer invariant violations found: {viol}"
        
        # Check duplicates
        dups = conn.execute(text(f"""
            SELECT brand, sku, COUNT(*) FROM operational_forecast_rop
            WHERE run_id = '{RUN_ID}'
            GROUP BY brand, sku
            HAVING COUNT(*) > 1;
        """)).fetchall()
        assert len(dups) == 0, f"Found duplicate brand/sku combinations: {dups}"
        
    logger.info("All post-import SQL verification checks passed perfectly!")

def step6_safe_cleanup_redundant_tables(engine):
    logger.info("=== STEP 6: SAFELY DROPPING REDUNDANT OUTPUT TABLES ===")
    redundant_tables = [
        "forecast_results",
        "report_forecast_inventory",
        "report_rop",
        "inventory_results",
        "replenishment_results",
        "validation_results"
    ]
    with engine.begin() as conn:
        for t in redundant_tables:
            logger.info(f"Dropping redundant table: {t}...")
            conn.execute(text(f"DROP TABLE IF EXISTS `{t}`;"))
            logger.info(f"  Dropped: {t}")
            
    logger.info("Redundant output tables dropped successfully.")

def step7_verify_final_database_inventory(engine):
    logger.info("=== STEP 7: VERIFYING FINAL DATABASE INVENTORY ===")
    with engine.connect() as conn:
        tables = [row[0] for row in conn.execute(text("SHOW TABLES;")).fetchall()]
        logger.info(f"Final active tables ({len(tables)}): {tables}")
        
        expected_retained = [
            'brand_registry',
            'sku_master',
            'normalized_sales',
            'order_sales_data',
            'raw_rimmel_sales_data',
            'import_batches',
            'pipeline_runs',
            'model_registry',
            'operational_forecast_rop'
        ]
        
        summary = []
        for t in tables:
            cnt = conn.execute(text(f"SELECT COUNT(*) FROM `{t}`;")).scalar()
            role = "CANONICAL OPERATIONAL FORECAST OUTPUT" if t == "operational_forecast_rop" else \
                   "UNIFIED HISTORICAL SALES INPUT" if t == "normalized_sales" else \
                   "MASTER DIMENSION / METADATA" if t in ['brand_registry', 'sku_master', 'pipeline_runs', 'model_registry', 'import_batches'] else \
                   "RAW INGESTION STAGING / AUDIT"
            summary.append({"Table": t, "Rows": cnt, "Role": role})
            
        df_inv = pd.DataFrame(summary)
        print("\n" + "="*80)
        print("FINAL SIMPLIFIED MYSQL DATABASE INVENTORY")
        print("="*80)
        print(df_inv.to_string(index=False))
        print("="*80 + "\n")
        
        assert set(tables) == set(expected_retained), f"Inventory mismatch!\nExpected: {expected_retained}\nFound: {tables}"
        assert conn.execute(text("SELECT COUNT(*) FROM normalized_sales;")).scalar() == 134901, "normalized_sales rows changed!"
        assert conn.execute(text("SELECT COUNT(*) FROM operational_forecast_rop;")).scalar() == 1433, "operational_forecast_rop rows != 1,433!"
        
    logger.info("Final database inventory verified 100%!")

def main():
    engine = get_engine()
    step1_verify_backups()
    df_xl = step2_load_and_validate_excel()
    step3_create_canonical_table(engine)
    step4_import_operational_data(engine, df_xl)
    step5_verify_imported_data(engine)
    step6_safe_cleanup_redundant_tables(engine)
    step7_verify_final_database_inventory(engine)
    logger.info("=== ALL STEPS COMPLETED SUCCESSFULLY AND CERTIFIED ===")

if __name__ == "__main__":
    main()
