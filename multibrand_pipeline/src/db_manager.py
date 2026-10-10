"""
MySQL Database Connection and Schema Management for Multi-Brand Demand Forecasting.
Loads credentials securely from .env and executes schema DDL and operational persistence.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import sqlalchemy
from sqlalchemy import text
import pymysql
import pandas as pd

logger = logging.getLogger(__name__)

class DBManager:
    """Manages MySQL connections, connection pooling, and canonical data operations."""

    def __init__(self, env_path: Optional[str] = None):
        if env_path is None:
            # Locate .env by traversing up to the project root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            pipeline_root = os.path.dirname(current_dir)
            workspace_root = os.path.dirname(pipeline_root)
            candidate_paths = [
                os.path.join(workspace_root, '.env'),
                os.path.join(pipeline_root, '.env'),
                os.path.join(os.getcwd(), '.env')
            ]
            for p in candidate_paths:
                if os.path.exists(p):
                    env_path = p
                    break
            if env_path is None:
                env_path = candidate_paths[0]

        load_dotenv(env_path)
        
        self.host = os.getenv('MYSQL_HOST', '127.0.0.1')
        self.port = int(os.getenv('MYSQL_PORT', '3306'))
        self.database = os.getenv('MYSQL_DATABASE', 'multibrand_forecasting_dev')
        self.user = os.getenv('MYSQL_USER', 'forecast_app')
        self.password = os.getenv('MYSQL_PASSWORD')
        
        if not self.password:
            raise ValueError(
                "MYSQL_PASSWORD is not set in environment or .env file. "
                "Please configure .env based on .env.example."
            )
            
        self.connection_url = (
            f"mysql+pymysql://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}?charset=utf8mb4"
        )
        self.engine = sqlalchemy.create_engine(
            self.connection_url,
            pool_recycle=3600,
            pool_pre_ping=True
        )

    def test_connection(self) -> Dict[str, Any]:
        """Validates connection and returns statistics on authoritative normalized_sales."""
        with self.engine.connect() as conn:
            res = conn.execute(
                text("SELECT COUNT(*), MIN(`date`), MAX(`date`) FROM `normalized_sales`")
            ).fetchone()
            brand_res = conn.execute(
                text("SELECT brand_id, COUNT(*) FROM `normalized_sales` GROUP BY brand_id ORDER BY COUNT(*) DESC")
            ).fetchall()
            
        return {
            "status": "connected",
            "host": self.host,
            "database": self.database,
            "user": self.user,
            "normalized_sales_rows": res[0] if res else 0,
            "min_date": str(res[1]) if res else None,
            "max_date": str(res[2]) if res else None,
            "brands": {b[0]: b[1] for b in brand_res}
        }

    def execute_ddl(self, sql_script_path: str):
        """Executes a DDL script file containing multiple semicolon-separated statements."""
        if not os.path.exists(sql_script_path):
            raise FileNotFoundError(f"DDL script not found: {sql_script_path}")
            
        with open(sql_script_path, 'r', encoding='utf-8') as f:
            script_content = f.read()

        statements = [stmt.strip() for stmt in script_content.split(';') if stmt.strip()]
        
        with self.engine.begin() as conn:
            for stmt in statements:
                if stmt:
                    conn.execute(text(stmt))
        logger.info(f"Successfully executed DDL script: {sql_script_path}")

    def get_existing_tables(self) -> List[str]:
        """Returns a list of all tables currently existing in the target MySQL database."""
        with self.engine.connect() as conn:
            res = conn.execute(text("SHOW TABLES")).fetchall()
        return [r[0] for r in res]

    def ensure_operational_forecast_rop_schema(self):
        """Creates the canonical operational_forecast_rop table if not already present."""
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
        with self.engine.begin() as conn:
            conn.execute(text(ddl))
        logger.info("Verified canonical table operational_forecast_rop schema.")

    def save_operational_forecast_rop(self, records: List[Dict[str, Any]]) -> int:
        """
        Idempotently inserts or updates operational forecast records into operational_forecast_rop.
        Enforces UNIQUE KEY (run_id, brand, sku) via ON DUPLICATE KEY UPDATE.
        """
        self.ensure_operational_forecast_rop_schema()
        
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
        with self.engine.begin() as conn:
            conn.execute(text(insert_sql), records)
        logger.info(f"Persisted {len(records):,} operational records into MySQL operational_forecast_rop.")
        return len(records)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    db = DBManager()
    stats = db.test_connection()
    print("Database Connection Verified:")
    print(f"  Database: {stats['database']}")
    print(f"  Rows in normalized_sales: {stats['normalized_sales_rows']:,}")
    print(f"  Date Range: {stats['min_date']} to {stats['max_date']}")
    print("  Brand Breakdown:")
    for b, c in stats['brands'].items():
        print(f"    - {b}: {c:,} rows")
