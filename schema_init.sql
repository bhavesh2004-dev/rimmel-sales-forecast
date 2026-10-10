-- =====================================================================
-- MULTI-BRAND FORECASTING RELATIONAL SCHEMA INITIALIZATION
-- Database: multibrand_forecasting_dev
-- Designed for MySQL 8.0+
-- =====================================================================

USE multibrand_forecasting_dev;

-- 1. BRAND REGISTRY
CREATE TABLE IF NOT EXISTS brand_registry (
    brand_id VARCHAR(50) NOT NULL PRIMARY KEY,
    display_name VARCHAR(100) NOT NULL,
    status ENUM('ACTIVE', 'INACTIVE', 'COLD_START') NOT NULL DEFAULT 'ACTIVE',
    first_observed_date DATE NULL,
    last_observed_date DATE NULL,
    total_skus INT UNSIGNED NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 2. IMPORT BATCHES
CREATE TABLE IF NOT EXISTS import_batches (
    batch_id VARCHAR(50) NOT NULL PRIMARY KEY,
    source_name VARCHAR(150) NOT NULL,
    source_type VARCHAR(50) NOT NULL,
    brand_id VARCHAR(50) NULL,
    row_count INT UNSIGNED NOT NULL DEFAULT 0,
    min_date DATE NULL,
    max_date DATE NULL,
    status ENUM('PENDING', 'LOADED', 'FAILED') NOT NULL DEFAULT 'PENDING',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_batch_brand (brand_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 3. RAW RIMMEL SALES DATA (Source of Truth for Rimmel brand)
CREATE TABLE IF NOT EXISTS raw_rimmel_sales_data (
    id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    batch_id VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    sku VARCHAR(100) NOT NULL,
    category TEXT NULL,
    channel VARCHAR(50) NOT NULL,
    listing_id VARCHAR(150) NULL,
    parent_id VARCHAR(150) NULL,
    actual_sku VARCHAR(150) NULL,
    pack_multiplier TINYINT UNSIGNED DEFAULT 1,
    units_sold INT UNSIGNED DEFAULT 0,
    orders_count INT UNSIGNED DEFAULT 1,
    selling_price DECIMAL(12,2) DEFAULT 0.00,
    launch_date DATE NULL,
    current_stock INT DEFAULT NULL,
    child_asin VARCHAR(150) NULL,
    buy_box_percentage DECIMAL(10,4) DEFAULT NULL,
    amazon_sessions VARCHAR(20) DEFAULT NULL,
    fulfillment_type VARCHAR(10) DEFAULT NULL,
    ebay_promoted_flag ENUM('0','1') DEFAULT NULL,
    restock_date DATE NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_rimmel_date (date),
    INDEX idx_rimmel_sku (sku),
    INDEX idx_rimmel_batch (batch_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 4. SKU MASTER
CREATE TABLE IF NOT EXISTS sku_master (
    canonical_sku VARCHAR(100) NOT NULL PRIMARY KEY,
    brand_id VARCHAR(50) NOT NULL,
    product_title VARCHAR(255) NULL,
    category VARCHAR(150) NULL,
    current_stock INT DEFAULT 0,
    launch_date DATE NULL,
    behavior_class VARCHAR(50) NULL,
    adi DECIMAL(10,4) DEFAULT NULL,
    cv2 DECIMAL(10,4) DEFAULT NULL,
    zero_demand_ratio DECIMAL(6,4) DEFAULT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_sku_brand (brand_id),
    INDEX idx_sku_class (behavior_class)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 5. UNIFIED NORMALIZED SALES
CREATE TABLE IF NOT EXISTS normalized_sales (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    source_table VARCHAR(50) NOT NULL,
    source_row_id INT UNSIGNED NOT NULL,
    batch_id VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    brand_id VARCHAR(50) NOT NULL,
    display_brand_name VARCHAR(100) NOT NULL,
    canonical_sku VARCHAR(100) NOT NULL,
    raw_sku VARCHAR(100) NOT NULL,
    platform_group ENUM('Amazon', 'eBay', 'Website', 'Other') NOT NULL,
    raw_channel VARCHAR(100) NOT NULL,
    units_sold INT UNSIGNED NOT NULL DEFAULT 0,
    orders_count INT UNSIGNED NOT NULL DEFAULT 1,
    pack_multiplier TINYINT UNSIGNED NOT NULL DEFAULT 1,
    selling_price DECIMAL(12,2) NOT NULL DEFAULT 0.00,
    current_stock INT DEFAULT NULL,
    category VARCHAR(150) NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_norm_date_brand (date, brand_id),
    INDEX idx_norm_sku_date (canonical_sku, date),
    INDEX idx_norm_platform (platform_group),
    INDEX idx_norm_batch (batch_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 6. PIPELINE RUNS
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id VARCHAR(50) NOT NULL PRIMARY KEY,
    run_type ENUM('VALIDATION', 'FORECAST', 'FULL') NOT NULL,
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP NULL,
    status ENUM('STARTED', 'COMPLETED', 'FAILED') NOT NULL DEFAULT 'STARTED',
    model_id VARCHAR(50) NOT NULL,
    dataset_records INT UNSIGNED NOT NULL DEFAULT 0,
    forecast_start DATE NULL,
    forecast_end DATE NULL,
    notes TEXT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 7. MODEL REGISTRY
CREATE TABLE IF NOT EXISTS model_registry (
    model_id VARCHAR(50) NOT NULL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_family VARCHAR(50) NOT NULL,
    version VARCHAR(20) NOT NULL,
    trained_brands VARCHAR(255) NOT NULL,
    feature_count INT UNSIGNED NOT NULL,
    status ENUM('ACTIVE_PROD', 'ARCHIVED', 'CANDIDATE') NOT NULL DEFAULT 'CANDIDATE',
    metrics_json JSON NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 8. VALIDATION RESULTS
CREATE TABLE IF NOT EXISTS validation_results (
    id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    run_id VARCHAR(50) NOT NULL,
    model_id VARCHAR(50) NOT NULL,
    brand_id VARCHAR(50) NOT NULL,
    validation_window_start DATE NOT NULL,
    validation_window_end DATE NOT NULL,
    holdout_days INT UNSIGNED NOT NULL,
    sku_count INT UNSIGNED NOT NULL,
    wape DECIMAL(8,4) NULL,
    mae DECIMAL(10,4) NULL,
    rmse DECIMAL(10,4) NULL,
    bias DECIMAL(10,4) NULL,
    actual_units INT UNSIGNED NOT NULL DEFAULT 0,
    predicted_units INT UNSIGNED NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_val_run (run_id),
    INDEX idx_val_brand (brand_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 9. FORECAST RESULTS (Detailed by Platform and Date)
CREATE TABLE IF NOT EXISTS forecast_results (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    run_id VARCHAR(50) NOT NULL,
    canonical_sku VARCHAR(100) NOT NULL,
    brand_id VARCHAR(50) NOT NULL,
    forecast_date DATE NOT NULL,
    platform_group ENUM('Amazon', 'eBay', 'Website', 'Other') NOT NULL,
    predicted_quantity DECIMAL(10,4) NOT NULL DEFAULT 0.0000,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_fwd_run_sku (run_id, canonical_sku),
    INDEX idx_fwd_date (forecast_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 10. INVENTORY RESULTS (Days of Cover & Operational Risk)
CREATE TABLE IF NOT EXISTS inventory_results (
    id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    run_id VARCHAR(50) NOT NULL,
    canonical_sku VARCHAR(100) NOT NULL,
    brand_id VARCHAR(50) NOT NULL,
    current_stock INT NOT NULL DEFAULT 0,
    forecast_10d INT UNSIGNED NOT NULL DEFAULT 0,
    days_of_cover VARCHAR(30) NOT NULL,
    risk VARCHAR(50) NOT NULL,
    recommended_action VARCHAR(100) NOT NULL,
    confidence VARCHAR(20) NOT NULL,
    reason TEXT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_inv_run_sku (run_id, canonical_sku),
    INDEX idx_inv_risk (risk)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 11. REPLENISHMENT RESULTS (Tosif Reorder Rules)
CREATE TABLE IF NOT EXISTS replenishment_results (
    id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    run_id VARCHAR(50) NOT NULL,
    canonical_sku VARCHAR(100) NOT NULL,
    brand_id VARCHAR(50) NOT NULL,
    current_stock INT NOT NULL DEFAULT 0,
    forecast_10d INT UNSIGNED NOT NULL DEFAULT 0,
    lead_time_days INT UNSIGNED NOT NULL DEFAULT 10,
    min_stock_level INT UNSIGNED NOT NULL DEFAULT 6,
    avg_daily_usage DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    lead_time_demand DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    target_stock DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    replenishment_qty INT UNSIGNED NOT NULL DEFAULT 0,
    rop_status VARCHAR(50) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_rep_run_sku (run_id, canonical_sku),
    INDEX idx_rep_status (rop_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 12. REPORT FORECAST INVENTORY (Mirrors Sheet 1)
CREATE TABLE IF NOT EXISTS report_forecast_inventory (
    id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    run_id VARCHAR(50) NOT NULL,
    brand VARCHAR(100) NOT NULL,
    sku VARCHAR(100) NOT NULL,
    product VARCHAR(255) NULL,
    current_stock INT NOT NULL DEFAULT 0,
    forecast_period VARCHAR(50) NOT NULL,
    amazon_predicted INT UNSIGNED NOT NULL DEFAULT 0,
    ebay_predicted INT UNSIGNED NOT NULL DEFAULT 0,
    website_predicted INT UNSIGNED NOT NULL DEFAULT 0,
    other_predicted INT UNSIGNED NOT NULL DEFAULT 0,
    forecast_10d INT UNSIGNED NOT NULL DEFAULT 0,
    days_of_cover VARCHAR(30) NOT NULL,
    confidence VARCHAR(20) NOT NULL,
    risk VARCHAR(50) NOT NULL,
    recommended_action VARCHAR(100) NOT NULL,
    reason TEXT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_rfi_run (run_id),
    INDEX idx_rfi_brand (brand),
    INDEX idx_rfi_sku (sku)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 13. REPORT ROP (Mirrors Sheet 2)
CREATE TABLE IF NOT EXISTS report_rop (
    id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    run_id VARCHAR(50) NOT NULL,
    brand VARCHAR(100) NOT NULL,
    sku VARCHAR(100) NOT NULL,
    product VARCHAR(255) NULL,
    current_stock INT NOT NULL DEFAULT 0,
    forecast_10d INT UNSIGNED NOT NULL DEFAULT 0,
    avg_daily_usage DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    lead_time_demand DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    target_stock DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    replenishment_qty INT UNSIGNED NOT NULL DEFAULT 0,
    rop_status VARCHAR(50) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_rop_run (run_id),
    INDEX idx_rop_brand (brand),
    INDEX idx_rop_sku (sku)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
