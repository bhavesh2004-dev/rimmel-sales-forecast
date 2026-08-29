-- Production MySQL Database Schema (Version 3.0)

CREATE DATABASE IF NOT EXISTS sales_forecasting_db;
USE sales_forecasting_db;

-- 1. Brands Table
CREATE TABLE IF NOT EXISTS brands (
    brand_id INT AUTO_INCREMENT PRIMARY KEY,
    brand_name VARCHAR(100) UNIQUE NOT NULL
);

-- 2. Products Table (with category)
CREATE TABLE IF NOT EXISTS products (
    sku_code VARCHAR(100) PRIMARY KEY,
    brand_name VARCHAR(100),
    product_name VARCHAR(255),
    category VARCHAR(150) DEFAULT 'Cosmetics'
);

-- 3. Daily Sales & Inventory Table (with selling_price)
CREATE TABLE IF NOT EXISTS daily_sales (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    date DATE NOT NULL,
    order_item_sku VARCHAR(100) NOT NULL,
    channel VARCHAR(100),
    selling_price FLOAT DEFAULT 5.0,
    sold_qty INT DEFAULT 0,
    eod_stock FLOAT DEFAULT NULL,
    INDEX idx_date_sku (date, order_item_sku),
    FOREIGN KEY (order_item_sku) REFERENCES products(sku_code) ON DELETE CASCADE
);

-- 4. Forecast Results Table
CREATE TABLE IF NOT EXISTS forecast_results (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    order_item_sku VARCHAR(100) NOT NULL,
    date DATE NOT NULL,
    predicted_sales FLOAT NOT NULL,
    lower_bound_95 FLOAT NOT NULL,
    upper_bound_95 FLOAT NOT NULL,
    model_type VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_forecast (order_item_sku, date)
);
