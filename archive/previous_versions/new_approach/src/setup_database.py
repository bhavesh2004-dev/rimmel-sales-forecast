"""
Database Setup & Migration Script (Version 3.0)
Migrates the NEW dataset (Rimmel New.xlsx) into local SQL Database (SQLite) and generates production MySQL DDL script.
Includes new columns: selling_price ($ USD) and category.
"""
import os
import sqlite3
import pandas as pd
import numpy as np

DEFAULT_NEW_DATASET = 'Rimmel Brand Products Sales - 1 Jan 2025 to 31st July 2026 New.xlsx'

def create_sqlite_database(db_path: str, excel_path: str):
    print(f"Creating & Updating Version 3.0 SQL Database at: {db_path}...")
    
    if os.path.exists(db_path):
        os.remove(db_path)
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Brands Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS brands (
        brand_id INTEGER PRIMARY KEY AUTOINCREMENT,
        brand_name TEXT UNIQUE
    );
    """)
    
    # 2. Products Table (with category)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        sku_code TEXT PRIMARY KEY,
        brand_name TEXT,
        product_name TEXT,
        category TEXT
    );
    """)
    
    # 3. Daily Sales & Price Table (with selling_price)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS daily_sales (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        order_item_sku TEXT,
        channel TEXT,
        selling_price REAL,
        sold_qty REAL,
        eod_stock REAL,
        FOREIGN KEY(order_item_sku) REFERENCES products(sku_code)
    );
    """)
    
    # 4. Forecast Results Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS forecast_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_item_sku TEXT,
        date TEXT,
        predicted_sales REAL,
        lower_bound_95 REAL,
        upper_bound_95 REAL,
        model_type TEXT
    );
    """)
    
    conn.commit()
    
    print("Migrating Version 3.0 Excel dataset (113,031 rows) into SQL Database...")
    df = pd.read_excel(excel_path)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    sku_col = 'sku' if 'sku' in df.columns else 'order_item_sku'
    qty_col = 'sold_quantity' if 'sold_quantity' in df.columns else 'sold_qty'
    title_col = 'title' if 'title' in df.columns else 'product_name'
    stock_col = 'current_stock' if 'current_stock' in df.columns else 'eod_stock'
    price_col = 'selling_price' if 'selling_price' in df.columns else 'unit_price'
    
    # Insert brand
    cursor.execute("INSERT OR IGNORE INTO brands (brand_name) VALUES ('Rimmel');")
    
    # Insert products
    products = df[[sku_col, title_col, 'category']].drop_duplicates(subset=[sku_col]).copy()
    products.columns = ['sku_code', 'product_name', 'category']
    products['brand_name'] = 'Rimmel'
    products['category'] = products['category'].fillna('Cosmetics')
    products.to_sql('products', conn, if_exists='append', index=False)
    
    # Insert daily sales
    sales_df = df[['date', sku_col, 'channel', price_col, qty_col, stock_col]].copy()
    sales_df.columns = ['date', 'order_item_sku', 'channel', 'selling_price', 'sold_qty', 'eod_stock']
    sales_df['selling_price'] = sales_df['selling_price'].ffill().bfill().fillna(5.0)
    sales_df.to_sql('daily_sales', conn, if_exists='append', index=False)
    
    conn.commit()
    conn.close()
    print("Version 3.0 SQL Database created and populated successfully!")

def generate_mysql_schema_script(output_sql_path: str):
    sql_content = """-- Production MySQL Database Schema (Version 3.0)

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
"""
    with open(output_sql_path, 'w') as f:
        f.write(sql_content)
    print(f"Generated Version 3.0 MySQL DDL Script: {output_sql_path}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(base_dir, '..', '..')
    excel_path = os.path.join(project_dir, DEFAULT_NEW_DATASET)
    
    sqlite_db_path = os.path.join(project_dir, 'new_approach', 'sales_forecasting.db')
    create_sqlite_database(sqlite_db_path, excel_path)
    
    mysql_sql_path = os.path.join(project_dir, 'new_approach', 'schema_mysql.sql')
    generate_mysql_schema_script(mysql_sql_path)
