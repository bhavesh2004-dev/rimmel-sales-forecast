"""
Pure SQL Data Loader Module (Version 3.0)
Feeds data directly from SQL database (SQLite/MySQL) into the ML feature engineering & training pipeline via SQL queries.
Includes selling_price ($ USD) and category.
"""
import os
import sqlite3
import pandas as pd
import numpy as np

def load_data_from_sql(db_path_or_url: str, sku_code: str = None) -> pd.DataFrame:
    """
    Executes SQL SELECT queries to fetch daily sales, price, category & product metadata directly from database.
    """
    if db_path_or_url.endswith('.db') or os.path.exists(db_path_or_url):
        conn = sqlite3.connect(db_path_or_url)
        
        if sku_code:
            query = """
            SELECT 
                s.date,
                s.order_item_sku,
                p.product_name,
                p.brand_name,
                p.category,
                s.channel,
                s.selling_price,
                s.sold_qty,
                s.eod_stock
            FROM daily_sales s
            LEFT JOIN products p ON s.order_item_sku = p.sku_code
            WHERE s.order_item_sku = ?
            ORDER BY s.date ASC
            """
            df = pd.read_sql_query(query, conn, params=(sku_code,))
        else:
            query = """
            SELECT 
                s.date,
                s.order_item_sku,
                p.product_name,
                p.brand_name,
                p.category,
                s.channel,
                s.selling_price,
                s.sold_qty,
                s.eod_stock
            FROM daily_sales s
            LEFT JOIN products p ON s.order_item_sku = p.sku_code
            ORDER BY s.date ASC
            """
            df = pd.read_sql_query(query, conn)
            
        conn.close()
    else:
        from sqlalchemy import create_engine
        engine = create_engine(db_path_or_url)
        if sku_code:
            query = f"""
            SELECT 
                s.date,
                s.order_item_sku,
                p.product_name,
                p.brand_name,
                p.category,
                s.channel,
                s.selling_price,
                s.sold_qty,
                s.eod_stock
            FROM daily_sales s
            LEFT JOIN products p ON s.order_item_sku = p.sku_code
            WHERE s.order_item_sku = '{sku_code}'
            ORDER BY s.date ASC
            """
        else:
            query = """
            SELECT 
                s.date,
                s.order_item_sku,
                p.product_name,
                p.brand_name,
                p.category,
                s.channel,
                s.selling_price,
                s.sold_qty,
                s.eod_stock
            FROM daily_sales s
            LEFT JOIN products p ON s.order_item_sku = p.sku_code
            ORDER BY s.date ASC
            """
        df = pd.read_sql_query(query, engine)
        
    df['date'] = pd.to_datetime(df['date'])
    return df

def save_forecast_to_sql(db_path_or_url: str, forecast_df: pd.DataFrame):
    """
    Inserts 30-day predicted sales into forecast_results SQL table.
    """
    if db_path_or_url.endswith('.db') or os.path.exists(db_path_or_url):
        conn = sqlite3.connect(db_path_or_url)
        forecast_df.to_sql('forecast_results', conn, if_exists='append', index=False)
        conn.close()
    else:
        from sqlalchemy import create_engine
        engine = create_engine(db_path_or_url)
        forecast_df.to_sql('forecast_results', engine, if_exists='append', index=False)
    print("Successfully pushed 30-day forecast predictions into SQL forecast_results table!")
