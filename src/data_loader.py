"""
DATA LOADER MODULE
==================
Loads historical daily sales logs from the master SQLite database.
"""
import sqlite3
import pandas as pd
from config.settings import DB_PATH

def load_sales_data(database_path=DB_PATH):
    """
    Loads daily historical sales and inventory data from full_history_v4.
    
    Returns:
        pd.DataFrame: Clean daily sales dataframe with columns:
                      [sku, date, total_sales, current_stock, in_stock_flag, selling_price, category]
    """
    connection = sqlite3.connect(database_path)
    query = """
        SELECT sku, date, total_sales, current_stock, in_stock_flag, selling_price, category 
        FROM full_history_v4 
        ORDER BY sku, date
    """
    sales_dataframe = pd.read_sql(query, connection)
    connection.close()
    
    sales_dataframe['date'] = pd.to_datetime(sales_dataframe['date'])
    sales_dataframe['total_sales'] = sales_dataframe['total_sales'].astype(float)
    sales_dataframe['current_stock'] = sales_dataframe['current_stock'].astype(int)
    sales_dataframe['selling_price'] = sales_dataframe['selling_price'].astype(float)
    sales_dataframe['in_stock_flag'] = (sales_dataframe['current_stock'] > 0).astype(int)
    
    return sales_dataframe
