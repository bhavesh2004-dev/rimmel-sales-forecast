"""
Check stock on July 20 for RIM-MSC-ESL-101
"""
import os, sqlite3
import pandas as pd

BASE_DIR = r'C:\Users\bhave\Desktop\ml_project'
DB_PATH  = os.path.join(BASE_DIR, 'data', 'rimmel_clean.db')
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT sku, date, total_sales, current_stock, in_stock_flag FROM full_history_v4 WHERE sku IN ('RIM-MSC-ESL-101', 'RIM-SMPP-001') AND date >= '2026-07-15' AND date <= '2026-07-25' ORDER BY sku, date", conn)
conn.close()

print(df.to_string(index=False))
