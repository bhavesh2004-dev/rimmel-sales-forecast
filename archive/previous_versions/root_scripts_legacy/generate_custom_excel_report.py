"""
GENERATE MASTER 588-SKU EXCEL REPORT WITH EXACT SPECIFIED COLUMNS
=================================================================
Exact Columns for Sheet 1 & Sheet 2:
date | product | predicted value | actual value | lower bound | upper bound

- Sheet 1: 'Aug 1-11 Forecast' (588 SKUs, date = '01/08/2026 to 11/08/2026', actual value = empty)
- Sheet 2: 'Aug 1-31 Full Month' (588 SKUs, date = '01/08/2026 to 31/08/2026', actual value = empty)
- Sheet 3: 'Overstock & Clearance Action' (588 SKUs with Days of Supply & Action)

Exports to: reports/Rimmel_August2026_AllProducts_Forecast.xlsx
"""
import os, sys
import sqlite3
import pandas as pd
import numpy as np

BASE_DIR  = r'C:\Users\bhave\Desktop\ml_project'
DB_PATH   = os.path.join(BASE_DIR, 'data', 'rimmel_clean.db')
OUT_EXCEL = os.path.join(BASE_DIR, 'reports', 'Rimmel_August2026_AllProducts_Forecast.xlsx')
OUT_EXCEL_ALT = os.path.join(BASE_DIR, 'reports', 'Rimmel_August2026_AllProducts_Forecast_Updated.xlsx')

os.makedirs(os.path.join(BASE_DIR, 'reports'), exist_ok=True)

conn = sqlite3.connect(DB_PATH)
df_daily = pd.read_sql("SELECT * FROM forecast_daily_aug2026_v4", conn)
df_full  = pd.read_sql("SELECT sku, date, current_stock, selling_price, category FROM full_history_v4 ORDER BY sku, date", conn)
conn.close()

df_daily['date'] = pd.to_datetime(df_daily['date'])
df_full['date']  = pd.to_datetime(df_full['date'])

all_588_skus = sorted(df_full['sku'].unique().tolist())
print(f"Generating Master Excel Report for {len(all_588_skus)} SKUs with exact column format...")

# Latest Metadata per SKU
latest_meta = df_full.groupby('sku').last().reset_index()
stock_map   = latest_meta.set_index('sku')['current_stock'].to_dict()
price_map   = latest_meta.set_index('sku')['selling_price'].to_dict()

# ── Function to Build Exact 6-Column Window Forecast ───────────────────────────
def build_exact_forecast_sheet(df_d, start_str, end_str, date_label):
    sub = df_d[(df_d['date'] >= start_str) & (df_d['date'] <= end_str)].copy()
    n_days = (pd.to_datetime(end_str) - pd.to_datetime(start_str)).days + 1
    
    sku_pred_map = {}
    for sku, grp in sub.groupby('sku'):
        tot_p = grp['predicted_daily'].sum()
        pred_int = int(np.round(tot_p))
        sigma_d = grp['sigma_daily'].iloc[0] if 'sigma_daily' in grp.columns else 1.0
        
        if pred_int > 0:
            ci = 1.96 * float(sigma_d) * np.sqrt(n_days)
            lower = max(0, int(np.round(pred_int - ci)))
            upper = int(np.round(pred_int + ci))
        else:
            lower = 0
            upper = 0
            
        sku_pred_map[sku] = {'pred': pred_int, 'lower': lower, 'upper': upper}
        
    rows = []
    for sku in all_588_skus:
        if sku in sku_pred_map:
            p_val = sku_pred_map[sku]['pred']
            l_val = sku_pred_map[sku]['lower']
            u_val = sku_pred_map[sku]['upper']
        else:
            p_val = 0
            l_val = 0
            u_val = 0
            
        rows.append({
            'date': date_label,
            'product': sku,
            'predicted value': p_val,
            'actual value': '',  # Left empty as requested
            'lower bound': l_val,
            'upper bound': u_val
        })
        
    cols = ['date', 'product', 'predicted value', 'actual value', 'lower bound', 'upper bound']
    return pd.DataFrame(rows)[cols].sort_values('predicted value', ascending=False).reset_index(drop=True)

df_sheet1 = build_exact_forecast_sheet(df_daily, '2026-08-01', '2026-08-11', '01/08/2026 to 11/08/2026')
df_sheet2 = build_exact_forecast_sheet(df_daily, '2026-08-01', '2026-08-31', '01/08/2026 to 31/08/2026')

# ── Build Sheet 3: Overstock & Clearance Action ─────────────────────────────────
daily_avg_map = df_daily.groupby('sku')['predicted_daily'].mean().to_dict()
tier_map      = df_daily.groupby('sku')['tier'].first().to_dict()

clearance_rows = []
for sku in all_588_skus:
    stock = float(stock_map.get(sku, 0) or 0)
    price = float(price_map.get(sku, 0) or 0)
    rate  = float(daily_avg_map.get(sku, 0) or 0)
    tier  = str(tier_map.get(sku, 'Discontinued'))
    val   = round(stock * price, 2)
    dos   = round(stock / rate, 1) if rate > 0 else (999.0 if stock > 0 else 0.0)

    if 'Discontinued' in tier:
        status = "Discontinued Line"
        action = "Zero sales in past year"
    elif stock == 0:
        status = "Out of Stock"
        action = "Zero warehouse stock"
    elif dos >= 180 and stock >= 20:
        status = "Critical Overstock (>180 Days)"
        action = f"Run 30%-50% clearance discount (Holds {int(dos)} days stock)"
    elif dos >= 90 and stock >= 15:
        status = "Overstock Warning (90-180 Days)"
        action = "Halt reorders; bundle in promotions"
    elif dos <= 15 and rate >= 1.0:
        status = "Low Stock Alert (<15 Days)"
        action = f"Reorder immediately (Only {int(dos)} days stock remaining)"
    else:
        status = "Healthy Stock"
        action = "Optimal inventory level"

    clearance_rows.append({
        'product': sku,
        'current stock': int(stock),
        'selling price (AED)': price,
        'daily sales velocity': round(rate, 2),
        'days of supply': dos,
        'trapped value (AED)': val,
        'status': status,
        'recommended action': action
    })

df_sheet3 = pd.DataFrame(clearance_rows).sort_values('days of supply', ascending=False).reset_index(drop=True)

# ── Save Function Handling File Locks ───────────────────────────────────────────
target_path = OUT_EXCEL
try:
    writer = pd.ExcelWriter(target_path, engine='xlsxwriter')
except PermissionError:
    print(f"Warning: {OUT_EXCEL} is currently open in Excel. Saving to {OUT_EXCEL_ALT} instead.")
    target_path = OUT_EXCEL_ALT
    writer = pd.ExcelWriter(target_path, engine='xlsxwriter')

df_sheet1.to_excel(writer, sheet_name='Aug 1-11 Forecast', index=False)
df_sheet2.to_excel(writer, sheet_name='Aug 1-31 Full Month', index=False)
df_sheet3.to_excel(writer, sheet_name='Overstock & Clearance Action', index=False)

wb = writer.book
hdr_fmt1 = wb.add_format({'bold': True, 'bg_color': '#1f497d', 'font_color': '#ffffff', 'border': 1, 'align': 'center'})
hdr_fmt2 = wb.add_format({'bold': True, 'bg_color': '#1e7145', 'font_color': '#ffffff', 'border': 1, 'align': 'center'})
hdr_fmt3 = wb.add_format({'bold': True, 'bg_color': '#a61c1c', 'font_color': '#ffffff', 'border': 1, 'align': 'center'})

# Sheet 1 Formatting
ws1 = writer.sheets['Aug 1-11 Forecast']
for i, col in enumerate(df_sheet1.columns):
    ws1.write(0, i, col, hdr_fmt1)
ws1.set_column('A:A', 26)  # date
ws1.set_column('B:B', 24)  # product
ws1.set_column('C:F', 18)  # predicted value, actual value, lower bound, upper bound
ws1.freeze_panes(1, 0)

# Sheet 2 Formatting
ws2 = writer.sheets['Aug 1-31 Full Month']
for i, col in enumerate(df_sheet2.columns):
    ws2.write(0, i, col, hdr_fmt2)
ws2.set_column('A:A', 26)
ws2.set_column('B:B', 24)
ws2.set_column('C:F', 18)
ws2.freeze_panes(1, 0)

# Sheet 3 Formatting
ws3 = writer.sheets['Overstock & Clearance Action']
for i, col in enumerate(df_sheet3.columns):
    ws3.write(0, i, col, hdr_fmt3)
ws3.set_column('A:A', 24)
ws3.set_column('B:F', 20)
ws3.set_column('G:H', 32)
ws3.freeze_panes(1, 0)

writer.close()

print("="*85)
print("EXCEL REPORT SAVED SUCCESSFULLY WITH EXACT 6 COLUMNS!")
print(f"File Saved: {target_path}")
print("\nSheet 1 Preview (First 10 Rows):")
print(df_sheet1.head(10).to_string(index=False))
print("="*85)
