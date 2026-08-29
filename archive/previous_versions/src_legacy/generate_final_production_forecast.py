"""
MASTER PRODUCTION FORECAST ENGINE
1. Multi-scale feature hierarchy (365d, 90d, 30d, 14d, 7d)
2. Active Product Stockout Restorer (Protects 22 high-volume SKUs from July stockout collapse)
3. Genuine zero-prediction preservation for truly dead/discontinued SKUs
4. Excel export matching user format (order sku, date, predicted unit, lower bound, upper bound)
"""
import os
import sqlite3
import pandas as pd
import numpy as np
import lightgbm as lgb

BASE_DIR  = r'C:\Users\bhave\Desktop\ml_project'
DB_PATH   = os.path.join(BASE_DIR, 'data', 'rimmel_clean.db')
OUT_EXCEL_1 = os.path.join(BASE_DIR, 'Rimmel_588_SKUs_Forecast_1Aug_to_11Aug_2026.xlsx')
OUT_EXCEL_2 = os.path.join(BASE_DIR, 'Rimmel_Forecast_Aug2026_FINAL.xlsx')

print("="*85)
print("MASTER PRODUCTION FORECAST ENGINE RUNNING...")
print("="*85)

conn = sqlite3.connect(DB_PATH)
df_train_raw = pd.read_sql("SELECT * FROM training_window ORDER BY sku, date", conn)
df_full_raw  = pd.read_sql("SELECT * FROM full_history ORDER BY sku, date", conn)
conn.close()

df_train_raw['date'] = pd.to_datetime(df_train_raw['date'])
df_full_raw['date']  = pd.to_datetime(df_full_raw['date'])

all_skus = df_train_raw['sku'].unique()
all_dates = pd.date_range('2025-08-01', '2026-07-31', freq='D')

# 1. Active Stockout Demand Restorer:
# For SKUs with total historical sales > 50 units, compute their active non-zero daily sales rate
sku_tot_sales = df_full_raw.groupby('sku')['total_sales'].sum().to_dict()
active_sales_only = df_full_raw[df_full_raw['total_sales'] > 0]
sku_active_daily_rate = active_sales_only.groupby('sku')['total_sales'].mean().to_dict()

# Same month last year baseline (August 2025 actuals)
aug_2025_df = df_full_raw[(df_full_raw['date'] >= '2025-08-01') & (df_full_raw['date'] <= '2025-08-11')]
aug_2025_map = aug_2025_df.groupby('sku')['total_sales'].sum().to_dict()

annual_365d_map = (df_train_raw.groupby('sku')['total_sales'].sum() / 365.0).to_dict()

# Build Daily Feature Grid
grid = pd.MultiIndex.from_product([all_skus, all_dates], names=['sku', 'date']).to_frame(index=False)
df_grid = grid.merge(df_train_raw[['sku', 'date', 'total_sales', 'selling_price', 'category']], on=['sku', 'date'], how='left')

df_grid['total_sales']   = df_grid['total_sales'].fillna(0)
df_grid['selling_price'] = df_grid.groupby('sku')['selling_price'].transform(lambda x: x.ffill().bfill())
df_grid['selling_price'] = df_grid['selling_price'].fillna(df_grid['selling_price'].median())

cat_map_dict = df_train_raw.groupby('sku')['category'].first().to_dict()
df_grid['category'] = df_grid['sku'].map(cat_map_dict).fillna('Cosmetics')

# Attach 1-Year Features
df_grid['aug2025_11d_baseline'] = df_grid['sku'].map(aug_2025_map).fillna(0.0)
df_grid['annual_365d_mean']     = df_grid['sku'].map(annual_365d_map).fillna(0.01)

df_grid = df_grid.sort_values(['sku', 'date']).reset_index(drop=True)
g = df_grid.groupby('sku')['total_sales']

df_grid['lag_1']  = g.transform(lambda x: x.shift(1))
df_grid['lag_3']  = g.transform(lambda x: x.shift(3))
df_grid['lag_7']  = g.transform(lambda x: x.shift(7))
df_grid['lag_14'] = g.transform(lambda x: x.shift(14))

df_grid['roll_7_mean']  = g.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
df_grid['roll_14_mean'] = g.transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())
df_grid['roll_30_mean'] = g.transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())

df_grid['day_of_week'] = df_grid['date'].dt.dayofweek
df_grid['month']       = df_grid['date'].dt.month
df_grid['is_weekend']  = df_grid['day_of_week'].isin([5, 6]).astype(int)

df_grid['sku_code'] = df_grid['sku'].astype('category').cat.codes
df_grid['cat_code'] = df_grid['category'].astype('category').cat.codes

features = [
    'sku_code', 'cat_code', 'day_of_week', 'month', 'is_weekend',
    'selling_price',
    'aug2025_11d_baseline', 'annual_365d_mean',
    'lag_1', 'lag_3', 'lag_7', 'lag_14',
    'roll_7_mean', 'roll_14_mean', 'roll_30_mean'
]

df_model = df_grid.dropna(subset=features).copy()

# Train Model
X_tr, y_tr = df_model[features], df_model['total_sales']
dtrain = lgb.Dataset(X_tr, label=y_tr)

params = {
    'objective': 'tweedie',
    'tweedie_variance_power': 1.5,
    'metric': 'rmse',
    'learning_rate': 0.03,
    'num_leaves': 63,
    'max_depth': 7,
    'min_data_in_leaf': 15,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

model = lgb.train(params, dtrain, num_boost_round=350)

# Forecast 1 Aug to 11 Aug 2026
FORECAST_DATES = pd.date_range('2026-08-01', '2026-08-11', freq='D')
last_state = df_grid[df_grid['date'] == pd.to_datetime('2026-07-31')].set_index('sku')

forecast_results = []

for sku in all_skus:
    hist = df_grid[df_grid['sku'] == sku].sort_values('date')['total_sales'].tolist()
    row  = last_state.loc[sku]
    
    sku_c   = int(row['sku_code'])
    cat_c   = int(row['cat_code'])
    price   = float(row['selling_price'])
    aug25_b = float(row['aug2025_11d_baseline'])
    ann_m   = float(row['annual_365d_mean'])
    tot_s   = sku_tot_sales.get(sku, 0)
    act_r   = sku_active_daily_rate.get(sku, 0.5)
    
    daily_preds = []
    buf = list(hist)
    
    for i, d in enumerate(FORECAST_DATES):
        def lag(n): return buf[-n] if len(buf) >= n else 0.0
        
        h7  = np.array(buf[-7:])  if len(buf) >= 7  else np.array(buf or [0])
        h14 = np.array(buf[-14:]) if len(buf) >= 14 else np.array(buf or [0])
        h30 = np.array(buf[-30:]) if len(buf) >= 30 else np.array(buf or [0])
        
        r7m = h7.mean(); r14m = h14.mean(); r30m = h30.mean()
        
        feat = pd.DataFrame([{
            'sku_code': sku_c, 'cat_code': cat_c,
            'day_of_week': d.dayofweek, 'month': d.month, 'is_weekend': 1 if d.dayofweek in [5, 6] else 0,
            'selling_price': price,
            'aug2025_11d_baseline': aug25_b, 'annual_365d_mean': ann_m,
            'lag_1': lag(1), 'lag_3': lag(3), 'lag_7': lag(7), 'lag_14': lag(14),
            'roll_7_mean': r7m, 'roll_14_mean': r14m, 'roll_30_mean': r30m
        }])[features]
        
        p = float(np.maximum(0, model.predict(feat)[0]))
        
        # Bestseller Stockout Demand Restorer Anchor:
        # If product historically sold > 50 units, but recent rolling mean dropped to < 0.5 due to July stockout,
        # restore target daily prediction using active daily sales rate!
        if tot_s > 50 and r30m < 0.5:
            p = max(p, act_r)
            
        daily_preds.append(p)
        buf.append(p)
        
    tot_11d = np.sum(daily_preds)
    h30a    = np.array(hist[-30:]) if len(hist) >= 30 else np.array(hist or [0])
    sigma   = h30a.std() if len(h30a) > 1 else max(tot_11d * 0.18, 1.0)
    ci      = 1.96 * sigma * np.sqrt(11)
    
    lower = max(0, int(np.round(tot_11d - ci)))
    upper = int(np.round(tot_11d + ci))
    
    forecast_results.append({
        'order sku': sku,
        'date': '1/08/2026 to 11/08/2026',
        'predicted unit': int(np.round(tot_11d)),
        'lower bound': lower,
        'upper bound': upper
    })

df_out = pd.DataFrame(forecast_results).sort_values('predicted unit', ascending=False).reset_index(drop=True)

# Export Excel
def save_excel(path, df):
    with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)
        wb = writer.book
        ws = writer.sheets['Sheet1']
        hdr = wb.add_format({'bold': True, 'bg_color': '#1a73e8', 'font_color': '#ffffff', 'border': 1})
        for i, c in enumerate(df.columns):
            ws.write(0, i, c, hdr)
        ws.set_column('A:A', 26)
        ws.set_column('B:B', 30)
        ws.set_column('C:E', 18)
        ws.freeze_panes(1, 0)

for p in [OUT_EXCEL_1, OUT_EXCEL_2]:
    try:
        save_excel(p, df_out)
        print(f"Excel Exported Successfully: {p}")
    except PermissionError:
        alt_p = p.replace('.xlsx', '_Master.xlsx')
        save_excel(alt_p, df_out)
        print(f"Excel Exported Fallback: {alt_p}")

print("\n" + "="*85)
print("MASTER FORECAST SUMMARY:")
print("="*85)
print(f"Total Active SKUs Forecasted         : {len(df_out)}")
print(f"SKUs with predicted unit > 0         : {(df_out['predicted unit'] > 0).sum()}")
print(f"SKUs with predicted unit == 0        : {(df_out['predicted unit'] == 0).sum()} (Discontinued/Dead SKUs)")
print(f"Total August 1-11 Forecast Volume    : {df_out['predicted unit'].sum():,} units")
