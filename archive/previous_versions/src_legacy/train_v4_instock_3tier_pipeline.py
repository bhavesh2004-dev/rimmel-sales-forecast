"""
PATCHED V4 PRODUCTION ENGINE: DATA-DRIVEN TIERS & IN-STOCK VELOCITY (DIRECT HORIZON)
====================================================================================
Architecture:
- Data-Driven Tier Assignment (Tier A: Active Days >= 120 & Vol >= 400; Tier C: Active Days < 30 | Vol < 30; Tier B: Mid-Movers)
- Confirmed Discontinued Catalog List (74 SKUs receive hardcoded 0 forecast)
- Active In-Stock Velocity Features (roll_7, roll_14, roll_30, roll_90 computed strictly over in_stock_flag == 1 days)
- Direct Horizon Forecasting (Eliminates recursive buffer compounding)
- Exact sqrt(N) Statistical Confidence Interval Scaling

Outputs:
1. SQLite Table: forecast_aug_2026_v4_instock_master in data/rimmel_clean.db
2. Excel Report: reports/Rimmel_v4_InStock_3Tier_Forecast.xlsx
"""
import os
import sqlite3
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

BASE_DIR  = r'C:\Users\bhave\Desktop\ml_project'
DB_PATH   = os.path.join(BASE_DIR, 'data', 'rimmel_clean.db')
OUT_EXCEL = os.path.join(BASE_DIR, 'reports', 'Rimmel_v4_InStock_3Tier_Forecast.xlsx')

os.makedirs(os.path.join(BASE_DIR, 'reports'), exist_ok=True)

print("="*85)
print("EXECUTING PATCHED V4 PRODUCTION ENGINE (DATA-DRIVEN TIERS + IN-STOCK VELOCITY)")
print("="*85)

conn = sqlite3.connect(DB_PATH)
df_train_raw = pd.read_sql("SELECT * FROM training_window_v4 ORDER BY sku, date", conn)
df_full_raw  = pd.read_sql("SELECT * FROM full_history_v4 ORDER BY sku, date", conn)
conn.close()

df_train_raw['date'] = pd.to_datetime(df_train_raw['date'])
df_full_raw['date']  = pd.to_datetime(df_full_raw['date'])

all_skus  = sorted(df_full_raw['sku'].unique().tolist())
all_dates = pd.date_range('2025-08-01', '2026-07-31', freq='D')

# ── 1. Confirmed Discontinued List (74 SKUs) ──────────────────────────────────
train_1y_sales = df_train_raw[df_train_raw['total_sales'] > 0]['sku'].unique()
discontinued_74 = set(all_skus) - set(train_1y_sales)

# ── 2. Data-Driven Tier Assignment ─────────────────────────────────────────────
sku_stats = df_train_raw[df_train_raw['total_sales'] > 0].groupby('sku').agg(
    active_days=('date', 'nunique'),
    total_vol=('total_sales', 'sum')
)

tier_a_skus = set()
tier_b_skus = set()
tier_c_skus = set()

for s in all_skus:
    if s in discontinued_74 or s not in sku_stats.index:
        continue
    days = sku_stats.loc[s, 'active_days']
    vol  = sku_stats.loc[s, 'total_vol']
    if days >= 120 and vol >= 400:
        tier_a_skus.add(s)
    elif days < 30 or vol < 30:
        tier_c_skus.add(s)
    else:
        tier_b_skus.add(s)

print(f"Data-Driven Catalog Tiers (365-Day Window: 1 Aug 2025 - 31 Jul 2026):")
print(f"  Tier A (Core High-Velocity)     : {len(tier_a_skus)} SKUs")
print(f"  Tier B (Category Mid-Movers)    : {len(tier_b_skus)} SKUs")
print(f"  Tier C (Intermittent Tail)      : {len(tier_c_skus)} SKUs")
print(f"  Discontinued Catalog Lines      : {len(discontinued_74)} SKUs")

# ── 3. Grid Construction & In-Stock Velocity Features ──────────────────────────
grid = pd.MultiIndex.from_product([all_skus, all_dates], names=['sku', 'date']).to_frame(index=False)
df_grid = grid.merge(df_train_raw[['sku', 'date', 'total_sales', 'selling_price', 'in_stock_flag', 'category']], on=['sku', 'date'], how='left')

df_grid['total_sales']   = df_grid['total_sales'].fillna(0)
df_grid['in_stock_flag'] = df_grid['in_stock_flag'].fillna(1)
df_grid['selling_price'] = df_grid.groupby('sku')['selling_price'].transform(lambda x: x.ffill().bfill()).fillna(4.99)

cat_map_dict = df_train_raw.groupby('sku')['category'].first().to_dict()
df_grid['category'] = df_grid['sku'].map(cat_map_dict).fillna('Cosmetics')

# Price Elasticity (Rolling 30-Day Normalized)
pg = df_grid.groupby('sku')['selling_price']
df_grid['price_30d_avg'] = pg.transform(lambda x: x.rolling(30, min_periods=1).mean())
df_grid['price_elasticity_ratio'] = df_grid['selling_price'] / (df_grid['price_30d_avg'] + 0.001)

# In-Stock Only Selling Velocity (Skips forced zero-sales days during stockouts)
df_grid['sales_instock'] = np.where(df_grid['in_stock_flag'] == 1, df_grid['total_sales'], np.nan)
g_instock = df_grid.groupby('sku')['sales_instock']
annual_365d_map = (df_train_raw.groupby('sku')['total_sales'].sum() / 365.0).to_dict()
df_grid['annual_365d_mean'] = df_grid['sku'].map(annual_365d_map).fillna(0.01)

df_grid['roll_7_mean_instock']  = g_instock.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean()).fillna(df_grid['annual_365d_mean'])
df_grid['roll_14_mean_instock'] = g_instock.transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean()).fillna(df_grid['annual_365d_mean'])
df_grid['roll_30_mean_instock'] = g_instock.transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean()).fillna(df_grid['annual_365d_mean'])
df_grid['roll_90_mean_instock'] = g_instock.transform(lambda x: x.shift(1).rolling(90, min_periods=1).mean()).fillna(df_grid['annual_365d_mean'])

# Raw Weekly Seasonality Lags
g_raw = df_grid.groupby('sku')['total_sales']
df_grid['lag_7']  = g_raw.transform(lambda x: x.shift(7))
df_grid['lag_14'] = g_raw.transform(lambda x: x.shift(14))
df_grid['lag_28'] = g_raw.transform(lambda x: x.shift(28))

aug_2025_df  = df_full_raw[(df_full_raw['date'] >= '2025-08-01') & (df_full_raw['date'] <= '2025-08-11')]
aug_2025_map = aug_2025_df.groupby('sku')['total_sales'].sum().to_dict()
df_grid['aug2025_11d_baseline'] = df_grid['sku'].map(aug_2025_map).fillna(0.0)

df_grid['day_of_week'] = df_grid['date'].dt.dayofweek
df_grid['is_weekend']  = df_grid['day_of_week'].isin([5, 6]).astype(int)
df_grid['sku_code']    = df_grid['sku'].astype('category').cat.codes
df_grid['cat_code']    = df_grid['category'].astype('category').cat.codes

FORECAST_DATES = pd.date_range('2026-08-01', '2026-08-11', freq='D')
forecast_results = []

# ── 4. ENGINE 1: Tier A Core High-Velocity Engine ───────────────────────────────
print("\nTraining Tier A Core Engine with In-Stock Velocity Features...")
tier_a_features = [
    'in_stock_flag', 'price_elasticity_ratio', 'selling_price',
    'lag_7', 'lag_14',
    'roll_7_mean_instock', 'roll_14_mean_instock', 'roll_30_mean_instock',
    'aug2025_11d_baseline', 'annual_365d_mean',
    'day_of_week', 'is_weekend'
]
df_clean_ta = df_grid[df_grid['sku'].isin(tier_a_skus)].dropna(subset=tier_a_features).copy()
dtrain_ta   = lgb.Dataset(df_clean_ta[tier_a_features], label=df_clean_ta['total_sales'])

params_ta = {'objective': 'tweedie', 'tweedie_variance_power': 1.5, 'metric': 'rmse', 'learning_rate': 0.03, 'num_leaves': 63, 'max_depth': 7, 'verbose': -1, 'random_state': 42}
model_ta  = lgb.train(params_ta, dtrain_ta, num_boost_round=350)

# Direct Horizon evaluation (No recursive buffer append)
for sku in tier_a_skus:
    row  = df_grid[df_grid['sku']==sku].iloc[-1]
    hist = df_grid[df_grid['sku']==sku]['total_sales'].tolist()
    daily_preds = []
    
    for d in FORECAST_DATES:
        feat = pd.DataFrame([{
            'in_stock_flag': 1, 'price_elasticity_ratio': 1.0, 'selling_price': float(row['selling_price']),
            'lag_7': float(row['lag_7']), 'lag_14': float(row['lag_14']),
            'roll_7_mean_instock': float(row['roll_7_mean_instock']),
            'roll_14_mean_instock': float(row['roll_14_mean_instock']),
            'roll_30_mean_instock': float(row['roll_30_mean_instock']),
            'aug2025_11d_baseline': float(row['aug2025_11d_baseline']),
            'annual_365d_mean': float(row['annual_365d_mean']),
            'day_of_week': d.dayofweek, 'is_weekend': 1 if d.dayofweek in [5, 6] else 0
        }])[tier_a_features]
        
        p = float(np.maximum(0, model_ta.predict(feat)[0]))
        daily_preds.append(p)
        
    tot_11d = int(np.round(np.sum(daily_preds)))
    h30_instock = df_grid[(df_grid['sku']==sku) & (df_grid['in_stock_flag']==1)]['total_sales'].values
    sigma = float(h30_instock[-30:].std()) if len(h30_instock)>=30 and h30_instock.std()>0 else max(float(tot_11d / 11.0)*0.20, 1.0)
    ci = 1.96 * sigma * np.sqrt(11)

    forecast_results.append({
        'date': '1/08/2026 to 11/08/2026',
        'product': sku,
        'predicted': tot_11d,
        'lower bound': max(0, int(np.round(tot_11d - ci))),
        'upper bound': int(np.round(tot_11d + ci))
    })

# ── 5. ENGINE 2: Tier B Category Mid-Movers Engine ─────────────────────────────
print("Training Tier B Mid-Movers Engine with In-Stock Velocity Features...")
tier_b_features = [
    'in_stock_flag', 'sku_code', 'cat_code', 'selling_price',
    'annual_365d_mean', 'aug2025_11d_baseline',
    'roll_90_mean_instock', 'roll_30_mean_instock', 'roll_14_mean_instock',
    'lag_7', 'lag_14', 'lag_28',
    'day_of_week', 'is_weekend'
]
df_clean_tb = df_grid[df_grid['sku'].isin(tier_b_skus)].dropna(subset=tier_b_features).copy()
dtrain_tb   = lgb.Dataset(df_clean_tb[tier_b_features], label=df_clean_tb['total_sales'])

params_tb = {'objective': 'tweedie', 'tweedie_variance_power': 1.5, 'metric': 'rmse', 'learning_rate': 0.03, 'num_leaves': 63, 'max_depth': 7, 'verbose': -1, 'random_state': 42}
model_tb  = lgb.train(params_tb, dtrain_tb, num_boost_round=350)

# Direct Horizon evaluation (No recursive buffer append)
for sku in tier_b_skus:
    row  = df_grid[df_grid['sku']==sku].iloc[-1]
    hist = df_grid[df_grid['sku']==sku]['total_sales'].tolist()
    daily_preds = []
    
    for d in FORECAST_DATES:
        feat = pd.DataFrame([{
            'in_stock_flag': 1, 'sku_code': int(row['sku_code']), 'cat_code': int(row['cat_code']),
            'selling_price': float(row['selling_price']), 'annual_365d_mean': float(row['annual_365d_mean']),
            'aug2025_11d_baseline': float(row['aug2025_11d_baseline']),
            'roll_90_mean_instock': float(row['roll_90_mean_instock']),
            'roll_30_mean_instock': float(row['roll_30_mean_instock']),
            'roll_14_mean_instock': float(row['roll_14_mean_instock']),
            'lag_7': float(row['lag_7']), 'lag_14': float(row['lag_14']), 'lag_28': float(row['lag_28']),
            'day_of_week': d.dayofweek, 'is_weekend': 1 if d.dayofweek in [5, 6] else 0
        }])[tier_b_features]
        
        p = float(np.maximum(0, model_tb.predict(feat)[0]))
        daily_preds.append(p)
        
    tot_11d = int(np.round(np.sum(daily_preds)))
    h30_instock = df_grid[(df_grid['sku']==sku) & (df_grid['in_stock_flag']==1)]['total_sales'].values
    sigma = float(h30_instock[-30:].std()) if len(h30_instock)>=30 and h30_instock.std()>0 else max(float(tot_11d / 11.0)*0.25, 1.0)
    ci = 1.96 * sigma * np.sqrt(11)
    
    forecast_results.append({
        'date': '1/08/2026 to 11/08/2026',
        'product': sku,
        'predicted': tot_11d,
        'lower bound': max(0, int(np.round(tot_11d - ci))),
        'upper bound': int(np.round(tot_11d + ci))
    })

# ── 6. ENGINE 3: Tier C Intermittent Demand Engine (Croston SBA) ────────────────
print("Executing Tier C Engine (Croston's SBA Intermittent Demand Engine)...")
def croston_sba(ts_values, forecast_len=11, alpha=0.1):
    ts = np.array(ts_values, dtype=float)
    non_zeros = np.where(ts > 0)[0]
    if len(non_zeros) == 0: return 0.0
    if len(non_zeros) == 1: return (ts[non_zeros[0]] / len(ts)) * forecast_len
    z_sizes = ts[non_zeros]; p_intervals = np.diff(np.insert(non_zeros, 0, -1))
    z_hat = z_sizes[0]; p_hat = p_intervals[0]
    for i in range(1, len(z_sizes)):
        z_hat = alpha * z_sizes[i] + (1 - alpha) * z_hat
        p_hat = alpha * p_intervals[i] + (1 - alpha) * p_hat
    sba_factor = (1.0 - alpha / 2.0)
    return sba_factor * (z_hat / max(1.0, p_hat)) * forecast_len

for sku in tier_c_skus:
    hist = df_grid[df_grid['sku'] == sku].sort_values('date')['total_sales'].values
    raw_tot = croston_sba(hist, forecast_len=11, alpha=0.1)
    pred_val = int(np.round(raw_tot))
    
    forecast_results.append({
        'date': '1/08/2026 to 11/08/2026',
        'product': sku,
        'predicted': pred_val,
        'lower bound': max(0, int(np.round(pred_val * 0.6))),
        'upper bound': int(np.round(pred_val * 1.5))
    })

# ── 7. Confirmed Discontinued Lines (Hardcoded Zero) ───────────────────────────
for sku in discontinued_74:
    forecast_results.append({
        'date': '1/08/2026 to 11/08/2026',
        'product': sku,
        'predicted': 0,
        'lower bound': 0,
        'upper bound': 0
    })

df_out = pd.DataFrame(forecast_results).sort_values('predicted', ascending=False).reset_index(drop=True)
column_order = ['date', 'product', 'predicted', 'lower bound', 'upper bound']
df_out = df_out[column_order]

# Save to SQLite Table
conn = sqlite3.connect(DB_PATH)
df_out.to_sql('forecast_aug_2026_v4_instock_master', conn, if_exists='replace', index=False)
conn.close()

# Export Excel with Formatting
with pd.ExcelWriter(OUT_EXCEL, engine='xlsxwriter') as writer:
    df_out.to_excel(writer, sheet_name='Sheet1', index=False)
    wb = writer.book
    ws = writer.sheets['Sheet1']
    hdr = wb.add_format({'bold': True, 'bg_color': '#1a73e8', 'font_color': '#ffffff', 'border': 1})
    for i, c in enumerate(df_out.columns):
        ws.write(0, i, c, hdr)
    ws.set_column('A:A', 28)
    ws.set_column('B:B', 26)
    ws.set_column('C:E', 18)
    ws.freeze_panes(1, 0)

print("\n" + "="*85)
print("PATCHED V4 IN-STOCK FORECAST COMPLETE!")
print("="*85)
print(f"Report File Saved                    : {OUT_EXCEL}")
print(f"Total August 1-11 Forecast Volume    : {df_out['predicted'].sum():,} units")
print(f"SQLite Table Saved                   : forecast_aug_2026_v4_instock_master ({DB_PATH})")
print("\nTop 15 Forecast Preview (Exact 5-Column Format):")
print(df_out.head(15).to_string(index=False))
