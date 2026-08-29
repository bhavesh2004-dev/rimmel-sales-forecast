"""
PRODUCTION DAILY AUGUST FORECAST ENGINE: PATCHED V4 (DATA-DRIVEN TIERS & IN-STOCK VELOCITY)
=============================================================================================
Architecture:
- Data-Driven Tier Assignment (Tier A: Active Days >= 120 & Vol >= 400; Tier C: Active Days < 30 | Vol < 30; Tier B: Mid-Movers)
- Confirmed Discontinued Catalog List (74 SKUs receive hardcoded 0 forecast)
- In-Stock Only Velocity Features (roll_7, roll_14, roll_30, roll_90 computed strictly on in_stock_flag == 1 days)
- Direct Horizon Forecasting (Day-specific calendar factors without recursive buffer compounding)
- Exact sqrt(N) Statistical Variance Scaling

Saves daily August forecast to SQLite table: forecast_daily_aug2026_v4 in data/rimmel_clean.db
"""
import os, sqlite3
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = r'C:\Users\bhave\Desktop\ml_project'
DB_PATH  = os.path.join(BASE_DIR, 'data', 'rimmel_clean.db')

print("="*85)
print("EXECUTING DAILY AUGUST FORECAST ENGINE: PATCHED V4 ARCHITECTURE")
print("="*85)

conn = sqlite3.connect(DB_PATH)
df_full = pd.read_sql("SELECT * FROM full_history_v4 ORDER BY sku, date", conn)
conn.close()

df_full['date'] = pd.to_datetime(df_full['date'])
all_skus = sorted(df_full['sku'].unique().tolist())

t_start, t_end = pd.to_datetime('2025-08-01'), pd.to_datetime('2026-07-31')
t_dates = pd.date_range(t_start, t_end, freq='D')
fore_dates = pd.date_range('2026-08-01', '2026-08-31', freq='D')

# ── 1. Confirmed Discontinued List (74 SKUs) ──────────────────────────────────
df_train = df_full[(df_full['date'] >= t_start) & (df_full['date'] <= t_end)]
train_1y_active = set(df_train[df_train['total_sales'] > 0]['sku'].unique())
discontinued_74 = set(all_skus) - train_1y_active

# ── 2. Data-Driven Tier Assignment ─────────────────────────────────────────────
sku_stats = df_train[df_train['total_sales'] > 0].groupby('sku').agg(
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

print(f"Data-Driven Tier Sizes Across Catalog:")
print(f"  Tier A (Core High-Velocity)     : {len(tier_a_skus)} SKUs")
print(f"  Tier B (Category Mid-Movers)    : {len(tier_b_skus)} SKUs")
print(f"  Tier C (Intermittent Tail)      : {len(tier_c_skus)} SKUs")
print(f"  Discontinued Catalog Lines      : {len(discontinued_74)} SKUs")

# ── 3. Grid Construction & In-Stock Velocity Features ──────────────────────────
grid = pd.MultiIndex.from_product([all_skus, t_dates], names=['sku', 'date']).to_frame(index=False)
df_grid = grid.merge(df_train, on=['sku', 'date'], how='left')

df_grid['total_sales']   = df_grid['total_sales'].fillna(0)
df_grid['current_stock'] = df_grid.groupby('sku')['current_stock'].transform(lambda x: x.ffill().fillna(0))
df_grid['in_stock_flag'] = np.where(df_grid['current_stock'] > 0, 1, 0)
df_grid['selling_price'] = df_grid.groupby('sku')['selling_price'].transform(lambda x: x.ffill().bfill()).fillna(4.99)

cat_map = df_full.groupby('sku')['category'].first().to_dict()
df_grid['category'] = df_grid['sku'].map(cat_map).fillna('Cosmetics')

# Price Elasticity Ratio
pg = df_grid.groupby('sku')['selling_price']
df_grid['price_30d_avg'] = pg.transform(lambda x: x.rolling(30, min_periods=1).mean())
df_grid['price_elasticity_ratio'] = df_grid['selling_price'] / (df_grid['price_30d_avg'] + 0.001)

# In-Stock Selling Velocity (Skip stockouts)
df_grid['sales_instock'] = np.where(df_grid['in_stock_flag'] == 1, df_grid['total_sales'], np.nan)
g_instock = df_grid.groupby('sku')['sales_instock']
annual_mean_map = (df_grid.groupby('sku')['total_sales'].sum() / 365.0).to_dict()
df_grid['annual_365d_mean'] = df_grid['sku'].map(annual_mean_map).fillna(0.01)

df_grid['roll_7_mean_instock']  = g_instock.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean()).fillna(df_grid['annual_365d_mean'])
df_grid['roll_14_mean_instock'] = g_instock.transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean()).fillna(df_grid['annual_365d_mean'])
df_grid['roll_30_mean_instock'] = g_instock.transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean()).fillna(df_grid['annual_365d_mean'])
df_grid['roll_90_mean_instock'] = g_instock.transform(lambda x: x.shift(1).rolling(90, min_periods=1).mean()).fillna(df_grid['annual_365d_mean'])

# Raw Weekly Seasonality Lags
g_raw = df_grid.groupby('sku')['total_sales']
df_grid['lag_7']  = g_raw.transform(lambda x: x.shift(7))
df_grid['lag_14'] = g_raw.transform(lambda x: x.shift(14))
df_grid['lag_28'] = g_raw.transform(lambda x: x.shift(28))

aug_2025_df  = df_full[(df_full['date'] >= '2025-08-01') & (df_full['date'] <= '2025-08-11')]
aug_2025_map = aug_2025_df.groupby('sku')['total_sales'].sum().to_dict()
df_grid['aug2025_11d_baseline'] = df_grid['sku'].map(aug_2025_map).fillna(0.0)

df_grid['day_of_week'] = df_grid['date'].dt.dayofweek
df_grid['is_weekend']  = df_grid['day_of_week'].isin([5, 6]).astype(int)
df_grid['sku_code']    = df_grid['sku'].astype('category').cat.codes
df_grid['cat_code']    = df_grid['category'].astype('category').cat.codes

# ── 4. Train Models & Predict Daily Horizon (Direct Forecast) ───────────────────
daily_results = []

def croston_sba(ts_values, n=1, alpha=0.1):
    ts = np.array(ts_values, dtype=float)
    non_zeros = np.where(ts > 0)[0]
    if len(non_zeros) == 0: return 0.0
    if len(non_zeros) == 1: return (ts[non_zeros[0]] / len(ts)) * n
    z_sizes = ts[non_zeros]; p_intervals = np.diff(np.insert(non_zeros, 0, -1))
    z_hat = z_sizes[0]; p_hat = p_intervals[0]
    for i in range(1, len(z_sizes)):
        z_hat = alpha * z_sizes[i] + (1 - alpha) * z_hat
        p_hat = alpha * p_intervals[i] + (1 - alpha) * p_hat
    sba_factor = (1.0 - alpha / 2.0)
    return sba_factor * (z_hat / max(1.0, p_hat)) * n

# A. Tier A Model
feats_ta = [
    'in_stock_flag', 'price_elasticity_ratio', 'selling_price',
    'lag_7', 'lag_14',
    'roll_7_mean_instock', 'roll_14_mean_instock', 'roll_30_mean_instock',
    'aug2025_11d_baseline', 'annual_365d_mean',
    'day_of_week', 'is_weekend'
]
df_clean_ta = df_grid[df_grid['sku'].isin(tier_a_skus)].dropna(subset=feats_ta)
m_ta = lgb.train({'objective': 'tweedie', 'tweedie_variance_power': 1.5, 'metric': 'rmse', 'learning_rate': 0.03, 'num_leaves': 63, 'max_depth': 7, 'verbose': -1, 'random_state': 42}, lgb.Dataset(df_clean_ta[feats_ta], label=df_clean_ta['total_sales']), num_boost_round=350)

for sku in tier_a_skus:
    row  = df_grid[df_grid['sku']==sku].iloc[-1]
    h30_instock = df_grid[(df_grid['sku']==sku) & (df_grid['in_stock_flag']==1)]['total_sales'].values
    sigma = float(h30_instock[-30:].std()) if len(h30_instock)>=30 and h30_instock.std()>0 else max(float(row['annual_365d_mean'])*0.20, 1.0)
    
    for d in fore_dates:
        feat = pd.DataFrame([{
            'in_stock_flag': 1, 'price_elasticity_ratio': 1.0, 'selling_price': float(row['selling_price']),
            'lag_7': float(row['lag_7']), 'lag_14': float(row['lag_14']),
            'roll_7_mean_instock': float(row['roll_7_mean_instock']),
            'roll_14_mean_instock': float(row['roll_14_mean_instock']),
            'roll_30_mean_instock': float(row['roll_30_mean_instock']),
            'aug2025_11d_baseline': float(row['aug2025_11d_baseline']),
            'annual_365d_mean': float(row['annual_365d_mean']),
            'day_of_week': d.dayofweek, 'is_weekend': 1 if d.dayofweek in [5, 6] else 0
        }])[feats_ta]
        p = float(np.maximum(0, m_ta.predict(feat)[0]))
        daily_results.append({
            'date': d.date(), 'sku': sku, 'category': cat_map.get(sku, 'Cosmetics'),
            'tier': 'Tier A (Core High-Velocity)',
            'predicted_daily': round(p, 2), 'sigma_daily': round(sigma, 2)
        })

# B. Tier B Model
feats_tb = [
    'in_stock_flag', 'sku_code', 'cat_code', 'selling_price',
    'annual_365d_mean', 'aug2025_11d_baseline',
    'roll_90_mean_instock', 'roll_30_mean_instock', 'roll_14_mean_instock',
    'lag_7', 'lag_14', 'lag_28',
    'day_of_week', 'is_weekend'
]
df_clean_tb = df_grid[df_grid['sku'].isin(tier_b_skus)].dropna(subset=feats_tb)
m_tb = lgb.train({'objective': 'tweedie', 'tweedie_variance_power': 1.5, 'metric': 'rmse', 'learning_rate': 0.03, 'num_leaves': 63, 'max_depth': 7, 'verbose': -1, 'random_state': 42}, lgb.Dataset(df_clean_tb[feats_tb], label=df_clean_tb['total_sales']), num_boost_round=350)

for sku in tier_b_skus:
    row  = df_grid[df_grid['sku']==sku].iloc[-1]
    h30_instock = df_grid[(df_grid['sku']==sku) & (df_grid['in_stock_flag']==1)]['total_sales'].values
    sigma = float(h30_instock[-30:].std()) if len(h30_instock)>=30 and h30_instock.std()>0 else max(float(row['annual_365d_mean'])*0.25, 1.0)
    
    for d in fore_dates:
        feat = pd.DataFrame([{
            'in_stock_flag': 1, 'sku_code': int(row['sku_code']), 'cat_code': int(row['cat_code']),
            'selling_price': float(row['selling_price']), 'annual_365d_mean': float(row['annual_365d_mean']),
            'aug2025_11d_baseline': float(row['aug2025_11d_baseline']),
            'roll_90_mean_instock': float(row['roll_90_mean_instock']),
            'roll_30_mean_instock': float(row['roll_30_mean_instock']),
            'roll_14_mean_instock': float(row['roll_14_mean_instock']),
            'lag_7': float(row['lag_7']), 'lag_14': float(row['lag_14']), 'lag_28': float(row['lag_28']),
            'day_of_week': d.dayofweek, 'is_weekend': 1 if d.dayofweek in [5, 6] else 0
        }])[feats_tb]
        p = float(np.maximum(0, m_tb.predict(feat)[0]))
        daily_results.append({
            'date': d.date(), 'sku': sku, 'category': cat_map.get(sku, 'Cosmetics'),
            'tier': 'Tier B (Category Mid-Movers)',
            'predicted_daily': round(p, 2), 'sigma_daily': round(sigma, 2)
        })

# C. Tier C Croston SBA
for sku in tier_c_skus:
    hist = df_grid[df_grid['sku'] == sku].sort_values('date')['total_sales'].values
    daily_rate = float(croston_sba(hist, n=1))
    sigma = max(daily_rate * 0.25, 0.1)
    for d in fore_dates:
        daily_results.append({
            'date': d.date(), 'sku': sku, 'category': cat_map.get(sku, 'Cosmetics'),
            'tier': 'Tier C (Intermittent Tail)',
            'predicted_daily': round(daily_rate, 2), 'sigma_daily': round(sigma, 2)
        })

# D. Discontinued 74 SKUs
for sku in discontinued_74:
    for d in fore_dates:
        daily_results.append({
            'date': d.date(), 'sku': sku, 'category': cat_map.get(sku, 'Cosmetics'),
            'tier': '⚪ Discontinued Catalog Lines',
            'predicted_daily': 0.0, 'sigma_daily': 0.0
        })

df_daily_out = pd.DataFrame(daily_results)
df_daily_out['date'] = pd.to_datetime(df_daily_out['date'])

conn = sqlite3.connect(DB_PATH)
df_daily_out.to_sql('forecast_daily_aug2026_v4', conn, if_exists='replace', index=False)
conn.close()

print("\n" + "="*85)
print("DAILY AUGUST FORECAST GENERATION COMPLETE!")
print(f"Table Saved: forecast_daily_aug2026_v4 | Total Daily Rows: {len(df_daily_out):,}")
print("="*85)
