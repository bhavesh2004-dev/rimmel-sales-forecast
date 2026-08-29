"""
TIER-TAILORED PRODUCTION FORECAST ENGINE (WITH ENHANCED TIER B 365-DAY BASELINE)
================================================================================
Applies optimal tier-specific features:
- Tier A (Top 20 Bestsellers): Price Elasticity Ratio + Bi-Weekly Lags + 1-Year Baseline
- Tier B (Next 100 Mid-Movers): Annual 365d Baseline + August 2025 Baseline + 90d Trend + Multi-Lag
- Tier C (Tail 394 Slow-Movers): Croston's SBA Intermittent Demand Engine

Exports to: Rimmel_TierTailored_Aug1_to_11_2026_Report.xlsx
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
OUT_EXCEL = os.path.join(BASE_DIR, 'Rimmel_TierTailored_Aug1_to_11_2026_Report.xlsx')

print("="*85)
print("RUNNING TIER-TAILORED PRODUCTION FORECAST ENGINE (ENHANCED TIER B)")
print("="*85)

conn = sqlite3.connect(DB_PATH)
df_train_raw = pd.read_sql("SELECT * FROM training_window ORDER BY sku, date", conn)
df_full_raw  = pd.read_sql("SELECT * FROM full_history ORDER BY sku, date", conn)
conn.close()

df_train_raw['date'] = pd.to_datetime(df_train_raw['date'])
df_full_raw['date']  = pd.to_datetime(df_full_raw['date'])

all_skus  = df_train_raw['sku'].unique()
all_dates = pd.date_range('2025-08-01', '2026-07-31', freq='D')

sku_1y_totals = df_train_raw.groupby('sku')['total_sales'].sum().sort_values(ascending=False)
top_20_skus   = set(sku_1y_totals.head(20).index)
next_100_skus  = set(sku_1y_totals.iloc[20:120].index)
tail_400_skus  = set(sku_1y_totals.iloc[120:].index)

# Grid Construction
grid = pd.MultiIndex.from_product([all_skus, all_dates], names=['sku', 'date']).to_frame(index=False)
df_grid = grid.merge(df_train_raw[['sku', 'date', 'total_sales', 'selling_price', 'category']], on=['sku', 'date'], how='left')

df_grid['total_sales']   = df_grid['total_sales'].fillna(0)
df_grid['selling_price'] = df_grid.groupby('sku')['selling_price'].transform(lambda x: x.ffill().bfill())
df_grid['selling_price'] = df_grid['selling_price'].fillna(df_grid['selling_price'].median())

cat_map_dict = df_train_raw.groupby('sku')['category'].first().to_dict()
df_grid['category'] = df_grid['sku'].map(cat_map_dict).fillna('Cosmetics')

# Price Elasticity Ratio (for Tier A)
pg = df_grid.groupby('sku')['selling_price']
df_grid['price_30d_avg'] = pg.transform(lambda x: x.rolling(30, min_periods=1).mean())
df_grid['price_elasticity_ratio'] = df_grid['selling_price'] / (df_grid['price_30d_avg'] + 0.001)

# August 2025 Baseline & 365d Annual Mean (FOR BOTH TIER A & TIER B)
aug_2025_df = df_full_raw[(df_full_raw['date'] >= '2025-08-01') & (df_full_raw['date'] <= '2025-08-11')]
aug_2025_map = aug_2025_df.groupby('sku')['total_sales'].sum().to_dict()
annual_365d_map = (df_train_raw.groupby('sku')['total_sales'].sum() / 365.0).to_dict()

df_grid['aug2025_11d_baseline'] = df_grid['sku'].map(aug_2025_map).fillna(0.0)
df_grid['annual_365d_mean']     = df_grid['sku'].map(annual_365d_map).fillna(0.01)

df_grid = df_grid.sort_values(['sku', 'date']).reset_index(drop=True)
g = df_grid.groupby('sku')['total_sales']

df_grid['roll_365_mean'] = g.transform(lambda x: x.shift(1).rolling(365, min_periods=30).mean())
df_grid['roll_90_mean']  = g.transform(lambda x: x.shift(1).rolling(90, min_periods=14).mean())
df_grid['roll_30_mean']  = g.transform(lambda x: x.shift(1).rolling(30, min_periods=7).mean())
df_grid['roll_14_mean']  = g.transform(lambda x: x.shift(1).rolling(14, min_periods=3).mean())
df_grid['roll_7_mean']   = g.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())

df_grid['lag_1']  = g.transform(lambda x: x.shift(1))
df_grid['lag_3']  = g.transform(lambda x: x.shift(3))
df_grid['lag_7']  = g.transform(lambda x: x.shift(7))
df_grid['lag_14'] = g.transform(lambda x: x.shift(14))
df_grid['lag_28'] = g.transform(lambda x: x.shift(28))

df_grid['day_of_week'] = df_grid['date'].dt.dayofweek
df_grid['is_weekend']  = df_grid['day_of_week'].isin([5, 6]).astype(int)

df_grid['sku_code'] = df_grid['sku'].astype('category').cat.codes
df_grid['cat_code'] = df_grid['category'].astype('category').cat.codes

FORECAST_DATES = pd.date_range('2026-08-01', '2026-08-11', freq='D')
forecast_results = []

# ── TIER A ENGINE (Bestsellers: Price Elasticity + Bi-Weekly Lags + 1-Year Baseline) ──
print("Training Tier A Engine (Top 20 Bestsellers)...")
tier_a_features = [
    'price_elasticity_ratio', 'selling_price',
    'lag_1', 'lag_3', 'lag_7', 'lag_14',
    'roll_7_mean', 'roll_14_mean', 'roll_30_mean',
    'aug2025_11d_baseline', 'annual_365d_mean',
    'day_of_week', 'is_weekend'
]
df_clean_ta = df_grid[df_grid['sku'].isin(top_20_skus)].dropna(subset=tier_a_features).copy()
dtrain_ta   = lgb.Dataset(df_clean_ta[tier_a_features], label=df_clean_ta['total_sales'])

params_ta = {'objective': 'tweedie', 'tweedie_variance_power': 1.5, 'metric': 'rmse', 'learning_rate': 0.03, 'num_leaves': 63, 'max_depth': 7, 'verbose': -1, 'random_state': 42}
model_ta  = lgb.train(params_ta, dtrain_ta, num_boost_round=350)

for sku in top_20_skus:
    row  = df_grid[df_grid['sku']==sku].iloc[-1]
    hist = df_grid[df_grid['sku']==sku]['total_sales'].tolist()
    daily_preds = []
    buf = list(hist)
    
    for d in FORECAST_DATES:
        h7  = np.array(buf[-7:])  if len(buf)>=7  else np.array(buf)
        h14 = np.array(buf[-14:]) if len(buf)>=14 else np.array(buf)
        h30 = np.array(buf[-30:]) if len(buf)>=30 else np.array(buf)
        
        feat = pd.DataFrame([{
            'price_elasticity_ratio': 1.0, # August price normalized
            'selling_price': float(row['selling_price']),
            'lag_1': buf[-1], 'lag_3': buf[-3], 'lag_7': buf[-7], 'lag_14': buf[-14],
            'roll_7_mean': h7.mean(), 'roll_14_mean': h14.mean(), 'roll_30_mean': h30.mean(),
            'aug2025_11d_baseline': float(row['aug2025_11d_baseline']),
            'annual_365d_mean': float(row['annual_365d_mean']),
            'day_of_week': d.dayofweek, 'is_weekend': 1 if d.dayofweek in [5, 6] else 0
        }])[tier_a_features]
        
        p = float(np.maximum(0, model_ta.predict(feat)[0]))
        daily_preds.append(p)
        buf.append(p)
        
    tot_11d = np.sum(daily_preds)
    h30a    = np.array(hist[-30:]) if len(hist) >= 30 else np.array(hist or [0])
    sigma   = h30a.std() if len(h30a) > 1 else max(tot_11d * 0.15, 1.0)
    ci      = 1.96 * sigma * np.sqrt(11)
    
    forecast_results.append({
        'order sku': sku,
        'date': '1/08/2026 to 11/08/2026',
        'predicted unit': int(np.round(tot_11d)),
        'lower bound': max(0, int(np.round(tot_11d - ci))),
        'upper bound': int(np.round(tot_11d + ci)),
        'tier_engine': 'Tier A (Price Elasticity & Bi-Weekly Lag Engine)'
    })

# ── TIER B ENGINE (Mid-Movers: 365d Annual Baseline + 90-Day Trend + Multi-Lag Window) ──
print("Training Enhanced Tier B Engine (Next 100 Mid-Movers with 365d Annual Baseline)...")
tier_b_features_enhanced = [
    'sku_code', 'cat_code', 'selling_price',
    'annual_365d_mean', 'aug2025_11d_baseline',
    'roll_90_mean', 'roll_30_mean', 'roll_14_mean',
    'lag_7', 'lag_14', 'lag_28',
    'day_of_week', 'is_weekend'
]
df_clean_tb = df_grid[df_grid['sku'].isin(next_100_skus)].dropna(subset=tier_b_features_enhanced).copy()
dtrain_tb   = lgb.Dataset(df_clean_tb[tier_b_features_enhanced], label=df_clean_tb['total_sales'])

params_tb = {'objective': 'tweedie', 'tweedie_variance_power': 1.5, 'metric': 'rmse', 'learning_rate': 0.03, 'num_leaves': 63, 'max_depth': 7, 'verbose': -1, 'random_state': 42}
model_tb  = lgb.train(params_tb, dtrain_tb, num_boost_round=350)

for sku in next_100_skus:
    row  = df_grid[df_grid['sku']==sku].iloc[-1]
    hist = df_grid[df_grid['sku']==sku]['total_sales'].tolist()
    daily_preds = []
    buf = list(hist)
    
    for d in FORECAST_DATES:
        h90 = np.array(buf[-90:]) if len(buf)>=90 else np.array(buf)
        h30 = np.array(buf[-30:]) if len(buf)>=30 else np.array(buf)
        h14 = np.array(buf[-14:]) if len(buf)>=14 else np.array(buf)
        
        feat = pd.DataFrame([{
            'sku_code': int(row['sku_code']), 'cat_code': int(row['cat_code']),
            'selling_price': float(row['selling_price']),
            'annual_365d_mean': float(row['annual_365d_mean']),
            'aug2025_11d_baseline': float(row['aug2025_11d_baseline']),
            'roll_90_mean': h90.mean(), 'roll_30_mean': h30.mean(), 'roll_14_mean': h14.mean(),
            'lag_7': buf[-7] if len(buf)>=7 else 0, 'lag_14': buf[-14] if len(buf)>=14 else 0, 'lag_28': buf[-28] if len(buf)>=28 else 0,
            'day_of_week': d.dayofweek, 'is_weekend': 1 if d.dayofweek in [5, 6] else 0
        }])[tier_b_features_enhanced]
        
        p = float(np.maximum(0, model_tb.predict(feat)[0]))
        daily_preds.append(p)
        buf.append(p)
        
    tot_11d = np.sum(daily_preds)
    h30a    = np.array(hist[-30:]) if len(hist) >= 30 else np.array(hist or [0])
    sigma   = h30a.std() if len(h30a) > 1 else max(tot_11d * 0.18, 1.0)
    ci      = 1.96 * sigma * np.sqrt(11)
    
    forecast_results.append({
        'order sku': sku,
        'date': '1/08/2026 to 11/08/2026',
        'predicted unit': int(np.round(tot_11d)),
        'lower bound': max(0, int(np.round(tot_11d - ci))),
        'upper bound': int(np.round(tot_11d + ci)),
        'tier_engine': 'Tier B (365d Annual Baseline + 90d Trend Engine)'
    })

# ── TIER C ENGINE (Slow-Movers: Croston SBA Engine) ──
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

for sku in tail_400_skus:
    hist = df_grid[df_grid['sku'] == sku].sort_values('date')['total_sales'].values
    tot_11d = croston_sba(hist, forecast_len=11, alpha=0.1)
    tot_h_s = np.sum(hist)
    pred_val = int(np.round(tot_11d)) if tot_h_s >= 5 else 0
    
    forecast_results.append({
        'order sku': sku,
        'date': '1/08/2026 to 11/08/2026',
        'predicted unit': pred_val,
        'lower bound': max(0, int(np.round(pred_val * 0.7))),
        'upper bound': int(np.round(pred_val * 1.4)),
        'tier_engine': 'Tier C (Croston SBA Intermittent Engine)'
    })

df_out = pd.DataFrame(forecast_results).sort_values('predicted unit', ascending=False).reset_index(drop=True)

# Export Excel
with pd.ExcelWriter(OUT_EXCEL, engine='xlsxwriter') as writer:
    df_out.to_excel(writer, sheet_name='Sheet1', index=False)
    wb = writer.book
    ws = writer.sheets['Sheet1']
    hdr = wb.add_format({'bold': True, 'bg_color': '#1a73e8', 'font_color': '#ffffff', 'border': 1})
    for i, c in enumerate(df_out.columns):
        ws.write(0, i, c, hdr)
    ws.set_column('A:A', 26)
    ws.set_column('B:B', 30)
    ws.set_column('C:E', 18)
    ws.set_column('F:F', 45)
    ws.freeze_panes(1, 0)

print(f"\nReport Saved: {OUT_EXCEL}")
print(f"Total August 1-11 Forecast Volume: {df_out['predicted unit'].sum():,} units")
