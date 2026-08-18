"""
FIXED MODEL TRAINING — Adds SKU-level baseline features to break convergence
===========================================================================
Root cause of 58-unique-value problem:
  Slow movers share identical short-term features (roll_7=0, roll_30=0, lag_1=0).
  LightGBM sends them to the SAME leaf => same prediction.

  Example from diagnosis:
    RIM-EBP-HAZEL   : 13,711 units/12m, avg 38.7/day -> predicted only 33 (WRONG)
    RIM-SCD-EYE-002 : 9,023  units/12m, avg 36.4/day -> predicted only 1  (WRONG)
    Both had roll_7=0, roll_30=0 on Jul 31 (temporary stockout/gap)
    But their TRUE annual demand is 38+ units/day!

Fix: Add 3 SKU-level baseline features computed over the full training window:
  1. sku_annual_avg  = total_units / 365  (unique per SKU, captures true demand level)
  2. sku_sale_rate   = days_with_sales / 365  (how often this SKU sells)
  3. sku_avg_on_days = total_units / days_with_sales  (avg per active day)

These are computed ONCE before training and act as SKU-identity features.
The model can now distinguish RIM-EBP-HAZEL (38/day) from a true dead SKU (0.01/day).
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
OUT_EXCEL = os.path.join(BASE_DIR, 'Rimmel_588_SKUs_Forecast_1Aug_to_11Aug_2026.xlsx')

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load clean data
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 75)
print("STEP 1 — Loading clean data from SQLite...")
print("=" * 75)

conn = sqlite3.connect(DB_PATH)
df_raw  = pd.read_sql("SELECT * FROM training_window ORDER BY sku, date", conn)
df_smry = pd.read_sql("SELECT * FROM sku_summary", conn)
conn.close()

df_raw['date'] = pd.to_datetime(df_raw['date'])
all_skus  = df_raw['sku'].unique()
all_dates = pd.date_range('2025-08-01', '2026-07-31', freq='D')

print(f"  Rows       : {len(df_raw):,}")
print(f"  SKUs       : {len(all_skus)}")
print(f"  Total units: {df_raw['total_sales'].sum():,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Compute SKU-Level Baseline Features (computed once, over full window)
# These are the KEY FIX that breaks the slow-mover convergence.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("STEP 2 — Computing SKU-level baseline features (the convergence fix)...")
print("=" * 75)

sku_baselines = df_raw.groupby('sku').agg(
    total_units    = ('total_sales', 'sum'),
    days_with_sales= ('total_sales', lambda x: (x > 0).sum())
).reset_index()

sku_baselines['sku_annual_avg']  = sku_baselines['total_units'] / 365.0
sku_baselines['sku_sale_rate']   = sku_baselines['days_with_sales'] / 365.0
sku_baselines['sku_avg_on_days'] = (
    sku_baselines['total_units'] / sku_baselines['days_with_sales'].replace(0, 1)
)

print("Sample SKU baselines (top 10 vs problem SKUs):")
showcase = ['RIM-MSC-E3DL-003','RIM-EBP-HAZEL','RIM-SCD-EYE-002',
            'RIM-RADBRICK-001','RIM-MSC-VF-X10-001','RIM-BBCREAM-MEDIUM']
print(sku_baselines[sku_baselines['sku'].isin(showcase)][
    ['sku','total_units','sku_annual_avg','sku_sale_rate','sku_avg_on_days']
].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Build full daily calendar grid + features
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("STEP 3 — Building calendar grid and engineering features...")
print("=" * 75)

grid = pd.MultiIndex.from_product([all_skus, all_dates], names=['sku','date']).to_frame(index=False)
df_grid = grid.merge(df_raw[['sku','date','total_sales','selling_price','category']], 
                     on=['sku','date'], how='left')

df_grid['total_sales']   = df_grid['total_sales'].fillna(0)
df_grid['selling_price'] = df_grid.groupby('sku')['selling_price'].transform(lambda x: x.ffill().bfill())
df_grid['selling_price'] = df_grid['selling_price'].fillna(df_grid['selling_price'].median())

cat_fill = df_raw.groupby('sku')['category'].first().to_dict()
df_grid['category'] = df_grid['sku'].map(cat_fill).fillna('Uncategorised')

# Merge SKU baseline features — THIS IS THE FIX
df_grid = df_grid.merge(
    sku_baselines[['sku','sku_annual_avg','sku_sale_rate','sku_avg_on_days']],
    on='sku', how='left'
)

df_grid = df_grid.sort_values(['sku','date']).reset_index(drop=True)
g = df_grid.groupby('sku')['total_sales']

# Lag features
df_grid['lag_1']  = g.transform(lambda x: x.shift(1))
df_grid['lag_3']  = g.transform(lambda x: x.shift(3))
df_grid['lag_7']  = g.transform(lambda x: x.shift(7))
df_grid['lag_14'] = g.transform(lambda x: x.shift(14))
df_grid['lag_28'] = g.transform(lambda x: x.shift(28))

# Rolling windows
df_grid['roll_7_mean']  = g.transform(lambda x: x.shift(1).rolling(7,  min_periods=1).mean())
df_grid['roll_14_mean'] = g.transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())
df_grid['roll_30_mean'] = g.transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
df_grid['roll_7_std']   = g.transform(lambda x: x.shift(1).rolling(7,  min_periods=2).std().fillna(0))
df_grid['roll_30_std']  = g.transform(lambda x: x.shift(1).rolling(30, min_periods=2).std().fillna(0))
df_grid['roll_30_max']  = g.transform(lambda x: x.shift(1).rolling(30, min_periods=1).max())
df_grid['momentum']     = df_grid['roll_7_mean'] / (df_grid['roll_30_mean'] + 0.001)

# Price
pg = df_grid.groupby('sku')['selling_price']
df_grid['price_30avg'] = pg.transform(lambda x: x.rolling(30, min_periods=1).mean())
df_grid['price_ratio'] = df_grid['selling_price'] / (df_grid['price_30avg'] + 0.001)

# Calendar
df_grid['dow']        = df_grid['date'].dt.dayofweek
df_grid['month']      = df_grid['date'].dt.month
df_grid['dom']        = df_grid['date'].dt.day
df_grid['week']       = df_grid['date'].dt.isocalendar().week.astype(int)
df_grid['is_weekend'] = (df_grid['dow'] >= 5).astype(int)
df_grid['quarter']    = df_grid['date'].dt.quarter

# Encoded categoricals
df_grid['sku_code'] = df_grid['sku'].astype('category').cat.codes
df_grid['cat_code'] = df_grid['category'].astype('category').cat.codes

# Days active
first_sale = df_raw[df_raw['total_sales'] > 0].groupby('sku')['date'].min()
df_grid['first_sale']  = df_grid['sku'].map(first_sale)
df_grid['days_active'] = (df_grid['date'] - df_grid['first_sale']).dt.days.clip(lower=0).fillna(0)

FEATURES = [
    'sku_code', 'cat_code',
    'dow', 'month', 'dom', 'week', 'is_weekend', 'quarter',
    'selling_price', 'price_ratio',
    'lag_1', 'lag_3', 'lag_7', 'lag_14', 'lag_28',
    'roll_7_mean', 'roll_14_mean', 'roll_30_mean',
    'roll_7_std', 'roll_30_std', 'roll_30_max',
    'momentum', 'days_active',
    # THE FIX — 3 new SKU baseline features:
    'sku_annual_avg', 'sku_sale_rate', 'sku_avg_on_days'
]

df_model = df_grid.dropna(subset=FEATURES).copy()
print(f"  Feature rows ready: {len(df_model):,}")
print(f"  Features total    : {len(FEATURES)} (3 new baseline features added)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Train
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("STEP 4 — Training LightGBM...")
print("=" * 75)

VAL_START = pd.to_datetime('2026-06-01')
df_tr  = df_model[df_model['date'] <  VAL_START]
df_val = df_model[df_model['date'] >= VAL_START]

X_tr, y_tr   = df_tr[FEATURES],  df_tr['total_sales']
X_val, y_val = df_val[FEATURES], df_val['total_sales']

print(f"  Train: {len(X_tr):,} rows")
print(f"  Val  : {len(X_val):,} rows")

dtrain = lgb.Dataset(X_tr, label=y_tr)
dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

params = {
    'objective':              'tweedie',
    'tweedie_variance_power': 1.5,
    'metric':                 'rmse',
    'learning_rate':          0.03,
    'num_leaves':             63,
    'max_depth':              7,
    'min_data_in_leaf':       20,
    'feature_fraction':       0.8,
    'bagging_fraction':       0.8,
    'bagging_freq':           5,
    'lambda_l1':              0.1,
    'lambda_l2':              0.1,
    'verbose':                -1,
}

model = lgb.train(
    params, dtrain,
    num_boost_round=600,
    valid_sets=[dval],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
)

val_pred = np.maximum(0, model.predict(X_val))
val_rmse = np.sqrt(np.mean((val_pred - y_val.values)**2))

mask_nz = y_val.values > 0
val_mape = np.mean(np.abs(val_pred[mask_nz] - y_val.values[mask_nz]) / y_val.values[mask_nz]) * 100

vol_weight = y_val.values[mask_nz] / y_val.values[mask_nz].sum()
vw_mape    = np.sum(vol_weight * np.abs(val_pred[mask_nz] - y_val.values[mask_nz]) / y_val.values[mask_nz]) * 100

print(f"\n  Best iteration             : {model.best_iteration}")
print(f"  Validation RMSE            : {val_rmse:.2f}")
print(f"  MAPE  (non-zero days)      : {val_mape:.1f}%  -> accuracy {100-val_mape:.1f}%")
print(f"  Volume-Weighted MAPE       : {vw_mape:.1f}%  -> accuracy {100-vw_mape:.1f}%")

fi = pd.DataFrame({'feature': FEATURES, 'gain': model.feature_importance('gain')})
fi = fi.sort_values('gain', ascending=False)
print(f"\n  Top 12 Features by Gain:")
print(fi.head(12).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Forecast 1-11 Aug 2026
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("STEP 5 — Forecasting 1 Aug to 11 Aug 2026...")
print("=" * 75)

FORECAST_DATES = pd.date_range('2026-08-01', '2026-08-11', freq='D')

last_state = df_grid[df_grid['date'] == pd.to_datetime('2026-07-31')].set_index('sku')

sku_baseline_map = sku_baselines.set_index('sku').to_dict('index')

results = []
for sku in all_skus:
    hist = df_grid[df_grid['sku'] == sku].sort_values('date')['total_sales'].tolist()

    try:
        row = last_state.loc[sku]
    except KeyError:
        row = df_grid[df_grid['sku'] == sku].sort_values('date').iloc[-1]

    sku_code  = int(row['sku_code'])
    cat_code  = int(row['cat_code'])
    price     = float(row['selling_price'])
    price_30  = float(row['price_30avg']) if pd.notna(row.get('price_30avg')) else price
    days_act  = int(row['days_active'])   if pd.notna(row.get('days_active'))  else 0
    bl        = sku_baseline_map.get(sku, {})
    ann_avg   = float(bl.get('sku_annual_avg',  0))
    sale_rate = float(bl.get('sku_sale_rate',   0))
    avg_on_d  = float(bl.get('sku_avg_on_days', 0))

    daily_preds = []
    buf = list(hist)

    for i, d in enumerate(FORECAST_DATES):
        def lag(n): return buf[-n] if len(buf) >= n else 0.0

        h7  = np.array(buf[-7:])  if len(buf) >= 7  else np.array(buf or [0])
        h14 = np.array(buf[-14:]) if len(buf) >= 14 else np.array(buf or [0])
        h30 = np.array(buf[-30:]) if len(buf) >= 30 else np.array(buf or [0])

        r7m  = h7.mean();  r14m = h14.mean(); r30m = h30.mean()
        r7s  = h7.std() if len(h7)>1 else 0.0
        r30s = h30.std() if len(h30)>1 else 0.0
        r30x = h30.max()
        mom  = r7m / (r30m + 0.001)

        feat = pd.DataFrame([{
            'sku_code': sku_code, 'cat_code': cat_code,
            'dow': d.dayofweek, 'month': d.month, 'dom': d.day,
            'week': int(d.isocalendar().week), 'is_weekend': int(d.dayofweek >= 5),
            'quarter': d.quarter,
            'selling_price': price, 'price_ratio': price / (price_30 + 0.001),
            'lag_1': lag(1), 'lag_3': lag(3), 'lag_7': lag(7),
            'lag_14': lag(14), 'lag_28': lag(28),
            'roll_7_mean': r7m, 'roll_14_mean': r14m, 'roll_30_mean': r30m,
            'roll_7_std': r7s, 'roll_30_std': r30s, 'roll_30_max': r30x,
            'momentum': mom, 'days_active': days_act + i + 1,
            'sku_annual_avg':  ann_avg,
            'sku_sale_rate':   sale_rate,
            'sku_avg_on_days': avg_on_d,
        }])

        p = float(np.maximum(0, model.predict(feat[FEATURES])[0]))
        daily_preds.append(p)
        buf.append(p)

    total_11d = sum(daily_preds)
    h30a  = np.array(hist[-30:]) if len(hist) >= 30 else np.array(hist or [0])
    sigma = h30a.std() if len(h30a) > 1 else max(total_11d * 0.20, 1.0)
    ci    = 1.96 * sigma * np.sqrt(11)

    results.append({
        'order sku':      sku,
        'date':           '1/08/2026 to 11/08/2026',
        'predicted unit': int(round(total_11d)),
        'lower bound':    max(0, int(round(total_11d - ci))),
        'upper bound':    int(round(total_11d + ci)),
    })

df_out = pd.DataFrame(results).sort_values('predicted unit', ascending=False).reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Sanity checks
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("STEP 6 — Sanity checks...")
print("=" * 75)
print(f"  Total SKUs         : {len(df_out)}")
print(f"  Unique pred values : {df_out['predicted unit'].nunique()}  (was 58 before, should be much higher)")
print(f"  SKUs with pred > 0 : {(df_out['predicted unit'] > 0).sum()}")
print(f"  SKUs with pred == 0: {(df_out['predicted unit'] == 0).sum()}")
print(f"  Total forecast     : {df_out['predicted unit'].sum():,} units")
print(f"\n  Key validation (actual client-confirmed values):")
print(f"    RIM-MSC-E3DL-003 actual Aug 1-11: ~1,046 units")
p_best = df_out[df_out['order sku'] == 'RIM-MSC-E3DL-003']['predicted unit'].values
print(f"    RIM-MSC-E3DL-003 predicted     : {p_best[0] if len(p_best) else 'N/A'} units")
p_hazel = df_out[df_out['order sku'] == 'RIM-EBP-HAZEL']['predicted unit'].values
print(f"    RIM-EBP-HAZEL predicted (prev=33): {p_hazel[0] if len(p_hazel) else 'N/A'} units")
p_eye = df_out[df_out['order sku'] == 'RIM-SCD-EYE-002']['predicted unit'].values
print(f"    RIM-SCD-EYE-002 predicted (prev=1): {p_eye[0] if len(p_eye) else 'N/A'} units")

print(f"\n  Top 20:")
print(df_out.head(20).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Export
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("STEP 7 — Exporting Excel + SQLite...")
print("=" * 75)

def save_excel(path, df):
    with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Forecast', index=False)
        wb = writer.book
        ws = writer.sheets['Forecast']
        hdr = wb.add_format({'bold': True, 'bg_color': '#1a73e8', 'font_color': '#ffffff', 'border': 1})
        for i, c in enumerate(df.columns):
            ws.write(0, i, c, hdr)
        ws.set_column('A:A', 26)
        ws.set_column('B:B', 30)
        ws.set_column('C:E', 18)
        ws.freeze_panes(1, 0)

try:
    save_excel(OUT_EXCEL, df_out)
    print(f"  Excel saved: {OUT_EXCEL}")
except PermissionError:
    alt = OUT_EXCEL.replace('.xlsx', '_v2.xlsx')
    save_excel(alt, df_out)
    print(f"  Excel saved (alt): {alt}")

conn = sqlite3.connect(DB_PATH)
df_out.to_sql('forecast_aug_2026', conn, if_exists='replace', index=False)
conn.close()
print(f"  SQLite saved: forecast_aug_2026 table in {DB_PATH}")
print("\nDONE.")
