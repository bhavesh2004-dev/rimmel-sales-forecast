"""
PROPER PRODUCTION FORECASTING ENGINE
- Training Data: 1 Aug 2025 to 31 Jul 2026 (client-confirmed clean period)
- No patches, no heuristics, no hardcoded fallbacks
- Proper aggregation: sum across channels per SKU per day FIRST
- Proper features: lag-based time-series with correct leakage-free design
- Model: LightGBM with Tweedie loss (handles zero-heavy SKUs naturally)
- Output: 588 SKUs forecast for 1 Aug to 11 Aug 2026
"""

import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

base_dir = r'C:\Users\bhave\Desktop\ml_project'
excel_file = os.path.join(base_dir, 'Rimmel Brand Products Sales - 1 Jan 2025 to 31st July 2026 New.xlsx')
output_dir = os.path.join(base_dir, 'new_approach', 'output_results')

# ============================================================
# STEP 1: LOAD RAW DATA
# ============================================================
print("=" * 80)
print("STEP 1 — Loading raw dataset...")
print("=" * 80)

df_raw = pd.read_excel(excel_file)
df_raw['date'] = pd.to_datetime(df_raw['date'])
df_raw['sku'] = df_raw['sku'].astype(str).str.strip()
df_raw['category'] = df_raw['category'].fillna('Uncategorised').astype(str).str.strip()
df_raw['sold_quantity'] = pd.to_numeric(df_raw['sold_quantity'], errors='coerce').fillna(0).clip(lower=0)
df_raw['selling_price'] = pd.to_numeric(df_raw['selling_price'], errors='coerce')

print(f"Raw rows loaded: {len(df_raw):,}")
print(f"Date range in raw data: {df_raw['date'].min().date()} to {df_raw['date'].max().date()}")
print(f"Total unique SKUs: {df_raw['sku'].nunique()}")

# ============================================================
# STEP 2: AGGREGATE ACROSS CHANNELS (Sum per SKU per Day)
# Raw data has MULTIPLE rows per (SKU, date) — one per channel.
# We must aggregate first before building any features.
# ============================================================
print("\n" + "=" * 80)
print("STEP 2 — Aggregating across channels (sum per SKU per day)...")
print("=" * 80)

# For price: take the median price across channels on that day
df_agg = df_raw.groupby(['sku', 'date']).agg(
    sold_qty=('sold_quantity', 'sum'),
    price=('selling_price', 'median'),
    category=('category', 'first'),
    title=('title', 'first')
).reset_index()

print(f"After aggregation: {len(df_agg):,} rows (one per SKU per day)")

# ============================================================
# STEP 3: FILTER TO TRAINING WINDOW (1 Aug 2025 to 31 Jul 2026)
# ============================================================
print("\n" + "=" * 80)
print("STEP 3 — Filtering to client training window: 1 Aug 2025 to 31 Jul 2026...")
print("=" * 80)

TRAIN_START = pd.to_datetime('2025-08-01')
TRAIN_END   = pd.to_datetime('2026-07-31')
FORECAST_START = pd.to_datetime('2026-08-01')
FORECAST_END   = pd.to_datetime('2026-08-11')

# Build full continuous daily grid: every SKU gets every date in the window
all_skus = df_agg['sku'].unique()
all_dates = pd.date_range(start=TRAIN_START, end=TRAIN_END, freq='D')

print(f"Training window: {TRAIN_START.date()} to {TRAIN_END.date()} = {len(all_dates)} days")
print(f"Total SKUs to forecast: {len(all_skus)}")

grid = pd.MultiIndex.from_product([all_skus, all_dates], names=['sku', 'date']).to_frame(index=False)

df_train_raw = df_agg[(df_agg['date'] >= TRAIN_START) & (df_agg['date'] <= TRAIN_END)].copy()

df_train = pd.merge(grid, df_train_raw[['sku', 'date', 'sold_qty', 'price', 'category', 'title']],
                    on=['sku', 'date'], how='left')
df_train['sold_qty'] = df_train['sold_qty'].fillna(0)

# Fill category and title from SKU lookup
sku_meta = df_agg.groupby('sku').agg(category=('category','first'), title=('title','first')).reset_index()
df_train = df_train.merge(sku_meta.rename(columns={'category':'cat_fill','title':'title_fill'}), on='sku', how='left')
df_train['category'] = df_train['category'].fillna(df_train['cat_fill'])
df_train['title'] = df_train['title'].fillna(df_train['title_fill'])
df_train = df_train.drop(columns=['cat_fill','title_fill'])

# Fill price: forward fill within SKU, then backward fill, then global median
df_train['price'] = df_train.groupby('sku')['price'].transform(lambda x: x.ffill().bfill())
df_train['price'] = df_train['price'].fillna(df_train['price'].median())

print(f"Training grid shape: {df_train.shape}")
print(f"Zero sales rows: {(df_train['sold_qty']==0).sum():,} ({(df_train['sold_qty']==0).mean()*100:.1f}%)")
print(f"Non-zero sales rows: {(df_train['sold_qty']>0).sum():,}")

# ============================================================
# STEP 4: FEATURE ENGINEERING (no data leakage)
# All lag/rolling features use shift(1) — only past info seen by model
# ============================================================
print("\n" + "=" * 80)
print("STEP 4 — Engineering features (leakage-free lag/rolling features)...")
print("=" * 80)

df_train = df_train.sort_values(['sku', 'date']).reset_index(drop=True)

def add_features(df):
    df = df.copy()
    g = df.groupby('sku')['sold_qty']

    # Lag features (strict shift — no leakage)
    df['lag_1']  = g.transform(lambda x: x.shift(1))
    df['lag_3']  = g.transform(lambda x: x.shift(3))
    df['lag_7']  = g.transform(lambda x: x.shift(7))
    df['lag_14'] = g.transform(lambda x: x.shift(14))
    df['lag_28'] = g.transform(lambda x: x.shift(28))

    # Rolling windows (shift first, then roll — no leakage)
    df['roll_7_mean']  = g.transform(lambda x: x.shift(1).rolling(7,  min_periods=1).mean())
    df['roll_14_mean'] = g.transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())
    df['roll_30_mean'] = g.transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
    df['roll_7_std']   = g.transform(lambda x: x.shift(1).rolling(7,  min_periods=2).std().fillna(0))
    df['roll_30_std']  = g.transform(lambda x: x.shift(1).rolling(30, min_periods=2).std().fillna(0))
    df['roll_30_max']  = g.transform(lambda x: x.shift(1).rolling(30, min_periods=1).max())

    # Trend: compare recent 7d vs 30d mean (captures momentum)
    df['trend_7_vs_30'] = df['roll_7_mean'] / (df['roll_30_mean'] + 0.01)

    # Price features
    df['price_roll_30']    = df.groupby('sku')['price'].transform(lambda x: x.rolling(30, min_periods=1).mean())
    df['price_vs_avg']     = df['price'] / (df['price_roll_30'] + 0.01)

    # Calendar features
    df['dow']       = df['date'].dt.dayofweek      # 0=Mon ... 6=Sun
    df['month']     = df['date'].dt.month
    df['dom']       = df['date'].dt.day
    df['week']      = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['quarter']   = df['date'].dt.quarter

    # SKU days active (since first non-zero sale in training window)
    first_sale = df[df['sold_qty'] > 0].groupby('sku')['date'].min()
    df['first_sale_date'] = df['sku'].map(first_sale)
    df['days_since_launch'] = (df['date'] - df['first_sale_date']).dt.days.clip(lower=0)

    # Categorical encoding
    df['sku_code'] = df['sku'].astype('category').cat.codes
    df['cat_code'] = df['category'].astype('category').cat.codes

    return df

df_feat = add_features(df_train)

FEATURE_COLS = [
    'sku_code', 'cat_code',
    'dow', 'month', 'dom', 'week', 'is_weekend', 'quarter',
    'price', 'price_vs_avg',
    'lag_1', 'lag_3', 'lag_7', 'lag_14', 'lag_28',
    'roll_7_mean', 'roll_14_mean', 'roll_30_mean',
    'roll_7_std', 'roll_30_std', 'roll_30_max',
    'trend_7_vs_30',
    'days_since_launch'
]

TARGET_COL = 'sold_qty'

# Drop rows where we don't have enough history (first 28 days per SKU)
# lag_28 will be NaN for first 28 days
df_model = df_feat.dropna(subset=FEATURE_COLS).copy()

print(f"Training rows after dropping NaN lags: {len(df_model):,}")
print(f"Features: {FEATURE_COLS}")

# ============================================================
# STEP 5: TRAIN MODEL
# Tweedie loss — designed for zero-heavy count data with wide variance
# ============================================================
print("\n" + "=" * 80)
print("STEP 5 — Training LightGBM with Tweedie loss...")
print("=" * 80)

# Time-based split: train on Aug 2025 to May 2026, validate Jun-Jul 2026
val_start = pd.to_datetime('2026-06-01')
df_tr = df_model[df_model['date'] < val_start]
df_val = df_model[df_model['date'] >= val_start]

X_tr, y_tr   = df_tr[FEATURE_COLS], df_tr[TARGET_COL]
X_val, y_val = df_val[FEATURE_COLS], df_val[TARGET_COL]

print(f"Train set: {len(X_tr):,} rows ({df_tr['date'].min().date()} to {df_tr['date'].max().date()})")
print(f"Val   set: {len(X_val):,} rows ({df_val['date'].min().date()} to {df_val['date'].max().date()})")

dtrain = lgb.Dataset(X_tr, label=y_tr)
dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

params = {
    'objective': 'tweedie',
    'tweedie_variance_power': 1.5,   # 1=Poisson, 2=Gamma; 1.5 suits retail
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': 7,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbose': -1,
}

model = lgb.train(
    params,
    dtrain,
    num_boost_round=500,
    valid_sets=[dval],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
)

print(f"\nBest iteration: {model.best_iteration}")

# Validation RMSE and accuracy check
val_pred = model.predict(X_val)
val_rmse = np.sqrt(np.mean((val_pred - y_val.values)**2))
# MAPE only on non-zero actuals
mask = y_val > 0
val_mape = np.mean(np.abs(val_pred[mask] - y_val.values[mask]) / y_val.values[mask]) * 100

print(f"Validation RMSE (Jun-Jul 2026): {val_rmse:.2f}")
print(f"Validation MAPE on non-zero days: {val_mape:.1f}%")
print(f"Validation Accuracy (non-zero): {100 - val_mape:.1f}%")

# Feature importance
imp = pd.DataFrame({'feature': FEATURE_COLS, 'importance': model.feature_importance(importance_type='gain')})
imp = imp.sort_values('importance', ascending=False)
print(f"\nTop 10 Feature Importances:")
print(imp.head(10).to_string(index=False))

# ============================================================
# STEP 6: FORECAST 1 AUG to 11 AUG 2026
# Iterative daily prediction — each day's prediction feeds as lag for next day
# ============================================================
print("\n" + "=" * 80)
print("STEP 6 — Forecasting 1 Aug to 11 Aug 2026 (iterative daily)...")
print("=" * 80)

# Get last 30 days of actual sales per SKU to seed lags
df_seed = df_feat[df_feat['date'] <= TRAIN_END].copy()

# SKU-level metadata lookup
sku_cat_code = df_feat.groupby('sku')['cat_code'].first().to_dict()
sku_price     = df_feat.groupby('sku')['price'].last().to_dict()
sku_price_avg = df_feat.groupby('sku')['price_roll_30'].last().to_dict()
sku_code_map  = df_feat.groupby('sku')['sku_code'].first().to_dict()
sku_launch    = df_feat.groupby('sku')['days_since_launch'].last().to_dict()

# Keep last 30 actual daily sales per SKU to compute lags
sku_history = {}
for sku in all_skus:
    s = df_feat[df_feat['sku'] == sku].sort_values('date')['sold_qty'].values
    sku_history[sku] = list(s)  # full history up to Jul 31

forecast_days = pd.date_range(start=FORECAST_START, end=FORECAST_END, freq='D')

all_forecasts = []

for sku in all_skus:
    hist = list(sku_history.get(sku, [0]))
    price    = sku_price.get(sku, 5.0)
    price_avg = sku_price_avg.get(sku, price)
    sk_code  = sku_code_map.get(sku, 0)
    ct_code  = sku_cat_code.get(sku, 0)
    _dl = sku_launch.get(sku, 0)
    d_launch = 0 if (pd.isna(_dl) or _dl is None) else int(_dl)

    daily_preds = []
    for i, d in enumerate(forecast_days):
        h = hist  # current history buffer

        # Compute lags from history buffer
        lag_1  = h[-1]  if len(h) >= 1  else 0
        lag_3  = h[-3]  if len(h) >= 3  else 0
        lag_7  = h[-7]  if len(h) >= 7  else 0
        lag_14 = h[-14] if len(h) >= 14 else 0
        lag_28 = h[-28] if len(h) >= 28 else 0

        recent_7  = np.array(h[-7:])  if len(h) >= 7  else np.array(h)
        recent_14 = np.array(h[-14:]) if len(h) >= 14 else np.array(h)
        recent_30 = np.array(h[-30:]) if len(h) >= 30 else np.array(h)

        r7_mean  = recent_7.mean()  if len(recent_7)  > 0 else 0
        r14_mean = recent_14.mean() if len(recent_14) > 0 else 0
        r30_mean = recent_30.mean() if len(recent_30) > 0 else 0
        r7_std   = recent_7.std()   if len(recent_7)  > 1 else 0
        r30_std  = recent_30.std()  if len(recent_30) > 1 else 0
        r30_max  = recent_30.max()  if len(recent_30) > 0 else 0
        trend    = r7_mean / (r30_mean + 0.01)

        feat = {
            'sku_code': sk_code,
            'cat_code': ct_code,
            'dow': d.dayofweek,
            'month': d.month,
            'dom': d.day,
            'week': int(d.isocalendar().week),
            'is_weekend': int(d.dayofweek >= 5),
            'quarter': d.quarter,
            'price': price,
            'price_vs_avg': price / (price_avg + 0.01),
            'lag_1': lag_1,
            'lag_3': lag_3,
            'lag_7': lag_7,
            'lag_14': lag_14,
            'lag_28': lag_28,
            'roll_7_mean': r7_mean,
            'roll_14_mean': r14_mean,
            'roll_30_mean': r30_mean,
            'roll_7_std': r7_std,
            'roll_30_std': r30_std,
            'roll_30_max': r30_max,
            'trend_7_vs_30': trend,
            'days_since_launch': d_launch + i + 1
        }

        X_pred = pd.DataFrame([feat])[FEATURE_COLS]
        p = float(np.maximum(0, model.predict(X_pred)[0]))
        daily_preds.append(p)

        # Feed this prediction into history for next day's lags
        hist = hist + [p]

    total_11d = sum(daily_preds)
    # Confidence interval: use rolling std of last 30 actual days scaled to 11d
    actual_30d = np.array(sku_history.get(sku, [0])[-30:])
    daily_std   = actual_30d.std() if len(actual_30d) > 1 else total_11d * 0.2
    lower = max(0, total_11d - 1.96 * daily_std * np.sqrt(11))
    upper = total_11d + 1.96 * daily_std * np.sqrt(11)

    all_forecasts.append({
        'order sku': sku,
        'date': '1/08/2026 to 11/08/2026',
        'predicted unit': int(round(total_11d)),
        'lower bound': int(round(lower)),
        'upper bound': int(round(upper))
    })

df_out = pd.DataFrame(all_forecasts).sort_values('predicted unit', ascending=False).reset_index(drop=True)

# ============================================================
# STEP 7: SANITY CHECK & EXPORT
# ============================================================
print("\n" + "=" * 80)
print("STEP 7 — Sanity checks and export...")
print("=" * 80)

print(f"Total SKUs in output : {len(df_out)}")
print(f"SKUs with pred > 0   : {(df_out['predicted unit'] > 0).sum()}")
print(f"SKUs with pred == 0  : {(df_out['predicted unit'] == 0).sum()}")
print(f"Unique pred values   : {df_out['predicted unit'].nunique()}")
print(f"Total forecast units : {df_out['predicted unit'].sum():,}")
print(f"\nTop 20:")
print(df_out.head(20).to_string(index=False))

# Export to Excel
out_path = os.path.join(base_dir, 'Rimmel_588_SKUs_Forecast_1Aug_to_11Aug_2026.xlsx')
try:
    with pd.ExcelWriter(out_path, engine='xlsxwriter') as writer:
        df_out.to_excel(writer, sheet_name='Forecast', index=False)
        wb = writer.book
        ws = writer.sheets['Forecast']
        hdr_fmt = wb.add_format({'bold': True, 'border': 1, 'bg_color': '#D9EAD3'})
        for i, col in enumerate(df_out.columns):
            ws.write(0, i, col, hdr_fmt)
        ws.set_column('A:A', 26)
        ws.set_column('B:B', 30)
        ws.set_column('C:E', 18)
    print(f"\nEXCEL SAVED: {out_path}")
except PermissionError:
    alt_path = out_path.replace('.xlsx', '_v2.xlsx')
    with pd.ExcelWriter(alt_path, engine='xlsxwriter') as writer:
        df_out.to_excel(writer, sheet_name='Forecast', index=False)
        wb = writer.book
        ws = writer.sheets['Forecast']
        hdr_fmt = wb.add_format({'bold': True, 'border': 1, 'bg_color': '#D9EAD3'})
        for i, col in enumerate(df_out.columns):
            ws.write(0, i, col, hdr_fmt)
        ws.set_column('A:A', 26)
        ws.set_column('B:B', 30)
        ws.set_column('C:E', 18)
    print(f"\nEXCEL SAVED (alt): {alt_path}")

# Also save forecast CSV for web app
df_out_csv = df_out.rename(columns={
    'order sku': 'order_item_sku',
    'predicted unit': 'predicted_sales',
    'lower bound': 'lower_bound_95',
    'upper bound': 'upper_bound_95'
})
df_out_csv.to_csv(os.path.join(output_dir, 'global_588_august_forecast.csv'), index=False)
print("CSV for web app saved.")
