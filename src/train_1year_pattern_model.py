"""
Phase 2 & 3: Production 1-Year Pattern Machine Learning Engine
================================================================
Reads from SQLite table 'training_window' (1 Aug 2025 - 31 Jul 2026)
Feature Matrix:
1. August 2025 Same-Period Baseline (11-day actuals from last year)
2. 365-Day Annual Daily Average & Active Rate
3. 90-Day Quarterly Demand Trend (May-Jul 2026)
4. 30-Day Monthly Demand Trend (July 2026)
5. Short Lags (lag_1, lag_7, lag_14, lag_28)
6. Price Elasticity & Category Seasonality Index

Model: LightGBM Regressor (Tweedie Loss)
"""
import os
import sqlite3
import pandas as pd
import numpy as np
import lightgbm as lgb

BASE_DIR  = r'C:\Users\bhave\Desktop\ml_project'
DB_PATH   = os.path.join(BASE_DIR, 'data', 'rimmel_clean.db')
OUT_EXCEL = os.path.join(BASE_DIR, 'Rimmel_588_SKUs_Forecast_1Aug_to_11Aug_2026.xlsx')

print("="*85)
print("PHASE 2 & 3: 1-YEAR PATTERN MACHINE LEARNING ENGINE")
print("="*85)

# 1. Connect to SQLite database
conn = sqlite3.connect(DB_PATH)
df_train_raw = pd.read_sql("SELECT * FROM training_window ORDER BY sku, date", conn)
df_full_raw  = pd.read_sql("SELECT * FROM full_history ORDER BY sku, date", conn)
conn.close()

df_train_raw['date'] = pd.to_datetime(df_train_raw['date'])
df_full_raw['date']  = pd.to_datetime(df_full_raw['date'])

all_skus = df_train_raw['sku'].unique()
all_dates = pd.date_range('2025-08-01', '2026-07-31', freq='D')

print(f"Data Loaded from SQLite DB ({DB_PATH})")
print(f"Training Period : 1 Aug 2025 to 31 Jul 2026 ({len(all_dates)} days)")
print(f"Active SKUs     : {len(all_skus)}")

# 2. Extract 1-Year Pattern Anchors per SKU
# Anchor 1: August 2025 Same-Period Actual Sales (Aug 1 to Aug 11, 2025)
aug_2025_df = df_full_raw[(df_full_raw['date'] >= '2025-08-01') & (df_full_raw['date'] <= '2025-08-11')]
aug_2025_map = aug_2025_df.groupby('sku')['total_sales'].sum().to_dict()

# Anchor 2: 365-Day Annual Daily Average
annual_365d_map = (df_train_raw.groupby('sku')['total_sales'].sum() / 365.0).to_dict()
active_days_map = (df_train_raw[df_train_raw['total_sales'] > 0].groupby('sku')['date'].count() / 365.0).to_dict()

# Anchor 3: 90-Day Quarterly Average (May - Jul 2026)
qtr_90d_df = df_train_raw[df_train_raw['date'] >= '2026-05-01']
qtr_90d_map = (qtr_90d_df.groupby('sku')['total_sales'].sum() / 92.0).to_dict()

# Anchor 4: Category August Seasonality Index
df_train_raw['month'] = df_train_raw['date'].dt.month
cat_aug_sales = df_full_raw[df_full_raw['date'].dt.month == 8].groupby('category')['total_sales'].mean().to_dict()
cat_ann_sales = df_full_raw.groupby('category')['total_sales'].mean().to_dict()

cat_aug_index = {}
for cat in df_train_raw['category'].unique():
    aug_m = cat_aug_sales.get(cat, 1.0)
    ann_m = cat_ann_sales.get(cat, 1.0)
    cat_aug_index[cat] = aug_m / max(0.01, ann_m)

# 3. Build Daily Feature Matrix across 365-Day Calendar Grid
grid = pd.MultiIndex.from_product([all_skus, all_dates], names=['sku', 'date']).to_frame(index=False)
df_grid = grid.merge(df_train_raw[['sku', 'date', 'total_sales', 'selling_price', 'category']], on=['sku', 'date'], how='left')

df_grid['total_sales']   = df_grid['total_sales'].fillna(0)
df_grid['selling_price'] = df_grid.groupby('sku')['selling_price'].transform(lambda x: x.ffill().bfill())
df_grid['selling_price'] = df_grid['selling_price'].fillna(df_grid['selling_price'].median())

cat_map_dict = df_train_raw.groupby('sku')['category'].first().to_dict()
df_grid['category'] = df_grid['sku'].map(cat_map_dict).fillna('Cosmetics')

# Attach 1-Year Pattern Features
df_grid['aug2025_11d_baseline'] = df_grid['sku'].map(aug_2025_map).fillna(0.0)
df_grid['annual_365d_mean']     = df_grid['sku'].map(annual_365d_map).fillna(0.01)
df_grid['annual_active_rate']   = df_grid['sku'].map(active_days_map).fillna(0.1)
df_grid['qtr_90d_mean']         = df_grid['sku'].map(qtr_90d_map).fillna(0.01)
df_grid['cat_aug_index']        = df_grid['category'].map(cat_aug_index).fillna(1.0)

df_grid = df_grid.sort_values(['sku', 'date']).reset_index(drop=True)
g = df_grid.groupby('sku')['total_sales']

# Rolling & Lag Features
df_grid['lag_1']  = g.transform(lambda x: x.shift(1))
df_grid['lag_3']  = g.transform(lambda x: x.shift(3))
df_grid['lag_7']  = g.transform(lambda x: x.shift(7))
df_grid['lag_14'] = g.transform(lambda x: x.shift(14))
df_grid['lag_28'] = g.transform(lambda x: x.shift(28))

df_grid['roll_7_mean']  = g.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
df_grid['roll_14_mean'] = g.transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())
df_grid['roll_30_mean'] = g.transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
df_grid['roll_30_std']  = g.transform(lambda x: x.shift(1).rolling(30, min_periods=2).std().fillna(0))

# Momentum & Price Ratios
df_grid['momentum_7_vs_90'] = df_grid['roll_7_mean'] / (df_grid['qtr_90d_mean'] + 0.001)
df_grid['price_vs_annual']  = df_grid['selling_price'] / (df_grid.groupby('sku')['selling_price'].transform('mean') + 0.001)

# Calendar Features
df_grid['day_of_week'] = df_grid['date'].dt.dayofweek
df_grid['month']       = df_grid['date'].dt.month
df_grid['is_weekend']  = df_grid['day_of_week'].isin([5, 6]).astype(int)

df_grid['sku_code'] = df_grid['sku'].astype('category').cat.codes
df_grid['cat_code'] = df_grid['category'].astype('category').cat.codes

FEATURES = [
    'sku_code', 'cat_code', 'day_of_week', 'month', 'is_weekend',
    'selling_price', 'price_vs_annual',
    'aug2025_11d_baseline', 'annual_365d_mean', 'annual_active_rate', 'qtr_90d_mean', 'cat_aug_index',
    'lag_1', 'lag_3', 'lag_7', 'lag_14', 'lag_28',
    'roll_7_mean', 'roll_14_mean', 'roll_30_mean', 'roll_30_std',
    'momentum_7_vs_90'
]

df_model = df_grid.dropna(subset=FEATURES).copy()
print(f"\nEngineered {len(FEATURES)} Features across {len(df_model):,} Clean Training Rows")

# 4. Train LightGBM Model with Tweedie Loss
VAL_START = pd.to_datetime('2026-06-01')
df_tr  = df_model[df_model['date'] < VAL_START]
df_val = df_model[df_model['date'] >= VAL_START]

X_tr, y_tr   = df_tr[FEATURES], df_tr['total_sales']
X_val, y_val = df_val[FEATURES], df_val['total_sales']

dtrain = lgb.Dataset(X_tr, label=y_tr)
dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

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

print("\nTraining LightGBM 1-Year Pattern Engine...")
model = lgb.train(
    params, dtrain,
    num_boost_round=600,
    valid_sets=[dval],
    callbacks=[lgb.early_stopping(50, verbose=False)]
)

val_pred = np.maximum(0, model.predict(X_val))
val_rmse = np.sqrt(np.mean((val_pred - y_val.values)**2))
mask_nz  = y_val.values > 0
vol_w    = y_val.values[mask_nz] / y_val.values[mask_nz].sum()
vw_mape  = np.sum(vol_w * np.abs(val_pred[mask_nz] - y_val.values[mask_nz]) / y_val.values[mask_nz]) * 100

print(f"Model Training Complete!")
print(f"Validation RMSE                      : {val_rmse:.2f}")
print(f"Volume-Weighted Catalog Accuracy    : {100 - vw_mape:.1f}%")

# 5. Forecast August 1 to August 11, 2026 (11 Days)
print("\nGenerating August 1 to August 11, 2026 Forecasts...")
FORECAST_DATES = pd.date_range('2026-08-01', '2026-08-11', freq='D')
last_state = df_grid[df_grid['date'] == pd.to_datetime('2026-07-31')].set_index('sku')

forecast_results = []

for sku in all_skus:
    hist = df_grid[df_grid['sku'] == sku].sort_values('date')['total_sales'].tolist()
    row  = last_state.loc[sku]
    
    sku_c     = int(row['sku_code'])
    cat_c     = int(row['cat_code'])
    price     = float(row['selling_price'])
    aug25_b   = float(row['aug2025_11d_baseline'])
    ann_mean  = float(row['annual_365d_mean'])
    act_rate  = float(row['annual_active_rate'])
    q90_mean  = float(row['qtr_90d_mean'])
    cat_idx   = float(row['cat_aug_index'])
    
    daily_preds = []
    buf = list(hist)
    
    for i, d in enumerate(FORECAST_DATES):
        def lag(n): return buf[-n] if len(buf) >= n else 0.0
        
        h7  = np.array(buf[-7:])  if len(buf) >= 7  else np.array(buf or [0])
        h14 = np.array(buf[-14:]) if len(buf) >= 14 else np.array(buf or [0])
        h30 = np.array(buf[-30:]) if len(buf) >= 30 else np.array(buf or [0])
        
        r7m = h7.mean(); r14m = h14.mean(); r30m = h30.mean()
        r30s = h30.std() if len(h30) > 1 else 0.0
        mom  = r7m / (q90_mean + 0.001)
        
        feat = pd.DataFrame([{
            'sku_code': sku_c, 'cat_code': cat_c,
            'day_of_week': d.dayofweek, 'month': d.month, 'is_weekend': 1 if d.dayofweek in [5, 6] else 0,
            'selling_price': price, 'price_vs_annual': 1.0,
            'aug2025_11d_baseline': aug25_b, 'annual_365d_mean': ann_mean,
            'annual_active_rate': act_rate, 'qtr_90d_mean': q90_mean, 'cat_aug_index': cat_idx,
            'lag_1': lag(1), 'lag_3': lag(3), 'lag_7': lag(7), 'lag_14': lag(14), 'lag_28': lag(28),
            'roll_7_mean': r7m, 'roll_14_mean': r14m, 'roll_30_mean': r30m, 'roll_30_std': r30s,
            'momentum_7_vs_90': mom
        }])
        
        p = float(np.maximum(0, model.predict(feat[FEATURES])[0]))
        
        # Balance short-term lag prediction with 1-Year August Pattern Anchor
        # If product had August 2025 history, ensure prediction considers same-period baseline
        if aug25_b > 0 and ann_mean > 2.0:
            pattern_daily_anchor = (0.45 * (aug25_b / 11.0)) + (0.35 * ann_mean) + (0.20 * p)
            p = (0.60 * p) + (0.40 * pattern_daily_anchor)
            
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

# 6. Save to SQLite and Excel
conn = sqlite3.connect(DB_PATH)
df_out.to_sql('forecast_aug_2026_1year_pattern', conn, if_exists='replace', index=False)
conn.close()

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

try:
    save_excel(OUT_EXCEL, df_out)
    print(f"\nExcel Report Saved: {OUT_EXCEL}")
except PermissionError:
    alt = OUT_EXCEL.replace('.xlsx', '_FINAL_1YEAR.xlsx')
    save_excel(alt, df_out)
    print(f"\nExcel Report Saved (Alt): {alt}")

print("\n" + "="*85)
print("1-YEAR PATTERN FORECAST COMPLETE! TOP 20 PREVIEW:")
print("="*85)
print(df_out.head(20).to_string(index=False))
