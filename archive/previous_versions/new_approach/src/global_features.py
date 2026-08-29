"""
Global Feature Engineering Engine
Generates:
1. Calendar Features (day_of_week, day_of_month, month, is_weekend, quarter)
2. Frequency & Target Encodings (category, sku)
3. Price Elasticity Features (selling_price, rolling_mean_price_30d, price_discount_ratio, price_drop_pct, is_discounted)
4. Lags & Rolling Features (lag_1, lag_7, lag_14, lag_30, rolling_mean_7d, rolling_mean_14d, rolling_mean_30d, rolling_std_7d)
5. COLD-START IMPUTATION: Missing lags for cold-start SKUs (<60 days) are imputed using Category-Level Baseline Lags!
"""
import pandas as pd
import numpy as np

def generate_global_features(df_clean):
    print("Building global features across full dataset...")
    df = df_clean.copy().sort_values(['sku', 'date']).reset_index(drop=True)
    
    # 1. Calendar Features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # 2. Encodings
    cat_counts = df.groupby('category')['sku'].nunique().to_dict()
    df['category_encoded'] = df['category'].map(cat_counts).fillna(1)
    
    sku_freq = df['sku'].value_counts().to_dict()
    df['sku_encoded'] = df['sku'].map(sku_freq).fillna(1)
    
    # 3. Price Features per SKU
    df['rolling_mean_price_30d'] = df.groupby('sku')['selling_price'].transform(lambda x: x.rolling(30, min_periods=1).mean())
    df['price_discount_ratio'] = df['selling_price'] / np.maximum(0.01, df['rolling_mean_price_30d'])
    df['price_discount_ratio'] = df['price_discount_ratio'].clip(0.5, 2.0)
    
    df['price_drop_pct'] = (df['rolling_mean_price_30d'] - df['selling_price']) / np.maximum(0.01, df['rolling_mean_price_30d'])
    df['price_drop_pct'] = df['price_drop_pct'].clip(-0.5, 0.5)
    df['is_discounted'] = (df['price_drop_pct'] > 0.10).astype(int)
    
    # 4. Lag and Rolling Features per SKU
    for lag in [1, 7, 14, 30]:
        df[f'lag_{lag}'] = df.groupby('sku')['target_sales'].shift(lag)
        
    df['rolling_mean_7d'] = df.groupby('sku')['target_sales'].transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    df['rolling_mean_14d'] = df.groupby('sku')['target_sales'].transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())
    df['rolling_mean_30d'] = df.groupby('sku')['target_sales'].transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
    df['rolling_std_7d'] = df.groupby('sku')['target_sales'].transform(lambda x: x.shift(1).rolling(7, min_periods=1).std()).fillna(0)
    
    # 5. COLD-START IMPUTATION (Category Baseline)
    # Calculate category-level average lag features on established records
    lag_cols = ['lag_1', 'lag_7', 'lag_14', 'lag_30', 'rolling_mean_7d', 'rolling_mean_14d', 'rolling_mean_30d']
    cat_lag_means = df.groupby(['category', 'date'])[lag_cols].transform('mean')
    
    for col in lag_cols:
        # Fill missing SKU lags with category baseline averages
        df[col] = df[col].fillna(cat_lag_means[col]).fillna(0.1)
        
    print("Global Feature Engineering Complete.")
    return df

def get_global_feature_columns():
    return [
        'day_of_week', 'day_of_month', 'month', 'quarter', 'is_weekend',
        'category_encoded', 'sku_encoded',
        'selling_price', 'rolling_mean_price_30d', 'price_discount_ratio', 'price_drop_pct', 'is_discounted',
        'lag_1', 'lag_7', 'lag_14', 'lag_30',
        'rolling_mean_7d', 'rolling_mean_14d', 'rolling_mean_30d', 'rolling_std_7d',
        'history_days'
    ]
