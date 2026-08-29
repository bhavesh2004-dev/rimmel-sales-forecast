"""
Feature Engineering Matrix Module (Version 3.0)
Constructs calendar features, long-term lags, rolling averages, trend velocity,
PLUS new Selling Price ($ USD) elasticity features and Product Category encodings.
"""
import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # 1. Target Column Check
    target_col = 'target_sales' if 'target_sales' in df.columns else 'sold_qty'
    if target_col not in df.columns and 'sold_quantity' in df.columns:
        target_col = 'sold_quantity'
    
    # 2. Calendar & Seasonality Features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # 3. Direct Long-Term Lags (Excludes lag_1..6 to prevent compounding error)
    df['lag_7'] = df[target_col].shift(7)
    df['lag_14'] = df[target_col].shift(14)
    df['lag_30'] = df[target_col].shift(30)
    
    # 4. Rolling Baseline Averages (Shifted by 1 to use up-to-yesterday sales)
    df['rolling_mean_7'] = df[target_col].shift(1).rolling(window=7, min_periods=3).mean()
    df['rolling_mean_30'] = df[target_col].shift(1).rolling(window=30, min_periods=7).mean()
    
    # 5. Trend Velocity Indicator
    df['trend_velocity'] = (df['rolling_mean_7'] + 1e-5) / (df['rolling_mean_30'] + 1e-5)
    
    # 6. NEW FEATURE: Selling Price ($ USD) Elasticity Features
    if 'selling_price' in df.columns:
        df['selling_price'] = df['selling_price'].ffill().bfill().fillna(5.0)
        df['rolling_mean_price_30d'] = df['selling_price'].shift(1).rolling(window=30, min_periods=7).mean().fillna(df['selling_price'])
        
        # Price Discount Ratio (< 1.0 means selling below normal 30-day baseline)
        df['price_discount_ratio'] = (df['selling_price'] + 1e-5) / (df['rolling_mean_price_30d'] + 1e-5)
        
        # Percentage Price Drop
        df['price_drop_pct'] = np.maximum(0, (df['rolling_mean_price_30d'] - df['selling_price']) / (df['rolling_mean_price_30d'] + 1e-5) * 100)
        
        # Binary Discount Flag (1 if price dropped > 10% below 30-day baseline)
        df['is_discounted'] = (df['price_discount_ratio'] < 0.90).astype(int)
    else:
        df['selling_price'] = 5.0
        df['price_discount_ratio'] = 1.0
        df['price_drop_pct'] = 0.0
        df['is_discounted'] = 0
        
    # 7. NEW FEATURE: Category Frequency Encoding
    if 'category' in df.columns:
        cat_freq = df['category'].value_counts().to_dict()
        df['category_freq_encoded'] = df['category'].map(cat_freq).fillna(0)
    else:
        df['category_freq_encoded'] = 0
        
    return df

def get_feature_columns() -> list:
    return [
        'day_of_week', 'day_of_month', 'month', 'is_weekend',
        'lag_7', 'lag_14', 'lag_30',
        'rolling_mean_7', 'rolling_mean_30', 'trend_velocity',
        'selling_price', 'price_discount_ratio', 'price_drop_pct', 'is_discounted',
        'category_freq_encoded'
    ]
