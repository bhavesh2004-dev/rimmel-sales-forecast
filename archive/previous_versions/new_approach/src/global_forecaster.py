"""
Global LightGBM Tweedie Engine & Walk-Forward Evaluator (588 SKUs)
Features:
- Single Global LightGBM model trained with Tweedie Loss (objective='tweedie', power=1.5)
- Handles both zero-heavy intermittent SKUs and sales spike days simultaneously
- 3-Fold Walk-Forward Cross Validation (May, June, July 2026)
- Measures WAPE, Bias, RMSE, and Volume Accuracy %
"""
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_wape(y_true, y_pred):
    total_true = np.sum(y_true)
    if total_true == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / total_true * 100)

def calculate_bias(y_true, y_pred):
    total_true = np.sum(y_true)
    if total_true == 0:
        return 0.0
    return float(np.sum(y_pred - y_true) / total_true * 100)

def calculate_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def train_tweedie_model(X_train, y_train):
    params = {
        'objective': 'tweedie',
        'tweedie_variance_power': 1.5,
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'min_data_in_leaf': 20,
        'bagging_fraction': 0.8,
        'feature_fraction': 0.8,
        'random_state': 42,
        'verbosity': -1,
        'n_estimators': 150
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return model

def run_walk_forward_cv(df_featured, feature_cols):
    print("\n" + "="*85)
    print("RUNNING 3-FOLD WALK-FORWARD TIME SERIES CROSS-VALIDATION (MAY, JUNE, JULY 2026)")
    print("="*85)
    
    folds = [
        ('May 2026', '2026-05-01', '2026-05-31'),
        ('June 2026', '2026-06-01', '2026-06-30'),
        ('July 2026', '2026-07-01', '2026-07-31')
    ]
    
    cv_metrics = []
    
    for f_name, s_date, e_date in folds:
        train_mask = df_featured['date'] < s_date
        test_mask = (df_featured['date'] >= s_date) & (df_featured['date'] <= e_date)
        
        train_df = df_featured[train_mask].dropna(subset=feature_cols)
        test_df = df_featured[test_mask].dropna(subset=feature_cols)
        
        model = train_tweedie_model(train_df[feature_cols], train_df['target_sales'])
        preds = np.maximum(0, model.predict(test_df[feature_cols]))
        
        actual_total = test_df['sold_qty'].sum()
        pred_total = preds.sum()
        
        wape = calculate_wape(test_df['sold_qty'].values, preds)
        bias = calculate_bias(test_df['sold_qty'].values, preds)
        rmse = calculate_rmse(test_df['sold_qty'].values, preds)
        mae = mean_absolute_error(test_df['sold_qty'].values, preds)
        vol_acc = max(0.0, (1 - abs(actual_total - pred_total) / max(1, actual_total)) * 100)
        
        cv_metrics.append({
            'Test Window': f_name,
            'Actual Units': int(actual_total),
            'Predicted Units': int(pred_total),
            'WAPE %': round(wape, 1),
            'Bias %': round(bias, 1),
            'RMSE Error': round(rmse, 2),
            'MAE Error': round(mae, 2),
            'Volume Accuracy %': round(vol_acc, 1)
        })
        
        print(f"[{f_name}] Actual: {actual_total:,.0f} | Pred: {pred_total:,.0f} | WAPE: {wape:.1f}% | Bias: {bias:+.1f}% | Vol Acc: {vol_acc:.1f}%")
        
    res_df = pd.DataFrame(cv_metrics)
    
    avg_wape = res_df['WAPE %'].mean()
    avg_bias = res_df['Bias %'].mean()
    avg_vol_acc = res_df['Volume Accuracy %'].mean()
    avg_rmse = res_df['RMSE Error'].mean()
    
    print("\n" + "="*85)
    print("WALK-FORWARD CROSS-VALIDATION SUMMARY ACROSS ALL 588 SKUs:")
    print("="*85)
    print(f"Average Catalog WAPE %           : {avg_wape:.1f}%  (Honest daily forecasting error)")
    print(f"Average Forecast Bias %          : {avg_bias:+.1f}% (Net over/under-ordering bias)")
    print(f"Average Volume-Weighted Accuracy : {avg_vol_acc:.1f}%  (Monthly total volume accuracy)")
    print(f"Average Daily RMSE               : {avg_rmse:.2f} units")
    print("="*85)
    
    return res_df
