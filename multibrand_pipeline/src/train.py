"""
Multi-Brand Model Training & Benchmark Module
Trains Global LightGBM Regressor using the audited 60-feature schema.
Evaluates official locked benchmark (Sep 1-10) and supports final refit for operational forecasting.
Principle: Single model serialization standard with rich provenance metadata.
"""

import os
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np
import lightgbm as lgb

logger = logging.getLogger(__name__)

def prepare_feature_matrix(
    df: pd.DataFrame,
    feature_list: List[str],
    categorical_features: List[str]
) -> pd.DataFrame:
    """Selects and formats model feature matrix."""
    X = df[feature_list].copy()
    for col in categorical_features:
        if col in X.columns:
            X[col] = X[col].astype('category')
    return X

def compute_forecasting_metrics(
    actual: np.ndarray,
    predicted: np.ndarray
) -> Dict[str, float]:
    """Computes retail forecasting metrics: WAPE, Bias%, RMSE, MAE."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    
    tot_actual = float(np.sum(actual))
    tot_pred = float(np.sum(predicted))
    abs_err = np.abs(actual - predicted)
    
    wape = float(np.sum(abs_err) / max(1e-4, tot_actual) * 100)
    bias_pct = float((tot_pred - tot_actual) / max(1e-4, tot_actual) * 100)
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mae = float(np.mean(abs_err))
    
    return {
        'total_actual': tot_actual,
        'total_predicted': tot_pred,
        'wape': round(wape, 2),
        'bias_pct': round(bias_pct, 2),
        'rmse': round(rmse, 4),
        'mae': round(mae, 4)
    }

def train_and_evaluate(
    df_features: pd.DataFrame,
    config: Dict[str, Any],
    feature_schema: Dict[str, Any],
    refit_mode: bool = False
) -> Tuple[Any, Dict[str, Any]]:
    """
    Trains Global LightGBM on training window, evaluates on locked benchmark,
    and optionally refits on training + benchmark for final forward forecasting.
    """
    logger.info("Initializing Global LightGBM Training & Benchmark Execution...")
    timeline = config['timeline']
    model_params = config['model_parameters']
    feature_list = feature_schema['feature_list']
    cat_features = feature_schema['categorical_features']
    
    train_start = timeline['training_start']
    train_end = timeline['refit_training_end'] if refit_mode else timeline['training_end']
    bench_start = timeline['benchmark_start']
    bench_end = timeline['benchmark_end']
    
    # 1. Training Split
    train_mask = (df_features['date'] >= train_start) & (df_features['date'] <= train_end)
    train_df = df_features[train_mask].copy()
    logger.info(f"Training set: {train_start} to {train_end} ({len(train_df):,} rows)")
    
    X_train = prepare_feature_matrix(train_df, feature_list, cat_features)
    y_train = train_df['model_units_sold'].values
    
    # 2. Fit Initial Model
    logger.info(f"Fitting Global LightGBM (150 trees, max_depth=6, 60 features)...")
    model = lgb.LGBMRegressor(**model_params)
    model.fit(X_train, y_train)
    
    # 3. Benchmark Evaluation (Always evaluate on locked benchmark)
    bench_mask = (df_features['date'] >= bench_start) & (df_features['date'] <= bench_end)
    bench_df = df_features[bench_mask].copy()
    logger.info(f"Locked benchmark: {bench_start} to {bench_end} ({len(bench_df):,} rows)")
    
    metrics_summary = {}
    if not bench_df.empty:
        X_bench = prepare_feature_matrix(bench_df, feature_list, cat_features)
        y_bench = bench_df['model_units_sold'].values
        raw_preds = np.clip(model.predict(X_bench), 0, None)
        
        # Apply Exp6 Calibration for zero-demand / stockout
        calib_cfg = config.get('calibration', {})
        alpha = calib_cfg.get('zero_demand_alpha', 0.10)
        beta = calib_cfg.get('stockout_beta', 0.10)
        
        z_mask = (bench_df['v7'] == 0) & (bench_df['v14'] == 0) & (bench_df['v30'] == 0)
        stk_mask = (bench_df['in_stock_flag'] == 0)
        
        calib_preds = raw_preds.copy()
        calib_preds[z_mask.values] *= alpha
        calib_preds[stk_mask.values] *= beta
        
        bench_df['prediction_raw'] = raw_preds
        bench_df['prediction_calib'] = calib_preds
        
        overall_raw = compute_forecasting_metrics(y_bench, raw_preds)
        overall_calib = compute_forecasting_metrics(y_bench, calib_preds)
        
        logger.info(f"Benchmark RAW Overall: WAPE={overall_raw['wape']}%, Bias={overall_raw['bias_pct']}%, RMSE={overall_raw['rmse']}")
        logger.info(f"Benchmark CALIB Overall: WAPE={overall_calib['wape']}%, Bias={overall_calib['bias_pct']}%, RMSE={overall_calib['rmse']}")
        
        # Brand breakdown
        brand_metrics = {}
        for b_id in bench_df['brand_id'].unique():
            b_sub = bench_df[bench_df['brand_id'] == b_id]
            brand_metrics[b_id] = {
                'raw': compute_forecasting_metrics(b_sub['model_units_sold'].values, b_sub['prediction_raw'].values),
                'calib': compute_forecasting_metrics(b_sub['model_units_sold'].values, b_sub['prediction_calib'].values)
            }
            logger.info(f"  Brand {b_id}: Raw WAPE={brand_metrics[b_id]['raw']['wape']}%, Calib WAPE={brand_metrics[b_id]['calib']['wape']}%")
            
        # Platform breakdown
        platform_metrics = {}
        for plat in bench_df['platform_group'].unique():
            p_sub = bench_df[bench_df['platform_group'] == plat]
            platform_metrics[plat] = {
                'raw': compute_forecasting_metrics(p_sub['model_units_sold'].values, p_sub['prediction_raw'].values),
                'calib': compute_forecasting_metrics(p_sub['model_units_sold'].values, p_sub['prediction_calib'].values)
            }
            logger.info(f"  Platform {plat}: Raw WAPE={platform_metrics[plat]['raw']['wape']}%, Calib WAPE={platform_metrics[plat]['calib']['wape']}%")
            
        # Error diagnostics
        actual_zero_rate = float((bench_df['model_units_sold'] == 0).mean() * 100)
        pred_zero_raw = float((bench_df['prediction_raw'] < 0.05).mean() * 100)
        pred_zero_calib = float((bench_df['prediction_calib'] < 0.05).mean() * 100)
        
        # False positives (actual == 0, predicted > 0.5)
        fp_rows_raw = int(((bench_df['model_units_sold'] == 0) & (bench_df['prediction_raw'] >= 0.5)).sum())
        fp_rows_calib = int(((bench_df['model_units_sold'] == 0) & (bench_df['prediction_calib'] >= 0.5)).sum())
        
        # Under/Over prediction on active sales rows (actual > 0)
        active_mask = bench_df['model_units_sold'] > 0
        active_under = int((active_mask & (bench_df['prediction_calib'] < bench_df['model_units_sold'])).sum())
        active_over = int((active_mask & (bench_df['prediction_calib'] > bench_df['model_units_sold'])).sum())
        
        # Stockout error
        stk_sub = bench_df[bench_df['in_stock_flag'] == 0]
        stk_metrics = compute_forecasting_metrics(stk_sub['model_units_sold'].values, stk_sub['prediction_calib'].values) if len(stk_sub) > 0 else {}
        
        diagnostics = {
            'actual_zero_rate_pct': round(actual_zero_rate, 2),
            'pred_zero_rate_raw_pct': round(pred_zero_raw, 2),
            'pred_zero_rate_calib_pct': round(pred_zero_calib, 2),
            'false_positive_rows_raw': fp_rows_raw,
            'false_positive_rows_calib': fp_rows_calib,
            'active_sales_rows': int(active_mask.sum()),
            'active_underprediction_rows': active_under,
            'active_overprediction_rows': active_over,
            'stockout_rows': len(stk_sub),
            'stockout_error_metrics': stk_metrics
        }
        
        metrics_summary = {
            'benchmark_period': f"{bench_start} to {bench_end}",
            'total_observations': len(bench_df),
            'overall_raw': overall_raw,
            'overall_calib': overall_calib,
            'by_brand': brand_metrics,
            'by_platform': platform_metrics,
            'diagnostics': diagnostics
        }
        
    # 4. Optional Final Refit Mode
    if refit_mode:
        logger.info(f"Refitting final production model on full window: {train_start} to {timeline['refit_training_end']}...")
        full_mask = (df_features['date'] >= train_start) & (df_features['date'] <= timeline['refit_training_end'])
        full_train_df = df_features[full_mask].copy()
        X_full = prepare_feature_matrix(full_train_df, feature_list, cat_features)
        y_full = full_train_df['model_units_sold'].values
        model = lgb.LGBMRegressor(**model_params)
        model.fit(X_full, y_full)
        logger.info("Final operational model refit complete.")
        
    # 5. Persist Model and Metadata
    model_path = config['paths']['model_path']
    meta_path = config['paths']['model_metadata_path']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    model_metadata = {
        'model_name': 'GlobalMultiBrandLGBM',
        'pipeline_version': config['pipeline']['version'],
        'serialization_format': 'pickle_sklearn_LGBMRegressor',
        'training_window': f"{train_start} to {train_end}",
        'refit_mode_enabled': refit_mode,
        'feature_count': len(feature_list),
        'features': feature_list,
        'categorical_features': cat_features,
        'hyperparameters': model_params,
        'brands_covered': list(df_features['brand_id'].unique()),
        'platforms_covered': list(df_features['platform_group'].unique()),
        'training_rows': len(train_df),
        'benchmark_metrics': metrics_summary,
        'created_at': datetime.now().isoformat()
    }
    
    with open(meta_path, 'w') as f:
        json.dump(model_metadata, f, indent=2)
        
    logger.info(f"Model saved to {model_path} and metadata to {meta_path}")
    return model, model_metadata
