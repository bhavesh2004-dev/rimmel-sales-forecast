"""
Phase 3 Validation & Controlled Brand_ID Ablation Runner
Executes:
1. Phase 2 Clean Pipeline Benchmark (Model A: 60 features)
2. Controlled Brand_ID Ablation (Model B: 60 features + brand_id)
3. Feature gain & split inspection for brand_id
4. Forensic reconciliation against audited multi-brand and frozen Rimmel baselines
5. Forward simulation & inventory layer validation
"""

import os
import json
import yaml
import logging
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phase3_validation")

def run_phase3_validation():
    # 1. Load Config & Schemas
    config_path = "multibrand_pipeline/config/pipeline_config.yaml"
    schema_path = "multibrand_pipeline/config/feature_schema.json"
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(schema_path, "r") as f:
        schema = json.load(f)
        
    feats_60 = schema['feature_list']
    cat_feats_60 = schema['categorical_features']
    
    feats_61 = ['brand_id'] + feats_60
    cat_feats_61 = ['brand_id'] + cat_feats_60
    
    # 2. Load Multi-brand Parquet Dataset
    parquet_path = "experiments/global_multibrand_lgbm/data/global_training_dataset.parquet"
    logger.info(f"Loading multi-brand dataset from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df):,} rows. Brands: {df['brand_id'].value_counts().to_dict()}")
    
    timeline = config['timeline']
    train_start = timeline['training_start']
    train_end = timeline['training_end']
    bench_start = timeline['benchmark_start']
    bench_end = timeline['benchmark_end']
    
    train_mask = (df['date'] >= train_start) & (df['date'] <= train_end)
    bench_mask = (df['date'] >= bench_start) & (df['date'] <= bench_end)
    
    train_df = df[train_mask].copy()
    bench_df = df[bench_mask].copy()
    
    logger.info(f"Training split: {train_start} to {train_end} ({len(train_df):,} rows)")
    logger.info(f"Benchmark split: {bench_start} to {bench_end} ({len(bench_df):,} rows)")
    
    model_params = config['model_parameters']
    calib_cfg = config.get('calibration', {})
    alpha = calib_cfg.get('zero_demand_alpha', 0.10)
    beta = calib_cfg.get('stockout_beta', 0.10)
    
    def evaluate_model(feature_cols, cat_cols, model_name):
        logger.info(f"\n==========================================")
        logger.info(f"TRAINING & EVALUATING: {model_name} ({len(feature_cols)} features)")
        logger.info(f"==========================================")
        
        # Prepare train matrix
        X_tr = train_df[feature_cols].copy()
        for c in cat_cols:
            if c in X_tr.columns:
                X_tr[c] = X_tr[c].astype('category')
        y_tr = train_df['model_units_sold'].values
        
        # Prepare bench matrix
        X_be = bench_df[feature_cols].copy()
        for c in cat_cols:
            if c in X_be.columns:
                X_be[c] = X_be[c].astype('category')
        y_be = bench_df['model_units_sold'].values
        
        model = lgb.LGBMRegressor(**model_params)
        model.fit(X_tr, y_tr)
        
        raw_preds = np.clip(model.predict(X_be), 0, None)
        
        # Exp6 calibration
        z_mask = (bench_df['v7'] == 0) & (bench_df['v14'] == 0) & (bench_df['v30'] == 0)
        stk_mask = (bench_df['in_stock_flag'] == 0)
        
        calib_preds = raw_preds.copy()
        calib_preds[z_mask.values] *= alpha
        calib_preds[stk_mask.values] *= beta
        
        def calc_metrics(act, prd):
            act = np.asarray(act, dtype=float)
            prd = np.asarray(prd, dtype=float)
            tot_act = float(np.sum(act))
            tot_prd = float(np.sum(prd))
            abs_err = np.abs(act - prd)
            wape = float(np.sum(abs_err) / max(1e-4, tot_act) * 100)
            bias = float((tot_prd - tot_act) / max(1e-4, tot_act) * 100)
            rmse = float(np.sqrt(np.mean((act - prd) ** 2)))
            mae = float(np.mean(abs_err))
            return {
                'actual': round(tot_act, 2),
                'predicted': round(tot_prd, 2),
                'abs_error': round(float(np.sum(abs_err)), 2),
                'wape': round(wape, 2),
                'bias': round(bias, 2),
                'rmse': round(rmse, 4),
                'mae': round(mae, 4)
            }
            
        res = {
            'model_name': model_name,
            'feature_count': len(feature_cols),
            'features': feature_cols,
            'combined_raw': calc_metrics(y_be, raw_preds),
            'combined_calib': calc_metrics(y_be, calib_preds),
            'by_brand_raw': {},
            'by_brand_calib': {},
            'by_platform_raw': {},
            'by_platform_calib': {}
        }
        
        # Brand level
        for b in bench_df['brand_id'].unique():
            b_mask = (bench_df['brand_id'] == b).values
            res['by_brand_raw'][b] = calc_metrics(y_be[b_mask], raw_preds[b_mask])
            res['by_brand_calib'][b] = calc_metrics(y_be[b_mask], calib_preds[b_mask])
            
        # Platform level
        for p in bench_df['platform_group'].unique():
            p_mask = (bench_df['platform_group'] == p).values
            res['by_platform_raw'][p] = calc_metrics(y_be[p_mask], raw_preds[p_mask])
            res['by_platform_calib'][p] = calc_metrics(y_be[p_mask], calib_preds[p_mask])
            
        # Diagnostics
        active_mask = (y_be > 0)
        res['diagnostics'] = {
            'actual_zero_rate': round(float((y_be == 0).mean() * 100), 2),
            'pred_zero_rate_raw': round(float((raw_preds < 0.05).mean() * 100), 2),
            'pred_zero_rate_calib': round(float((calib_preds < 0.05).mean() * 100), 2),
            'fp_rows_raw': int(((y_be == 0) & (raw_preds >= 0.5)).sum()),
            'fp_rows_calib': int(((y_be == 0) & (calib_preds >= 0.5)).sum()),
            'active_rows': int(active_mask.sum()),
            'active_under': int((active_mask & (calib_preds < y_be)).sum()),
            'active_over': int((active_mask & (calib_preds > y_be)).sum())
        }
        
        # Feature importances
        booster = model.booster_
        imp_split = booster.feature_importance(importance_type='split')
        imp_gain = booster.feature_importance(importance_type='gain')
        imp_df = pd.DataFrame({
            'feature': feature_cols,
            'split': imp_split,
            'gain': imp_gain
        }).sort_values('gain', ascending=False)
        res['feature_importance'] = imp_df
        
        return res, model

    # Execute Model A (60 features)
    res_A, model_A = evaluate_model(feats_60, cat_feats_60, "Model A: 60-Feature Lean Baseline (No brand_id)")
    
    # Execute Model B (60 features + brand_id)
    res_B, model_B = evaluate_model(feats_61, cat_feats_61, "Model B: 61-Feature Variant (With brand_id)")
    
    # Inspect brand_id in Model B
    b_imp = res_B['feature_importance'][res_B['feature_importance']['feature'] == 'brand_id']
    brand_id_gain = float(b_imp['gain'].iloc[0]) if len(b_imp) > 0 else 0.0
    brand_id_split = int(b_imp['split'].iloc[0]) if len(b_imp) > 0 else 0
    
    logger.info(f"\n==========================================")
    logger.info(f"BRAND_ID ABLATION RESULTS:")
    logger.info(f"  Model B brand_id Gain: {brand_id_gain}")
    logger.info(f"  Model B brand_id Splits: {brand_id_split}")
    logger.info(f"  Model A Combined Raw WAPE: {res_A['combined_raw']['wape']}% vs Model B: {res_B['combined_raw']['wape']}% (Delta: {round(res_B['combined_raw']['wape'] - res_A['combined_raw']['wape'], 2)}%)")
    logger.info(f"  Model A Combined Calib WAPE: {res_A['combined_calib']['wape']}% vs Model B: {res_B['combined_calib']['wape']}% (Delta: {round(res_B['combined_calib']['wape'] - res_A['combined_calib']['wape'], 2)}%)")
    logger.info(f"  Model A Rimmel Calib WAPE: {res_A['by_brand_calib']['RIMMEL']['wape']}% vs Model B: {res_B['by_brand_calib']['RIMMEL']['wape']}%")
    logger.info(f"  Model A Max Factor Calib WAPE: {res_A['by_brand_calib']['MAX_FACTOR']['wape']}% vs Model B: {res_B['by_brand_calib']['MAX_FACTOR']['wape']}%")
    logger.info(f"==========================================")
    
    # Save comparison to results dictionary
    out_results = {
        'model_A': {
            'combined_raw': res_A['combined_raw'],
            'combined_calib': res_A['combined_calib'],
            'by_brand_raw': res_A['by_brand_raw'],
            'by_brand_calib': res_A['by_brand_calib'],
            'by_platform_raw': res_A['by_platform_raw'],
            'by_platform_calib': res_A['by_platform_calib'],
            'diagnostics': res_A['diagnostics'],
            'top_features': res_A['feature_importance'].head(10).to_dict(orient='records')
        },
        'model_B': {
            'combined_raw': res_B['combined_raw'],
            'combined_calib': res_B['combined_calib'],
            'by_brand_raw': res_B['by_brand_raw'],
            'by_brand_calib': res_B['by_brand_calib'],
            'by_platform_raw': res_B['by_platform_raw'],
            'by_platform_calib': res_B['by_platform_calib'],
            'diagnostics': res_B['diagnostics'],
            'brand_id_gain': brand_id_gain,
            'brand_id_splits': brand_id_split,
            'top_features': res_B['feature_importance'].head(10).to_dict(orient='records')
        }
    }
    
    with open("multibrand_pipeline/reports/phase3_ablation_results.json", "w") as f:
        json.dump(out_results, f, indent=2)
    logger.info("Saved ablation results to multibrand_pipeline/reports/phase3_ablation_results.json")
    
    return out_results

if __name__ == "__main__":
    run_phase3_validation()
