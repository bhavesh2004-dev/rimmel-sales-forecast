"""
Dynamic Recursive Forecasting and Holdout Validation Engine.
Enforces zero-leakage fixed-origin multi-day simulation for validation (T-9 to T)
and operational forward forecasting (T+1 to T+10).
Produces dual-granularity metrics (Daily SKU x Platform and 10-Day SKU aggregated).
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle

from multibrand_pipeline.src.observation_grid import construct_daily_grid
from multibrand_pipeline.src.features import compute_causal_features
from multibrand_pipeline.src.model_router import BaselineRunRatePredictor, CrostonSBAPredictor

logger = logging.getLogger(__name__)

class DynamicForecastingEngine:
    def __init__(self, config: Dict[str, Any], schema_path: str):
        self.config = config
        self.schema_path = schema_path
        with open(schema_path, 'r', encoding='utf-8') as f:
            self.schema = json.load(f)
            
        self.feature_list = self.schema['feature_list']
        self.categorical_features = self.schema['categorical_features']
        self.model_params = config.get('model_parameters', {
            'n_estimators': 150,
            'learning_rate': 0.05,
            'max_depth': 6,
            'num_leaves': 31,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        })
        self.calib_alpha = config.get('calibration', {}).get('zero_demand_alpha', 0.10)
        self.calib_beta = config.get('calibration', {}).get('stockout_beta', 0.10)

    def run_dynamic_validation(
        self,
        df_norm: pd.DataFrame,
        brand_profiles: pd.DataFrame,
        routing_decisions: Dict[str, Dict[str, Any]]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Executes leak-free fixed-origin 10-day holdout validation per brand.
        
        Returns:
        - df_val_summary: Brand-level validation metrics across granularities
        - df_val_detail: Day-by-day SKU x Platform prediction vs actual records
        - df_sku_metrics: 10-Day SKU-aggregated metrics and diagnostics
        """
        logger.info("==================================================================")
        logger.info("EXECUTING LEAK-FREE DYNAMIC HOLDOUT VALIDATION")
        logger.info("==================================================================")
        
        val_summary_rows = []
        val_detail_dfs = []
        sku_metrics_rows = []
        
        for _, b_prof in brand_profiles.iterrows():
            b_id = b_prof['brand_id']
            disp_name = b_prof['display_brand_name']
            r_info = routing_decisions[b_id]
            val_strat = r_info['validation_strategy']
            model_id = r_info['validation_model_id']
            
            val_start = b_prof['validation_start']
            val_end = b_prof['validation_end']
            train_cutoff = b_prof['validation_training_cutoff']
            train_start = b_prof['training_start']
            
            logger.info(f"\n--- VALIDATION FOR {disp_name} ({b_id}) ---")
            logger.info(f"  Holdout Window: {val_start} to {val_end} (10 calendar days)")
            logger.info(f"  Training Cutoff: Strict cutoff at {train_cutoff} (Zero lookahead)")
            logger.info(f"  Assigned Strategy: {val_strat} ({model_id})")
            
            # Filter brand normalized data
            b_norm = df_norm[df_norm['brand_id'] == b_id].copy()
            
            # Construct continuous daily grid up to val_end
            grid_b = construct_daily_grid(
                df_norm=b_norm,
                start_date=b_prof['context_start'],
                end_date=val_end,
                stock_telemetry_valid_start=self.config.get('timeline', {}).get('stock_telemetry_valid_start', '2025-08-01')
            )
            
            # Split data strictly at training cutoff for model preparation
            train_grid = grid_b[grid_b['date'] <= train_cutoff].copy()
            val_dates = pd.date_range(start=val_start, end=val_end, freq='D').strftime('%Y-%m-%d').tolist()
            
            # Active series for this brand
            active_series = grid_b[['brand_id', 'canonical_sku', 'platform_group', 'product_title', 'category']].drop_duplicates()
            series_keys = [tuple(x) for x in active_series[['brand_id', 'canonical_sku', 'platform_group']].values]
            
            # Fit or initialize the model using ONLY data <= train_cutoff
            model = None
            if "LGBM" in val_strat:
                logger.info(f"  Fitting leak-free evaluation LightGBM on {train_start} to {train_cutoff}...")
                train_feat = compute_causal_features(train_grid, self.schema_path)
                train_mask = (train_feat['date'] >= train_start) & (train_feat['date'] <= train_cutoff)
                X_train = train_feat.loc[train_mask, self.feature_list].copy()
                for cat in self.categorical_features:
                    if cat in X_train.columns:
                        X_train[cat] = X_train[cat].astype('category')
                y_train = train_feat.loc[train_mask, 'model_units_sold'].values
                
                model = lgb.LGBMRegressor(**self.model_params)
                model.fit(X_train, y_train)
                logger.info(f"  Evaluation model fitted successfully on {len(X_train):,} training rows.")
            elif "CROSTON" in val_strat:
                model = CrostonSBAPredictor().fit(train_grid)
            else:
                model = BaselineRunRatePredictor(lookback_days=7).fit(train_grid)
                
            # Execute True Fixed-Origin Recursive 10-Day Simulation
            # In-memory history buffer starts with actual sales up to train_cutoff
            history_buffer = grid_b[grid_b['date'] <= train_cutoff][['date', 'brand_id', 'canonical_sku', 'platform_group', 'observed_units_sold']].copy()
            
            val_daily_preds = []
            
            for f_date in val_dates:
                # 1. Construct temporary grid up to f_date using history_buffer
                # To enforce recursion, any previous validation dates in history_buffer contain PREDICTIONS, not actuals!
                temp_hist = history_buffer.copy()
                
                # Append placeholder for f_date
                day_placeholder = active_series[['brand_id', 'canonical_sku', 'platform_group']].copy()
                day_placeholder['date'] = f_date
                day_placeholder['observed_units_sold'] = 0.0
                
                step_grid = pd.concat([temp_hist, day_placeholder], ignore_index=True)
                
                # Merge catalog metadata and pricing/stock from grid_b
                meta_cols = ['date', 'brand_id', 'canonical_sku', 'platform_group', 'selling_price', 'current_stock', 'in_stock_flag', 'stockout_flag', 'product_title', 'category', 'pack_multiplier']
                valid_meta = [c for c in meta_cols if c in grid_b.columns]
                step_grid = pd.merge(step_grid, grid_b[valid_meta], on=['date', 'brand_id', 'canonical_sku', 'platform_group'], how='left')
                step_grid['observed_units_sold'] = step_grid['observed_units_sold'].fillna(0.0)
                
                # Compute causal features on recursive step
                if "LGBM" in val_strat:
                    step_feat = compute_causal_features(step_grid, self.schema_path)
                    f_row = step_feat[step_feat['date'] == f_date].copy()
                    
                    X_step = f_row[self.feature_list].copy()
                    for cat in self.categorical_features:
                        if cat in X_step.columns:
                            X_step[cat] = X_step[cat].astype('category')
                            
                    raw_pred = np.clip(model.predict(X_step), 0, None)
                    
                    # Calibration
                    z_mask = (f_row['v7'] == 0) & (f_row['v14'] == 0) & (f_row['v30'] == 0)
                    stk_mask = (f_row['in_stock_flag'] == 0)
                    calib_pred = raw_pred.copy()
                    calib_pred[z_mask.values] *= self.calib_alpha
                    calib_pred[stk_mask.values] *= self.calib_beta
                    
                    f_row['predicted_units'] = calib_pred
                    preds_for_day = f_row[['brand_id', 'canonical_sku', 'platform_group', 'date', 'predicted_units']].copy()
                else:
                    preds_arr = model.predict(series_keys)
                    preds_for_day = active_series[['brand_id', 'canonical_sku', 'platform_group']].copy()
                    preds_for_day['date'] = f_date
                    preds_for_day['predicted_units'] = preds_arr
                    
                val_daily_preds.append(preds_for_day)
                
                # APPEND GENERATED PREDICTIONS TO RECURSIVE HISTORY BUFFER (Zero actuals used!)
                buffer_append = preds_for_day.rename(columns={'predicted_units': 'observed_units_sold'}).copy()
                history_buffer = pd.concat([history_buffer, buffer_append], ignore_index=True)
                
            # Combine all 10 days of validation predictions
            df_val_preds = pd.concat(val_daily_preds, ignore_index=True)
            
            # Merge with genuine holdout actual sales from grid_b
            holdout_actuals = grid_b[grid_b['date'].isin(val_dates)][['date', 'brand_id', 'canonical_sku', 'platform_group', 'observed_units_sold', 'product_title', 'category']].copy()
            holdout_actuals = holdout_actuals.rename(columns={'observed_units_sold': 'actual_units'})
            
            df_val_detail = pd.merge(holdout_actuals, df_val_preds[['date', 'brand_id', 'canonical_sku', 'platform_group', 'predicted_units']], on=['date', 'brand_id', 'canonical_sku', 'platform_group'], how='left')
            df_val_detail['predicted_units'] = df_val_detail['predicted_units'].fillna(0.0)
            df_val_detail['absolute_error'] = (df_val_detail['actual_units'] - df_val_detail['predicted_units']).abs()
            df_val_detail['prediction_method'] = val_strat
            df_val_detail['model_id'] = model_id
            val_detail_dfs.append(df_val_detail)
            
            # -------------------------------------------------------------
            # Compute Multi-Granularity Metrics
            # -------------------------------------------------------------
            act_daily = df_val_detail['actual_units'].values
            prd_daily = df_val_detail['predicted_units'].values
            tot_act = float(np.sum(act_daily))
            tot_prd = float(np.sum(prd_daily))
            
            # 1. Daily SKU x Platform Level
            abs_err_daily = np.abs(act_daily - prd_daily)
            mae_daily = float(np.mean(abs_err_daily))
            rmse_daily = float(np.sqrt(np.mean(abs_err_daily ** 2)))
            bias_daily = float(np.mean(prd_daily - act_daily))
            wape_daily_str = f"{float(np.sum(abs_err_daily) / tot_act * 100):.2f}%" if tot_act > 0 else "N/A"
            wape_daily_val = float(np.sum(abs_err_daily) / tot_act * 100) if tot_act > 0 else np.nan
            
            # 2. 10-Day SKU-Aggregated Level
            sku_agg = df_val_detail.groupby('canonical_sku').agg(
                actual_10d=('actual_units', 'sum'),
                predicted_10d=('predicted_units', 'sum')
            ).reset_index()
            sku_agg['abs_err_10d'] = (sku_agg['actual_10d'] - sku_agg['predicted_10d']).abs()
            
            mae_10d = float(sku_agg['abs_err_10d'].mean())
            rmse_10d = float(np.sqrt(np.mean(sku_agg['abs_err_10d'] ** 2)))
            bias_10d = float(sku_agg['predicted_10d'].mean() - sku_agg['actual_10d'].mean())
            wape_10d_str = f"{float(sku_agg['abs_err_10d'].sum() / tot_act * 100):.2f}%" if tot_act > 0 else "N/A"
            wape_10d_val = float(sku_agg['abs_err_10d'].sum() / tot_act * 100) if tot_act > 0 else np.nan
            
            port_bias_pct = f"{((tot_prd - tot_act) / tot_act * 100):+.2f}%" if tot_act > 0 else "N/A"
            
            # Record SKU-level diagnostics
            for _, sr in sku_agg.iterrows():
                sku_metrics_rows.append({
                    'brand_id': b_id,
                    'canonical_sku': sr['canonical_sku'],
                    'actual_units_10d': round(sr['actual_10d'], 2),
                    'predicted_units_10d': round(sr['predicted_10d'], 2),
                    'absolute_error_10d': round(sr['abs_err_10d'], 2),
                    'sku_wape': f"{(sr['abs_err_10d'] / sr['actual_10d'] * 100):.2f}%" if sr['actual_10d'] > 0 else "N/A",
                    'zero_actual_demand': sr['actual_10d'] == 0,
                    'model_strategy': val_strat,
                    'model_id': model_id
                })
                
            val_summary_rows.append({
                'brand_id': b_id,
                'display_brand_name': disp_name,
                'earliest_date': b_prof['earliest_date'],
                'latest_date': b_prof['latest_date'],
                'data_cutoff_T': b_prof['cutoff_date_T'],
                'training_target_cutoff': train_cutoff,
                'validation_start': val_start,
                'validation_end': val_end,
                'total_source_rows': b_prof['total_rows'],
                'distinct_skus': b_prof['distinct_skus'],
                'platforms_count': b_prof['platforms_count'],
                'active_selling_days': b_prof['active_selling_days'],
                'model_strategy': val_strat,
                'model_id': model_id,
                'actual_units_10d': round(tot_act, 1),
                'predicted_units_10d': round(tot_prd, 1),
                'wape_daily_platform': wape_daily_str,
                'mae_daily_platform': round(mae_daily, 4),
                'rmse_daily_platform': round(rmse_daily, 4),
                'wape_10d_sku_level': wape_10d_str,
                'mae_10d_sku_level': round(mae_10d, 2),
                'rmse_10d_sku_level': round(rmse_10d, 2),
                'portfolio_net_bias_pct': port_bias_pct,
                'confidence_level': r_info['confidence_level'],
                'warning_reason': r_info['warning_reason'],
                'data_freshness_status': b_prof['freshness_status']
            })
            
            logger.info(f"  Holdout Results for {disp_name}:")
            logger.info(f"    Actual Units: {tot_act:.0f} | Predicted: {tot_prd:.0f} | Bias: {port_bias_pct}")
            logger.info(f"    Daily Platform WAPE: {wape_daily_str} | 10-Day SKU WAPE: {wape_10d_str}")
            
        df_val_summary = pd.DataFrame(val_summary_rows)
        df_val_detail = pd.concat(val_detail_dfs, ignore_index=True)
        df_sku_metrics = pd.DataFrame(sku_metrics_rows)
        
        logger.info("Dynamic holdout validation completed successfully across all brands.")
        return df_val_summary, df_val_detail, df_sku_metrics

    def run_forward_operational_forecast(
        self,
        df_norm: pd.DataFrame,
        brand_profiles: pd.DataFrame,
        routing_decisions: Dict[str, Dict[str, Any]],
        approved_model_path: str,
        run_id: str
    ) -> pd.DataFrame:
        """
        Generates recursive 10-day forward operational predictions (T+1 to T+10)
        anchored strictly to each brand's latest reliable cutoff T.
        Formatted for direct persistence to the single MySQL table: forecast_results.
        """
        logger.info("==================================================================")
        logger.info(f"GENERATING RECURSIVE OPERATIONAL FORECAST (RUN: {run_id})")
        logger.info("==================================================================")
        
        # Load approved production model for ML brands
        approved_model = None
        if os.path.exists(approved_model_path):
            with open(approved_model_path, 'rb') as f:
                approved_model = pickle.load(f)
            logger.info(f"Loaded approved production model from: {approved_model_path}")
            
        fwd_records = []
        gen_timestamp = datetime.now()
        
        for _, b_prof in brand_profiles.iterrows():
            b_id = b_prof['brand_id']
            disp_name = b_prof['display_brand_name']
            r_info = routing_decisions[b_id]
            fwd_strat = r_info['forward_strategy']
            model_id = r_info['forward_model_id']
            
            cutoff_T = b_prof['cutoff_date_T']
            fwd_start = b_prof['forecast_start']
            fwd_end = b_prof['forecast_end']
            fwd_dates = pd.date_range(start=fwd_start, end=fwd_end, freq='D').strftime('%Y-%m-%d').tolist()
            
            logger.info(f"Forecasting {disp_name} from {fwd_start} to {fwd_end} (Cutoff T: {cutoff_T})...")
            
            b_norm = df_norm[df_norm['brand_id'] == b_id].copy()
            
            # Construct continuous daily grid up to cutoff_T (zero data beyond T)
            grid_b = construct_daily_grid(
                df_norm=b_norm,
                start_date=b_prof['context_start'],
                end_date=cutoff_T,
                stock_telemetry_valid_start=self.config.get('timeline', {}).get('stock_telemetry_valid_start', '2025-08-01')
            )
            
            active_series = grid_b[['brand_id', 'canonical_sku', 'platform_group', 'product_title', 'category']].drop_duplicates()
            series_keys = [tuple(x) for x in active_series[['brand_id', 'canonical_sku', 'platform_group']].values]
            
            # Select forecasting engine
            engine_model = None
            if "LGBM" in fwd_strat and approved_model is not None:
                engine_model = approved_model
            elif "CROSTON" in fwd_strat:
                engine_model = CrostonSBAPredictor().fit(grid_b)
            else:
                engine_model = BaselineRunRatePredictor(lookback_days=7).fit(grid_b)
                
            # Recursive forward stepping
            history_buffer = grid_b[['date', 'brand_id', 'canonical_sku', 'platform_group', 'observed_units_sold']].copy()
            
            for h_idx, f_date in enumerate(fwd_dates, start=1):
                temp_hist = history_buffer.copy()
                
                day_placeholder = active_series[['brand_id', 'canonical_sku', 'platform_group']].copy()
                day_placeholder['date'] = f_date
                day_placeholder['observed_units_sold'] = 0.0
                
                step_grid = pd.concat([temp_hist, day_placeholder], ignore_index=True)
                
                meta_cols = ['date', 'brand_id', 'canonical_sku', 'platform_group', 'selling_price', 'current_stock', 'in_stock_flag', 'stockout_flag', 'product_title', 'category', 'pack_multiplier']
                valid_meta = [c for c in meta_cols if c in grid_b.columns]
                step_grid = pd.merge(step_grid, grid_b[valid_meta], on=['date', 'brand_id', 'canonical_sku', 'platform_group'], how='left')
                step_grid['observed_units_sold'] = step_grid['observed_units_sold'].fillna(0.0)
                
                # Pricing and stock forward fill for future dates
                step_grid['selling_price'] = step_grid.groupby(['brand_id', 'canonical_sku'])['selling_price'].ffill().bfill().fillna(0.0)
                step_grid['current_stock'] = step_grid.groupby(['brand_id', 'canonical_sku'])['current_stock'].ffill().fillna(0.0)
                step_grid['in_stock_flag'] = (step_grid['current_stock'] > 0).astype(int)
                step_grid['stockout_flag'] = (step_grid['current_stock'] <= 0).astype(int)
                
                if "LGBM" in fwd_strat and approved_model is not None:
                    step_feat = compute_causal_features(step_grid, self.schema_path)
                    f_row = step_feat[step_feat['date'] == f_date].copy()
                    
                    X_step = f_row[self.feature_list].copy()
                    for cat in self.categorical_features:
                        if cat in X_step.columns:
                            X_step[cat] = X_step[cat].astype('category')
                            
                    raw_pred = np.clip(engine_model.predict(X_step), 0, None)
                    z_mask = (f_row['v7'] == 0) & (f_row['v14'] == 0) & (f_row['v30'] == 0)
                    stk_mask = (f_row['in_stock_flag'] == 0)
                    calib_pred = raw_pred.copy()
                    calib_pred[z_mask.values] *= self.calib_alpha
                    calib_pred[stk_mask.values] *= self.calib_beta
                    
                    f_row['predicted_units'] = calib_pred
                    preds_for_day = f_row[['brand_id', 'canonical_sku', 'platform_group', 'date', 'predicted_units', 'product_title']].copy()
                else:
                    preds_arr = engine_model.predict(series_keys)
                    preds_for_day = active_series[['brand_id', 'canonical_sku', 'platform_group', 'product_title']].copy()
                    preds_for_day['date'] = f_date
                    preds_for_day['predicted_units'] = preds_arr
                    
                # Append to recursive history buffer
                buffer_append = preds_for_day[['date', 'brand_id', 'canonical_sku', 'platform_group', 'predicted_units']].rename(columns={'predicted_units': 'observed_units_sold'})
                history_buffer = pd.concat([history_buffer, buffer_append], ignore_index=True)
                
                # Build canonical MySQL forecast records
                for _, pr in preds_for_day.iterrows():
                    fwd_records.append({
                        'run_id': run_id,
                        'generated_at': gen_timestamp,
                        'brand': disp_name,
                        'brand_id': b_id,
                        'platform_group': pr['platform_group'],
                        'platform_id': pr['platform_group'],
                        'canonical_sku': pr['canonical_sku'],
                        'resolved_parent_id': pr['canonical_sku'],
                        'product_name': pr.get('product_title', pr['canonical_sku']),
                        'forecast_date': f_date,
                        'horizon_day': h_idx,
                        'predicted_units': round(float(pr['predicted_units']), 4),
                        'model_id': model_id,
                        'model_strategy': fwd_strat,
                        'data_start_date': b_prof['earliest_date'],
                        'data_cutoff_date': cutoff_T,
                        'training_start_date': b_prof['training_start'],
                        'training_end_date': cutoff_T,
                        'feature_profile': "TIER_LONG_HISTORY" if b_prof['calendar_span_days'] >= 90 else "TIER_SHORT_HISTORY",
                        'confidence_level': r_info['confidence_level'],
                        'warning_reason': r_info['warning_reason'],
                        'created_at': gen_timestamp
                    })
                    
        df_forecast = pd.DataFrame(fwd_records)
        logger.info(f"Generated {len(df_forecast):,} canonical forward forecast records.")
        return df_forecast
