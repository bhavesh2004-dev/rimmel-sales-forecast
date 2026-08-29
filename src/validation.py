"""
VALIDATION & EVALUATION MODULE
==============================
Evaluates forecasting performance on protected holdout slices.
Computes WAPE, MAE, head-to-head model comparison, and confidence calibration.
"""
import pandas as pd
import numpy as np

def evaluate_holdout_performance(forecast_dataframe, ground_truth_sales_map):
    """
    Evaluates forecast accuracy against observed actual sales on a holdout period.
    
    Args:
        forecast_dataframe (pd.DataFrame): Output from dynamic forecasting engine.
        ground_truth_sales_map (dict): Mapping of SKU -> Actual units sold in holdout.
        
    Returns:
        tuple: (evaluated_df: pd.DataFrame, summary_metrics: dict)
    """
    df_eval = forecast_dataframe.copy()
    
    df_eval['Actual Sales'] = df_eval['Product SKU'].map(ground_truth_sales_map).fillna(0).astype(int)
    
    df_eval['Baseline Error']    = (df_eval['Actual Sales'] - df_eval['Baseline Prediction']).abs()
    df_eval['Momentum Error']    = (df_eval['Actual Sales'] - df_eval['Momentum Prediction']).abs()
    df_eval['Recommended Error'] = (df_eval['Actual Sales'] - df_eval['Recommended Forecast']).abs()
    
    # Head-to-Head Winner Judgment
    def judge_winner(row):
        e_base = row['Baseline Error']
        e_mom  = row['Momentum Error']
        if abs(e_base - e_mom) <= 2:
            return "Tied / Approximately Equal"
        elif e_mom < e_base:
            return "Momentum Performed Better"
        else:
            return "Baseline Performed Better"
            
    df_eval['Model Performance Comparison'] = df_eval.apply(judge_winner, axis=1)
    
    total_actual = df_eval['Actual Sales'].sum()
    total_baseline_err = df_eval['Baseline Error'].sum()
    total_momentum_err = df_eval['Momentum Error'].sum()
    total_recommended_err = df_eval['Recommended Error'].sum()
    
    wape_baseline = (total_baseline_err / total_actual) * 100.0 if total_actual > 0 else 0.0
    wape_momentum = (total_momentum_err / total_actual) * 100.0 if total_actual > 0 else 0.0
    wape_recommended = (total_recommended_err / total_actual) * 100.0 if total_actual > 0 else 0.0
    
    winner_counts = df_eval['Model Performance Comparison'].value_counts().to_dict()
    
    summary_metrics = {
        'total_actual_units': int(total_actual),
        'total_baseline_error_units': int(total_baseline_err),
        'total_momentum_error_units': int(total_momentum_err),
        'total_recommended_error_units': int(total_recommended_err),
        'wape_baseline_pct': round(wape_baseline, 2),
        'wape_momentum_pct': round(wape_momentum, 2),
        'wape_recommended_pct': round(wape_recommended, 2),
        'momentum_winner_count': winner_counts.get('Momentum Performed Better', 0),
        'baseline_winner_count': winner_counts.get('Baseline Performed Better', 0),
        'tied_winner_count': winner_counts.get('Tied / Approximately Equal', 0)
    }
    
    return df_eval, summary_metrics
