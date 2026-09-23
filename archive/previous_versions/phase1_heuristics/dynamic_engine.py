"""
DYNAMIC FORECASTING ENGINE (FACADE ORCHESTRATOR)
================================================
Unified pipeline orchestrator combining preprocessing, feature engineering,
baseline, momentum, adaptive blending, confidence, risk, and explainability.

Powers the main client production pipeline using 100% of available training data
(Aug 1, 2025 to Jul 31, 2026), with zero hardcoded holdout omissions.
"""
import pandas as pd
from config.settings import DEFAULT_PROD_TRAIN_START, DEFAULT_PROD_CUTOFF, DEFAULT_FORECAST_HORIZON_DAYS
from src.preprocessing import preprocess_sales_data
from src.feature_engineering import calculate_category_momentum, extract_sku_features
from src.baseline_forecast import compute_baseline_forecast
from src.momentum_forecast import compute_momentum_forecast
from src.adaptive_forecast import compute_adaptive_forecast
from src.confidence import classify_confidence
from src.risk import classify_risk_and_status
from src.inventory_insights import compute_inventory_metrics
from src.explanations import generate_forecast_explanation

def run_dynamic_forecast(sales_dataframe, 
                         train_start=DEFAULT_PROD_TRAIN_START, 
                         train_end=DEFAULT_PROD_CUTOFF, 
                         forecast_start=None,
                         forecast_end=None,
                         forecast_days=DEFAULT_FORECAST_HORIZON_DAYS):
    """
    Orchestrates the end-to-end production forecasting pipeline for all 588 SKUs.
    Defaults to full historical training data through July 31, 2026.
    
    Args:
        sales_dataframe (pd.DataFrame): Master historical daily sales dataset.
        train_start (str or pd.Timestamp): Earliest date for model training.
        train_end (str or pd.Timestamp): Cutoff date for model training (zero data leakage).
        forecast_start (str or pd.Timestamp, optional): Start date of forecast window.
        forecast_end (str or pd.Timestamp, optional): End date of forecast window.
        forecast_days (int, optional): Number of forward forecast days if explicit dates are omitted.
        
    Returns:
        pd.DataFrame: Clean catalog forecast dataframe with all business and inventory fields.
    """
    train_end_ts = pd.to_datetime(train_end)
    
    # Calculate forecast horizon and formatted date label
    if forecast_start is not None and forecast_end is not None:
        f_start_ts = pd.to_datetime(forecast_start)
        f_end_ts   = pd.to_datetime(forecast_end)
        horizon_days = max(int((f_end_ts - f_start_ts).days) + 1, 1)
        date_label = f"{f_start_ts.strftime('%d/%m/%Y')} to {f_end_ts.strftime('%d/%m/%Y')}"
    else:
        horizon_days = max(int(forecast_days), 1)
        f_start_ts = train_end_ts + pd.Timedelta(days=1)
        f_end_ts   = train_end_ts + pd.Timedelta(days=horizon_days)
        date_label = f"{f_start_ts.strftime('%d/%m/%Y')} to {f_end_ts.strftime('%d/%m/%Y')}"
        
    # 1. Preprocess & Isolate Training Slice (Full Data)
    training_slice = preprocess_sales_data(sales_dataframe, start_date=train_start, end_date=train_end)
    
    # 2. Extract Category Momentum Signals
    category_momentum_map = calculate_category_momentum(training_slice)
    
    all_skus = sorted(sales_dataframe['sku'].unique().tolist())
    forecast_records = []
    
    for sku in all_skus:
        # 3. Extract Grounded SKU Features
        features = extract_sku_features(training_slice, sku, category_momentum_map)
        
        # 4. Generate Baseline Forecast
        baseline_units = compute_baseline_forecast(features, forecast_horizon_days=horizon_days)
        
        # 5. Generate Momentum Forecast
        momentum_units = compute_momentum_forecast(features, forecast_horizon_days=horizon_days)
        
        # 6. Generate Adaptive Blend
        recommended_units, momentum_weight = compute_adaptive_forecast(
            features, baseline_units, momentum_units, forecast_horizon_days=horizon_days
        )
        
        # 7. Classify Confidence
        confidence_level = classify_confidence(features, baseline_units, momentum_units)
        
        # 8. Classify Risk & Status
        risk_status = classify_risk_and_status(
            features, baseline_units, momentum_units, recommended_units, confidence_level
        )
        
        # 9. Compute Inventory Insights & Days of Inventory
        inventory_metrics = compute_inventory_metrics(
            features, recommended_units, forecast_horizon_days=horizon_days, risk_status=risk_status
        )
        
        # 10. Generate Data-Grounded Explanation
        explanation = generate_forecast_explanation(
            features, baseline_units, momentum_units, recommended_units, confidence_level, risk_status
        )
        
        # Format clean production record
        record = {
            'Date': date_label,
            'Product SKU': sku,
            'Category': features['category'],
            'Current Stock': features['current_stock'],
            'Actual Sales': None,  # Left blank for client actual entry
            'Baseline Prediction': baseline_units,
            'Momentum Prediction': momentum_units,
            'Recommended Forecast': recommended_units,
            'Confidence': confidence_level,
            'Risk / Status': risk_status,
            'Reason': explanation,
            'Estimated Days of Inventory': inventory_metrics['days_of_inventory_str'],
            'Inventory Health Status': inventory_metrics['inventory_health_status'],
            'Recommended Daily Demand': inventory_metrics['recommended_daily_demand'],
            'Momentum Weight': momentum_weight,
            'Forecast Horizon Days': horizon_days
        }
        forecast_records.append(record)
        
    df_output = pd.DataFrame(forecast_records)
    return df_output
