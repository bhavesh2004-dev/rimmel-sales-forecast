"""
Rimmel Sales Forecasting & Decision Support System (src package)
"""
from src.data_loader import load_sales_data
from src.preprocessing import preprocess_sales_data
from src.feature_engineering import extract_sku_features, calculate_category_momentum
from src.baseline_forecast import compute_baseline_forecast
from src.momentum_forecast import compute_momentum_forecast
from src.adaptive_forecast import compute_adaptive_forecast
from src.confidence import classify_confidence
from src.risk import classify_risk_and_status
from src.inventory_insights import compute_inventory_metrics
from src.explanations import generate_forecast_explanation
from src.validation import evaluate_holdout_performance
from src.report_generator import generate_client_excel_report
from src.dynamic_engine import run_dynamic_forecast
