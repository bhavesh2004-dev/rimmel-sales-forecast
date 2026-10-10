#!/usr/bin/env python
"""
Dynamic Multi-Brand Demand Forecasting Master Pipeline Runner.
Principle: Brand as Data. Fully automated runtime discovery of data boundaries,
leak-free multi-day holdout validation, intelligent model routing, recursive forward forecasting,
single Excel validation reporting, and canonical MySQL forecast persistence.
"""

import os
import sys
import argparse
import logging
import yaml
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

# Ensure root workspace and multibrand_pipeline are in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from multibrand_pipeline.src.db_manager import DBManager
from multibrand_pipeline.src.data_discovery import discover_brand_coverage
from multibrand_pipeline.src.feature_profiles import FeatureProfiler
from multibrand_pipeline.src.model_router import ModelRouter
from multibrand_pipeline.src.dynamic_engine import DynamicForecastingEngine
from multibrand_pipeline.src.validation_reporter import generate_validation_workbook

def setup_logger(log_level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def load_yaml_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_dynamic_pipeline(
    config_path: Optional[str] = None,
    horizon_days: int = 10
) -> Dict[str, Any]:
    logger = logging.getLogger("run_dynamic_pipeline")
    logger.info("================================================================================")
    logger.info("STARTING DYNAMIC MULTI-BRAND DEMAND FORECASTING PIPELINE (PHASE 3)")
    logger.info("================================================================================")
    
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"RUN-DYN-{run_timestamp}"
    logger.info(f"Initialized Pipeline Execution Run: {run_id}")
    
    # 1. Load Configuration
    if config_path is None:
        config_path = os.path.join(CURRENT_DIR, "config", "dynamic_pipeline_config.yaml")
    config = load_yaml_config(config_path)
    
    schema_path = config['paths']['feature_schema_path']
    if not os.path.isabs(schema_path):
        schema_path = os.path.join(PROJECT_ROOT, schema_path)
        
    approved_model_path = config['paths']['approved_model_path']
    if not os.path.isabs(approved_model_path):
        approved_model_path = os.path.join(PROJECT_ROOT, approved_model_path)
        
    report_dir = config['paths']['excel_report_dir']
    if not os.path.isabs(report_dir):
        report_dir = os.path.join(PROJECT_ROOT, report_dir)
    os.makedirs(report_dir, exist_ok=True)
    excel_report_path = os.path.join(report_dir, f"multibrand_validation_{run_timestamp}.xlsx")
    
    # 2. Database Connection
    logger.info("\n[STAGE 1/6] Validating MySQL connection and runtime data availability...")
    db = DBManager()
    conn_info = db.test_connection()
    logger.info(f"  Connected to MySQL: {conn_info['database']} on {conn_info['host']}")
    
    # 3. Dynamic Data Discovery
    df_brands, df_quality = discover_brand_coverage(db, horizon_days=horizon_days)
    logger.info(f"  Discovered {len(df_brands)} brands in MySQL normalized_sales.")
    
    # Extract normalized sales for modeling
    with db.engine.connect() as conn:
        df_norm = pd.read_sql("SELECT * FROM normalized_sales", conn)
    df_norm['date'] = pd.to_datetime(df_norm['date']).dt.strftime('%Y-%m-%d')
    logger.info(f"  Loaded {len(df_norm):,} normalized transaction records across {df_norm['brand_id'].nunique()} brands.")
    
    # 4. Feature Profiling and Model Routing
    logger.info("\n[STAGE 2/6] Profiling feature availability and routing model strategies...")
    profiler = FeatureProfiler(schema_path)
    router = ModelRouter(config)
    
    routing_decisions = {}
    feature_tiers = []
    
    for _, b_prof in df_brands.iterrows():
        b_id = b_prof['brand_id']
        span_days = b_prof['calendar_span_days']
        
        f_tier = profiler.determine_feature_tier(span_days)
        f_tier['brand_id'] = b_id
        f_tier['display_brand_name'] = b_prof['display_brand_name']
        feature_tiers.append(f_tier)
        
        r_decision = router.route_brand(b_prof.to_dict())
        routing_decisions[b_id] = r_decision
        logger.info(f"  Brand {b_prof['display_brand_name']}: Val Strategy = {r_decision['validation_strategy']} | Fwd Strategy = {r_decision['forward_strategy']} | Confidence = {r_decision['confidence_level']}")

    # 5. Fixed-Origin Recursive Validation (Leak-Free)
    logger.info("\n[STAGE 3/6] Running leak-free fixed-origin holdout validation (10 calendar days)...")
    engine = DynamicForecastingEngine(config, schema_path)
    df_val_summary, df_val_detail, df_sku_metrics = engine.run_dynamic_validation(
        df_norm=df_norm,
        brand_profiles=df_brands,
        routing_decisions=routing_decisions
    )
    
    # 6. Single Excel Validation Workbook Generation
    logger.info("\n[STAGE 4/6] Exporting official multi-sheet validation workbook...")
    saved_excel = generate_validation_workbook(
        run_id=run_id,
        output_path=excel_report_path,
        df_val_summary=df_val_summary,
        df_val_detail=df_val_detail,
        df_sku_metrics=df_sku_metrics,
        df_quality=df_quality,
        feature_tiers=feature_tiers,
        metadata={"run_id": run_id, "timestamp": run_timestamp}
    )
    logger.info(f"  Validation Workbook Certified: {saved_excel}")
    
    # 7. Recursive Forward Operational Forecasting (T+1 to T+10)
    logger.info("\n[STAGE 5/6] Generating recursive 10-day forward operational forecast...")
    df_forecast = engine.run_forward_operational_forecast(
        df_norm=df_norm,
        brand_profiles=df_brands,
        routing_decisions=routing_decisions,
        approved_model_path=approved_model_path,
        run_id=run_id
    )
    
    # 8. Canonical MySQL Forecast Output Persistence
    logger.info("\n[STAGE 6/6] Persisting final forward forecast into MySQL forecast_results...")
    saved_count = db.save_forecast_results(df_forecast)
    logger.info(f"  Successfully inserted {saved_count:,} records into canonical table 'forecast_results'.")
    
    # 9. Pipeline Completion Summary
    logger.info("================================================================================")
    logger.info("DYNAMIC MULTI-BRAND PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Excel Report: {saved_excel}")
    logger.info(f"MySQL Table: forecast_results ({saved_count:,} records written)")
    logger.info("================================================================================")
    
    return {
        "run_id": run_id,
        "excel_report_path": saved_excel,
        "mysql_table": "forecast_results",
        "records_written": saved_count,
        "df_val_summary": df_val_summary,
        "df_brands": df_brands,
        "df_forecast": df_forecast
    }

if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser(description="Dynamic Multi-Brand Demand Forecasting Master Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to dynamic pipeline configuration YAML")
    parser.add_argument("--horizon", type=int, default=10, help="Forecast horizon days (default: 10)")
    args = parser.parse_args()
    
    results = run_dynamic_pipeline(config_path=args.config, horizon_days=args.horizon)
    print("\n=== PIPELINE RUN SUMMARY ===")
    print(results['df_val_summary'][['display_brand_name', 'data_cutoff_T', 'validation_start', 'validation_end', 'model_strategy', 'actual_units_10d', 'predicted_units_10d', 'wape_10d_sku_level', 'portfolio_net_bias_pct', 'confidence_level']].to_string(index=False))
