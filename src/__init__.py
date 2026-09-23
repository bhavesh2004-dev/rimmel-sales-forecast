"""
Rimmel Multi-Platform Demand Forecasting & Inventory Planning System
=====================================================================
Package entrypoint exposing the certified production forecasting engine (Exp6),
data normalization pipelines, and reporting utilities.
"""

# Core Certified Production Engine & Reporting
from src.generate_client_reports import main as generate_production_reports
from src.final_production_system import run_production_system

# Core Data Pipeline Modules
from src.data_ingestion import load_raw_dataset, compute_file_hash
from src.platform_mapping import apply_platform_mapping
from src.sku_mapping import apply_sku_hierarchy, build_sku_master
from src.normalization import normalize_transactions, build_daily_sku_platform_layer

__version__ = "1.0.0-production-certified"
__model__   = "Exp6_LightGBM_Regressor"
