"""
GLOBAL CONFIGURATION & SETTINGS
===============================
Rimmel Multi-Platform Demand Forecasting & Inventory Planning System
Defines certified production parameters, database paths, temporal boundaries,
calibration thresholds, and LightGBM model configuration.
"""
import os

# ── Base Directory & Project Structure ────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Core Storage & Deliverable Directories
DATA_DIR      = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR  = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
DB_PATH       = os.path.join(DATA_DIR, 'rimmel_clean.db')
REPORTS_DIR   = os.path.join(BASE_DIR, 'reports')
MODELS_DIR    = os.path.join(BASE_DIR, 'models')

# ── Certified Production System (Exp6 Architecture) ───────────────────────────
PRODUCTION_MODEL_VERSION = '1.0.0-production-certified'
MODEL_NAME               = 'Exp6_LightGBM_Combined_Calibration'
CATALOG_SKU_COUNT        = 674       # Total Canonical SKUs in master catalog
PLATFORM_GROUPS          = ['Amazon', 'eBay', 'Website', 'Other']
TOTAL_ACTIVE_SERIES      = 1413      # Active SKU-platform combinations

# Temporal Boundaries (Certified Production Timeline)
PRODUCTION_TRAIN_START   = '2025-08-01'
PRODUCTION_TRAIN_END     = '2026-09-10'  # 406 calendar days
DEFAULT_PROD_CUTOFF      = '2026-09-10'  # Primary operational cutoff

# Retrospective Holdout Validation Window (Unseen Benchmark)
VALIDATION_START         = '2026-09-01'
VALIDATION_END           = '2026-09-10'
VALIDATION_DAYS          = 10

# Forward Production Planning Horizon
FORWARD_FORECAST_START   = '2026-09-11'
FORWARD_FORECAST_END     = '2026-09-20'
FORECAST_HORIZON_DAYS    = 10
MAX_FORECAST_HORIZON_DAYS = 31

# ── Certified Exp6 Post-Hoc Calibration Thresholds ────────────────────────────
CALIBRATION_ALPHA = 0.10  # Zero-demand dampening (v7 == 0 -> y_hat * 0.10)
CALIBRATION_BETA  = 0.10  # Stockout dampening (current_stock == 0 -> y_hat * 0.10)

# ── Certified LightGBM Regressor Hyperparameters ──────────────────────────────
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 150,
    'max_depth': 6,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'random_state': 42,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_jobs': -1,
    'verbose': -1
}

# ── Shared Warehouse Inventory Policy ─────────────────────────────────────────
# GOVERNANCE RULE: Physical stock is held in ONE central pool per SKU.
# NEVER sum inventory across platforms.
MIN_SAFE_COVER_DAYS = 14.0
LEAN_COVER_DAYS     = 7.0
CRITICAL_STOCKOUT_UNITS = 0

# ── Backward Compatibility Aliases (Deprecated Phase 1 Constants) ─────────────
DEFAULT_PROD_TRAIN_START = PRODUCTION_TRAIN_START
DEFAULT_FORECAST_HORIZON_DAYS = FORECAST_HORIZON_DAYS
HOLDOUT_TRAIN_START = PRODUCTION_TRAIN_START
HOLDOUT_TRAIN_END   = '2026-08-31'
HOLDOUT_EVAL_START  = VALIDATION_START
HOLDOUT_EVAL_END    = VALIDATION_END
HOLDOUT_DAYS        = VALIDATION_DAYS
