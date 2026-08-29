"""
GLOBAL CONFIGURATION & SETTINGS
===============================
Defines database paths, protected holdout periods, default production horizons,
commercial tier thresholds, and business classification rules.
"""
import os

# Base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Database Paths
DB_PATH = os.path.join(BASE_DIR, 'data', 'rimmel_clean.db')
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

# Protected Historical Holdout Protocol (Strictly Isolated Gate)
HOLDOUT_TRAIN_START = '2025-08-01'
HOLDOUT_TRAIN_END   = '2026-07-20'
HOLDOUT_EVAL_START  = '2026-07-21'
HOLDOUT_EVAL_END    = '2026-07-31'
HOLDOUT_DAYS        = 11

# Production Forecast Defaults
DEFAULT_PROD_TRAIN_START = '2025-08-01'
DEFAULT_PROD_CUTOFF      = '2026-07-31'
DEFAULT_FORECAST_HORIZON_DAYS = 11
MAX_FORECAST_HORIZON_DAYS = 31

# Commercial Tier Thresholds (Annual Unit Volume)
TIER_A_MIN_UNITS = 10000
TIER_B_MIN_UNITS = 1500
TIER_C_MIN_UNITS = 200

# Volatility & Stability Thresholds
CV_STABLE_MAX = 0.85
CV_MODERATE_MAX = 1.25
CV_VOLATILE_MIN = 1.30
ZERO_DAYS_INTERMITTENT_PCT = 50.0

# Master Catalog Metadata
CATALOG_SKU_COUNT = 588
