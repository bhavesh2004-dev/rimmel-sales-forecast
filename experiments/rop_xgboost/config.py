"""
Configuration for Jev + XGBoost ROP Replenishment Experiment
============================================================
NOTE: This configuration is strictly for the ROP experiment in experiments/rop_xgboost/.
It does NOT modify any production settings in config/settings.py.
"""
import os

EXP_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.abspath(os.path.join(EXP_DIR, '..', '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DB_PATH = os.path.join(DATA_DIR, 'rimmel_clean.db')  # READ-ONLY INPUT
OUTPUTS_DIR = os.path.join(EXP_DIR, 'outputs')

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── Replenishment & Inventory Parameters ─────────────────────────────────────
# Note: Supplier lead time is missing from raw client files; parameterized as standard scenarios.
DEFAULT_LEAD_TIME_DAYS = 7        # Standard replenishment transit lead time (L)
SCENARIO_LEAD_TIME_DAYS = 14      # Extended lead time scenario
TARGET_SERVICE_LEVEL = 0.95       # 95% Cycle Service Level (quantile alpha = 0.95)
MAX_COVER_DAYS_TARGET = 30        # Order-Up-To target ceiling (S)
MIN_REORDER_THRESHOLD_UNITS = 1   # Minimum replenishment trigger

# ── Temporal Windows for Walk-Forward Backtest ─────────────────────────────────
# Backtest is strictly historical; leaves production holdout (Sep 1-10) untouched.
TRAIN_START_DATE = '2025-08-01'
TRAIN_END_DATE   = '2026-06-30'   # 11 months historical training
BACKTEST_START_DATE = '2026-07-01'
BACKTEST_END_DATE   = '2026-08-31' # 62 days out-of-sample backtest simulation

# ── XGBoost Quantile Regressor Parameters ────────────────────────────────────
XGB_QUANTILE_PARAMS = {
    'objective': 'reg:quantileerror',
    'quantile_alpha': TARGET_SERVICE_LEVEL,
    'n_estimators': 120,
    'max_depth': 5,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist'
}
