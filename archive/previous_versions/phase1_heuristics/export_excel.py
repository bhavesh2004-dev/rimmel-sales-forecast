"""
EXCEL EXPORT MODULE (PHASE 1.1)
===============================
Exports the two primary Excel workbooks:
1. rimmel_full_normalized.xlsx (contains normalized_transactions + catalog masters)
2. rimmel_ml_ready_normalized.xlsx (contains daily_sku_platform_obs + normalized_transactions + catalog masters)
"""
import os
import pandas as pd

DATA_DICTIONARY_ROWS = [
    {
        'column_name': 'source_row_id',
        'data_type': 'INTEGER',
        'business_description': 'Sequential 1-indexed identifier matching exact raw Excel row position (1 to 101,085).',
        'source_column': 'Implicit Row Index',
        'transformation': '1-indexed sequential integer assignment during ingestion.',
        'allowed_values': '1 to 101,085',
        'nullable': 'NO',
        'is_derived': 'YES',
        'is_ml_safe': 'NO (Lineage only)',
        'platform_applicability': 'All Platforms',
        'leakage_considerations': 'Safe (Metadata / Lineage only).'
    },
    {
        'column_name': 'date',
        'data_type': 'DATE (YYYY-MM-DD)',
        'business_description': 'Calendar date of the sales observation or transaction.',
        'source_column': 'date',
        'transformation': 'Standardized to ISO-8601 date string YYYY-MM-DD.',
        'allowed_values': '2025-01-01 to 2026-09-10 (Full) / 2025-08-01 to 2026-09-10 (ML)',
        'nullable': 'NO',
        'is_derived': 'NO',
        'is_ml_safe': 'YES',
        'platform_applicability': 'All Platforms',
        'leakage_considerations': 'Chronological index. All rolling features must strictly look backward (<= date).'
    },
    {
        'column_name': 'platform_group',
        'data_type': 'STRING',
        'business_description': 'High-level e-commerce platform rollup for modular forecasting.',
        'source_column': 'channel',
        'transformation': 'Mapped via CHANNEL_TO_PLATFORM_MAP.',
        'allowed_values': 'Amazon, eBay, Website, Other',
        'nullable': 'NO',
        'is_derived': 'YES',
        'is_ml_safe': 'YES',
        'platform_applicability': 'All Platforms',
        'leakage_considerations': 'Safe (Static platform mapping).'
    },
    {
        'column_name': 'channel',
        'data_type': 'STRING',
        'business_description': 'Exact merchant storefront / channel feed name from client raw data.',
        'source_column': 'channel',
        'transformation': 'Preserved verbatim from raw Excel.',
        'allowed_values': '8 distinct channel names',
        'nullable': 'NO',
        'is_derived': 'NO',
        'is_ml_safe': 'YES (Categorical)',
        'platform_applicability': 'All Platforms',
        'leakage_considerations': 'Safe.'
    },
    {
        'column_name': 'channel_type',
        'data_type': 'STRING',
        'business_description': 'Granular marketplace or fulfillment classification.',
        'source_column': 'channel',
        'transformation': 'Mapped via CHANNEL_TO_PLATFORM_MAP.',
        'allowed_values': 'Amazon, eBay, Website, TikTok, Retail Store',
        'nullable': 'NO',
        'is_derived': 'YES',
        'is_ml_safe': 'YES (Categorical)',
        'platform_applicability': 'All Platforms',
        'leakage_considerations': 'Safe.'
    },
    {
        'column_name': 'raw_sku',
        'data_type': 'STRING',
        'business_description': 'Original product SKU identifier logged verbatim in client feed.',
        'source_column': 'sku',
        'transformation': 'Preserved verbatim (trimmed).',
        'allowed_values': 'Alphanumeric SKU strings',
        'nullable': 'NO',
        'is_derived': 'NO',
        'is_ml_safe': 'YES',
        'platform_applicability': 'All Platforms',
        'leakage_considerations': 'Safe.'
    },
    {
        'column_name': 'canonical_sku',
        'data_type': 'STRING',
        'business_description': 'Master product SKU standardizing variant and multi-pack listings under base identity.',
        'source_column': 'actual_sku, sku',
        'transformation': 'actual_sku if valid and non-null, else raw_sku.',
        'allowed_values': 'Alphanumeric SKU strings',
        'nullable': 'NO',
        'is_derived': 'YES',
        'is_ml_safe': 'YES',
        'platform_applicability': 'All Platforms',
        'leakage_considerations': 'Safe static product identity.'
    },
    {
        'column_name': 'canonical_sku_source',
        'data_type': 'STRING',
        'business_description': 'Tracks whether canonical_sku came from ACTUAL_SKU mapping or RAW_SKU_FALLBACK.',
        'source_column': 'actual_sku',
        'transformation': "ACTUAL_SKU if actual_sku is not null else RAW_SKU_FALLBACK.",
        'allowed_values': 'ACTUAL_SKU, RAW_SKU_FALLBACK',
        'nullable': 'NO',
        'is_derived': 'YES',
        'is_ml_safe': 'NO (Metadata)',
        'platform_applicability': 'All Platforms',
        'leakage_considerations': 'Safe.'
    },
    {
        'column_name': 'raw_parent_id',
        'data_type': 'STRING',
        'business_description': 'Original parent variation family string from raw Excel (may be null).',
        'source_column': 'parent_id',
        'transformation': 'Preserved verbatim (trimmed).',
        'allowed_values': 'Parent family string or Null',
        'nullable': 'YES',
        'is_derived': 'NO',
        'is_ml_safe': 'YES',
        'platform_applicability': 'All Platforms',
        'leakage_considerations': 'Safe.'
    },
    {
        'column_name': 'resolved_parent_id',
        'data_type': 'STRING',
        'business_description': 'Resolved parent family identifier (falls back to raw_sku when raw_parent_id is null).',
        'source_column': 'parent_id, sku',
        'transformation': 'raw_parent_id if not null else raw_sku.',
        'allowed_values': 'Parent family string',
        'nullable': 'NO',
        'is_derived': 'YES',
        'is_ml_safe': 'YES',
        'platform_applicability': 'All Platforms',
        'leakage_considerations': 'Safe.'
    },
    {
        'column_name': 'parent_resolution_method',
        'data_type': 'STRING',
        'business_description': 'Method used to determine resolved_parent_id.',
        'source_column': 'parent_id',
        'transformation': 'FROM_RAW_PARENT_ID if raw_parent_id is not null else RAW_SKU_FALLBACK.',
        'allowed_values': 'FROM_RAW_PARENT_ID, RAW_SKU_FALLBACK',
        'nullable': 'NO',
        'is_derived': 'YES',
        'is_ml_safe': 'NO (Metadata)',
        'platform_applicability': 'All Platforms',
        'leakage_considerations': 'Safe.'
    },
    {
        'column_name': 'raw_listing_id',
        'data_type': 'STRING',
        'business_description': 'Original marketplace listing identifier from raw Excel (null on legacy TikTok).',
        'source_column': 'listing_id',
        'transformation': 'Preserved verbatim (trimmed).',
        'allowed_values': 'Listing ID or Null',
        'nullable': 'YES',
        'is_derived': 'NO',
        'is_ml_safe': 'YES',
        'platform_applicability': 'All Platforms',
        'leakage_considerations': 'Safe.'
    },
    {
        'column_name': 'resolved_listing_id',
        'data_type': 'STRING',
        'business_description': 'Universal platform listing identifier (ASIN, eBay Item ID, Web ID). Falls back to raw_sku.',
        'source_column': 'listing_id, sku',
        'transformation': 'raw_listing_id if not null else raw_sku.',
        'allowed_values': 'Listing ID string',
        'nullable': 'NO',
        'is_derived': 'YES',
        'is_ml_safe': 'YES',
        'platform_applicability': 'All Platforms',
        'leakage_considerations': 'Safe.'
    },
    {
        'column_name': 'listing_resolution_method',
        'data_type': 'STRING',
        'business_description': 'Method used to determine resolved_listing_id.',
        'source_column': 'listing_id',
        'transformation': 'FROM_RAW_LISTING_ID if raw_listing_id is not null else RAW_SKU_FALLBACK.',
        'allowed_values': 'FROM_RAW_LISTING_ID, RAW_SKU_FALLBACK',
        'nullable': 'NO',
        'is_derived': 'YES',
        'is_ml_safe': 'NO (Metadata)',
        'platform_applicability': 'All Platforms',
        'leakage_considerations': 'Safe.'
    },
    {
        'column_name': 'child_asin',
        'data_type': 'STRING',
        'business_description': 'Amazon Child ASIN. Populated ONLY for Amazon records; STRICTLY NULL for eBay, Website, Other.',
        'source_column': 'child_asin',
        'transformation': 'Preserved for Amazon; set to NULL for eBay, Website, and Other.',
        'allowed_values': '10-char Amazon ASIN string (Amazon only) or Null',
        'nullable': 'YES (Null on all non-Amazon platforms)',
        'is_derived': 'YES (Masked for non-Amazon)',
        'is_ml_safe': 'YES (Amazon Module Only)',
        'platform_applicability': 'Amazon Only',
        'leakage_considerations': 'Safe.'
    },
    {
        'column_name': 'category',
        'data_type': 'STRING',
        'business_description': 'Cosmetic product category taxonomy.',
        'source_column': 'category',
        'transformation': 'Preserved verbatim (trimmed).',
        'allowed_values': 'Eyeliner, Mascara, Lipstick, etc.',
        'nullable': 'NO',
        'is_derived': 'NO',
        'is_ml_safe': 'YES',
        'platform_applicability': 'All Platforms',
        'leakage_considerations': 'Safe.'
    },
    {
        'column_name': 'pack_multiplier',
        'data_type': 'INTEGER',
        'business_description': 'Number of physical units bundled in a single sellable pack listing.',
        'source_column': 'pack_multiplier',
        'transformation': 'Preserved verbatim as integer.',
        'allowed_values': '1, 2, 3, 5, 6',
        'nullable': 'NO',
        'is_derived': 'NO',
        'is_ml_safe': 'YES',
        'platform_applicability': 'All Platforms',
        'leakage_considerations': 'Safe.'
    },
    {
        'column_name': 'observed_units_sold',
        'data_type': 'INTEGER / FLOAT',
        'business_description': 'Physical units sold recorded on date for this entity (or 0 for OBSERVED_ZERO).',
        'source_column': 'units_sold',
        'transformation': 'Preserved verbatim from raw sales.',
        'allowed_values': '>= 0',
        'nullable': 'YES (Null when entity is unobserved/inactive)',
        'is_derived': 'NO',
        'is_ml_safe': 'YES (Target Variable)',
        'platform_applicability': 'All Platforms',
        'leakage_considerations': 'Target variable.'
    },
    {
        'column_name': 'orders_count',
        'data_type': 'INTEGER / FLOAT',
        'business_description': 'Number of distinct customer checkout orders.',
        'source_column': 'orders_count',
        'transformation': 'Preserved verbatim from raw data.',
        'allowed_values': '>= 0',
        'nullable': 'YES (Null when entity is unobserved/inactive)',
        'is_derived': 'NO',
        'is_ml_safe': 'YES',
        'platform_applicability': 'All Platforms',
        'leakage_considerations': 'Safe.'
    },
    {
        'column_name': 'selling_price',
        'data_type': 'FLOAT',
        'business_description': 'Recorded selling price per unit/pack in GBP (£).',
        'source_column': 'selling_price',
        'transformation': 'Preserved verbatim.',
        'allowed_values': '£0.00 to £48.99',
        'nullable': 'YES (Null on non-transaction days)',
        'is_derived': 'NO',
        'is_ml_safe': 'YES',
        'platform_applicability': 'All Platforms',
        'leakage_considerations': 'Exogenous signal.'
    },
    {
        'column_name': 'current_stock',
        'data_type': 'FLOAT',
        'business_description': 'Shared central warehouse inventory on date. Never summed across platforms.',
        'source_column': 'current_stock',
        'transformation': 'Preserved as shared central inventory pool.',
        'allowed_values': '>= 0.0 or Null',
        'nullable': 'YES (Null before Aug 1, 2025)',
        'is_derived': 'NO',
        'is_ml_safe': 'YES (Post-August only)',
        'platform_applicability': 'All Platforms (Shared Warehouse)',
        'leakage_considerations': 'Inventory signal. Must look backward only.'
    },
    {
        'column_name': 'has_inventory_signal',
        'data_type': 'INTEGER (0/1)',
        'business_description': 'Binary indicator whether warehouse stock was monitored on that date.',
        'source_column': 'current_stock, date',
        'transformation': '1 if date >= 2025-08-01 and current_stock is known, else 0.',
        'allowed_values': '0 or 1',
        'nullable': 'NO',
        'is_derived': 'YES',
        'is_ml_safe': 'YES',
        'platform_applicability': 'All Platforms',
        'leakage_considerations': 'Safe.'
    },
    {
        'column_name': 'stockout_flag',
        'data_type': 'INTEGER (0/1)',
        'business_description': 'Binary indicator whether shared warehouse stock was 0 (demand censored).',
        'source_column': 'current_stock, has_inventory_signal',
        'transformation': '1 if has_inventory_signal == 1 and current_stock == 0, else 0.',
        'allowed_values': '0 or 1',
        'nullable': 'NO',
        'is_derived': 'YES',
        'is_ml_safe': 'YES',
        'platform_applicability': 'All Platforms',
        'leakage_considerations': 'Safe.'
    },
    {
        'column_name': 'observation_state',
        'data_type': 'STRING',
        'business_description': 'Classification in the 9-state observation framework.',
        'source_column': 'units_sold, stockout_flag, launch_date, date',
        'transformation': 'OBSERVED_SALE, OBSERVED_ZERO, STOCKOUT_DEMAND_CENSORED, PRE_LAUNCH, etc.',
        'allowed_values': '9 observation states',
        'nullable': 'NO',
        'is_derived': 'YES',
        'is_ml_safe': 'YES',
        'platform_applicability': 'All Platforms',
        'leakage_considerations': 'Safe.'
    },
    {
        'column_name': 'model_units_sold',
        'data_type': 'FLOAT',
        'business_description': 'Demand representation for downstream ML models.',
        'source_column': 'units_sold, observation_state',
        'transformation': 'Exact units for sales, 0.0 for OBSERVED_ZERO, NaN for inactive/censored.',
        'allowed_values': '>= 0.0 or NaN',
        'nullable': 'YES',
        'is_derived': 'YES',
        'is_ml_safe': 'YES (Target Variable)',
        'platform_applicability': 'All Platforms',
        'leakage_considerations': 'Target variable.'
    },
    {
        'column_name': 'treatment_method',
        'data_type': 'STRING',
        'business_description': 'Demand representation treatment applied.',
        'source_column': 'Observation State',
        'transformation': 'AS_OBSERVED, ZERO, CENSORED_DEMAND, NOT_APPLICABLE, PRESERVE_UNCERTAINTY, LEAVE_UNAVAILABLE.',
        'allowed_values': 'Categorical policy strings',
        'nullable': 'NO',
        'is_derived': 'YES',
        'is_ml_safe': 'NO (Metadata)',
        'platform_applicability': 'All Platforms',
        'leakage_considerations': 'Safe.'
    },
    {
        'column_name': 'treatment_confidence',
        'data_type': 'FLOAT',
        'business_description': 'Confidence score of the observation state / demand representation.',
        'source_column': 'Observation State',
        'transformation': '1.0 for observed sales/zeros, 0.5 for censored, 0.0 for unobserved/inactive.',
        'allowed_values': '0.0 to 1.0',
        'nullable': 'NO',
        'is_derived': 'YES',
        'is_ml_safe': 'YES (Sample Weight)',
        'platform_applicability': 'All Platforms',
        'leakage_considerations': 'Safe.'
    }
]

def export_full_workbook(
    filepath: str,
    df_transactions: pd.DataFrame,
    df_sku_master: pd.DataFrame,
    df_platform_mapping: pd.DataFrame,
    df_observation_states: pd.DataFrame,
    df_data_quality: pd.DataFrame,
    df_reconciliation: pd.DataFrame,
    df_transformation_log: pd.DataFrame
):
    """Exports rimmel_full_normalized.xlsx."""
    print(f'[EXCEL EXPORT] Writing Full Workbook: {filepath} ({len(df_transactions):,} transaction rows)...')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df_dict = pd.DataFrame(DATA_DICTIONARY_ROWS)
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        df_transactions.to_excel(writer, sheet_name='normalized_transactions', index=False)
        df_sku_master.to_excel(writer, sheet_name='sku_master', index=False)
        df_platform_mapping.to_excel(writer, sheet_name='platform_mapping', index=False)
        df_observation_states.to_excel(writer, sheet_name='observation_states', index=False)
        df_data_quality.to_excel(writer, sheet_name='data_quality_issues', index=False)
        df_reconciliation.to_excel(writer, sheet_name='reconciliation', index=False)
        df_transformation_log.to_excel(writer, sheet_name='transformation_log', index=False)
        df_dict.to_excel(writer, sheet_name='data_dictionary', index=False)
    print(f'[EXCEL EXPORT] Successfully created: {filepath} ({os.path.getsize(filepath):,} bytes)')

def export_ml_ready_workbook(
    filepath: str,
    df_daily_obs: pd.DataFrame,
    df_transactions_ml: pd.DataFrame,
    df_sku_master: pd.DataFrame,
    df_platform_mapping: pd.DataFrame,
    df_observation_states: pd.DataFrame,
    df_data_quality: pd.DataFrame,
    df_reconciliation: pd.DataFrame,
    df_transformation_log: pd.DataFrame
):
    """Exports rimmel_ml_ready_normalized.xlsx with true Daily Observation layer."""
    print(f'[EXCEL EXPORT] Writing ML-Ready Workbook: {filepath} ({len(df_daily_obs):,} daily obs rows, {len(df_transactions_ml):,} transaction rows)...')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df_dict = pd.DataFrame(DATA_DICTIONARY_ROWS)
    
    meta_rows = [
        {'Parameter': 'PROJECT_MODULE', 'Value': 'Rimmel Brand Multi-Platform Sales Forecasting (ML-Eligible Dataset)'},
        {'Parameter': 'MODEL_ELIGIBLE_START', 'Value': '2025-08-01'},
        {'Parameter': 'MODEL_ELIGIBLE_END', 'Value': '2026-09-10'},
        {'Parameter': 'DAILY_OBSERVATION_GRAIN', 'Value': 'DATE x PLATFORM x CANONICAL_SKU (573,678 rows across 406 calendar days)'},
        {'Parameter': 'TOTAL_ML_TRANSACTION_ROWS', 'Value': f'{len(df_transactions_ml):,}'},
        {'Parameter': 'TOTAL_ML_UNITS_SOLD', 'Value': f'{int(df_transactions_ml["observed_units_sold"].sum()):,}'},
        {'Parameter': 'TOTAL_ML_ORDERS', 'Value': f'{int(df_transactions_ml["orders_count"].sum()):,}'},
        {'Parameter': 'UNIQUE_ACTIVE_CANONICAL_SKUS', 'Value': f'{df_daily_obs["canonical_sku"].nunique()}'},
        {'Parameter': 'NON_AMAZON_CHILD_ASIN_COUNT', 'Value': '0 (STRICTLY NULL on eBay, Website, and Other)'},
        {'Parameter': 'INVENTORY_TRACKING_RATIONALE', 'Value': 'Pre-August 2025 data is retained in the Full Normalized Dataset for historical analysis, but is excluded from this ML-eligible dataset because current_stock is unavailable/empty (0% populated) prior to 2025-08-01. The current_stock signal is active and reliable (82.5% to 98.5% populated) from 2025-08-01 onwards.'},
        {'Parameter': 'SHARED_WAREHOUSE_NOTE', 'Value': 'current_stock represents shared central warehouse inventory across all channels. It is NEVER summed across Amazon, eBay, and Website.'},
        {'Parameter': 'DATA_PREPARATION_NOTE', 'Value': 'Phase 1.1 Data Engineering Layer only. Zero forecasting formulas, weights, or risk thresholds have been modified.'}
    ]
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        pd.DataFrame(meta_rows).to_excel(writer, sheet_name='ml_eligibility_metadata', index=False)
        df_daily_obs.to_excel(writer, sheet_name='daily_sku_platform_obs', index=False)
        df_transactions_ml.to_excel(writer, sheet_name='normalized_transactions', index=False)
        df_sku_master.to_excel(writer, sheet_name='sku_master', index=False)
        df_platform_mapping.to_excel(writer, sheet_name='platform_mapping', index=False)
        df_observation_states.to_excel(writer, sheet_name='observation_states', index=False)
        df_data_quality.to_excel(writer, sheet_name='data_quality_issues', index=False)
        df_reconciliation.to_excel(writer, sheet_name='reconciliation', index=False)
        df_transformation_log.to_excel(writer, sheet_name='transformation_log', index=False)
        df_dict.to_excel(writer, sheet_name='data_dictionary', index=False)
    print(f'[EXCEL EXPORT] Successfully created: {filepath} ({os.path.getsize(filepath):,} bytes)')
