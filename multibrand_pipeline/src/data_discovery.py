"""
Dynamic Data Discovery and Timeline Profiling Module.
Discovers actual data availability, boundaries, volume, and sufficiency per brand from MySQL.
Calculates dynamic rolling validation (T-9 to T) and forecast (T+1 to T+10) windows.
"""

import os
import logging
from datetime import datetime, date
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np
from sqlalchemy import text

logger = logging.getLogger(__name__)

def discover_brand_coverage(
    db_manager,
    horizon_days: int = 10,
    reference_date: Optional[date] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scans MySQL normalized_sales to extract exact chronological boundaries,
    transaction volumes, active selling days, and dynamic windows for all brands.
    
    Returns:
    - df_brands: Summary profile per brand with dynamic validation & forecast dates
    - df_quality: Granular diagnostics including gap warnings and sufficiency status
    """
    logger.info("Scanning MySQL normalized_sales for runtime data discovery...")
    
    with db_manager.engine.connect() as conn:
        df_norm = pd.read_sql("SELECT brand_id, display_brand_name, canonical_sku, platform_group, date, units_sold, source_table, source_row_id FROM normalized_sales", conn)
        
    if df_norm.empty:
        raise RuntimeError("MySQL normalized_sales table is empty. Ingestion must precede data discovery.")
        
    df_norm['date'] = pd.to_datetime(df_norm['date'])
    today = reference_date if reference_date else date.today()
    
    brand_rows = []
    quality_rows = []
    
    for brand_id, g in df_norm.groupby('brand_id'):
        disp_name = g['display_brand_name'].iloc[0]
        min_dt = g['date'].min()
        max_dt = g['date'].max()
        
        earliest_date = min_dt.strftime('%Y-%m-%d')
        latest_date = max_dt.strftime('%Y-%m-%d')
        
        cal_span = (max_dt - min_dt).days + 1
        active_days = g['date'].dt.date.nunique()
        tot_rows = len(g)
        tot_units = float(g['units_sold'].sum())
        distinct_skus = g['canonical_sku'].nunique()
        platforms = sorted(g['platform_group'].unique().tolist())
        
        # Calculate maximum consecutive day gap
        dates_sorted = sorted(g['date'].dt.date.unique())
        if len(dates_sorted) > 1:
            gaps = [(dates_sorted[i] - dates_sorted[i-1]).days for i in range(1, len(dates_sorted))]
            max_gap = max(gaps)
        else:
            max_gap = 0
            
        completeness = (active_days / cal_span) * 100.0 if cal_span > 0 else 0.0
        
        # Dynamic window calculations based on latest reliable date T
        # Validation horizon: exactly horizon_days ending at T
        val_end_dt = max_dt
        val_start_dt = max_dt - pd.Timedelta(days=horizon_days - 1)
        val_train_cutoff_dt = val_start_dt - pd.Timedelta(days=1)
        
        # Forward forecast horizon: exactly horizon_days starting at T + 1
        fwd_start_dt = max_dt + pd.Timedelta(days=1)
        fwd_end_dt = max_dt + pd.Timedelta(days=horizon_days)
        
        val_start = val_start_dt.strftime('%Y-%m-%d')
        val_end = val_end_dt.strftime('%Y-%m-%d')
        val_train_cutoff = val_train_cutoff_dt.strftime('%Y-%m-%d')
        fwd_start = fwd_start_dt.strftime('%Y-%m-%d')
        fwd_end = fwd_end_dt.strftime('%Y-%m-%d')
        
        # Context and Training Start boundaries
        has_pre_aug_2025 = min_dt < pd.Timestamp('2025-08-01')
        has_2025 = min_dt < pd.Timestamp('2026-01-01')
        
        context_start = '2025-01-01' if has_2025 else earliest_date
        training_start = '2025-08-01' if has_pre_aug_2025 else earliest_date
        
        # Freshness and Warning classification
        stale_days = (today - max_dt.date()).days
        is_stale = stale_days > 7
        
        warnings = []
        if is_stale:
            warnings.append(f"STALE_FEED_{stale_days}D_AGO")
        if cal_span < 30:
            warnings.append("SHORT_HISTORY_UNDER_30D")
        if max_gap > 30:
            warnings.append(f"OBSERVATION_GAP_{max_gap}D")
        if tot_units < 500:
            warnings.append("MICRO_VOLUME_UNDER_500U")
        if completeness < 40.0:
            warnings.append("INTERMITTENT_COVERAGE_UNDER_40PCT")
            
        warning_str = "; ".join(warnings) if warnings else "NONE"
        freshness_status = f"HISTORICAL_AS_OF_{latest_date}" if is_stale else "CURRENT_ACTIVE"
        
        # Data sufficiency classification for modeling
        pre_val_active_days = g[g['date'] <= val_train_cutoff_dt]['date'].dt.date.nunique()
        pre_val_units = float(g[g['date'] <= val_train_cutoff_dt]['units_sold'].sum())
        
        if pre_val_active_days >= 100 and pre_val_units >= 2000 and cal_span >= 180:
            sufficiency = "SUFFICIENT_FOR_ML"
        elif pre_val_active_days >= 30 and pre_val_units >= 500:
            sufficiency = "LIMITED_FOR_POOLED_ML"
        elif pre_val_active_days >= 7:
            sufficiency = "SHORT_HISTORY_FALLBACK_ONLY"
        else:
            sufficiency = "INSUFFICIENT_COLD_START"
            
        brand_rows.append({
            'brand_id': brand_id,
            'display_brand_name': disp_name,
            'earliest_date': earliest_date,
            'latest_date': latest_date,
            'cutoff_date_T': latest_date,
            'horizon_days': horizon_days,
            'validation_start': val_start,
            'validation_end': val_end,
            'validation_training_cutoff': val_train_cutoff,
            'forecast_start': fwd_start,
            'forecast_end': fwd_end,
            'context_start': context_start,
            'training_start': training_start,
            'total_rows': tot_rows,
            'total_units': tot_units,
            'distinct_skus': distinct_skus,
            'platforms_count': len(platforms),
            'platforms_list': ", ".join(platforms),
            'active_selling_days': active_days,
            'calendar_span_days': cal_span,
            'longest_gap_days': max_gap,
            'coverage_completeness_pct': round(completeness, 2),
            'pre_val_active_days': pre_val_active_days,
            'pre_val_units': pre_val_units,
            'data_sufficiency': sufficiency,
            'freshness_status': freshness_status,
            'warning_reason': warning_str
        })
        
        # Quality diagnostics record
        dup_count = tot_rows - g['source_row_id'].nunique()
        quality_rows.append({
            'brand_id': brand_id,
            'earliest_date': earliest_date,
            'latest_date': latest_date,
            'calendar_span_days': cal_span,
            'active_selling_days': active_days,
            'coverage_completeness_pct': round(completeness, 2),
            'longest_gap_days': max_gap,
            'missing_invalid_dates': int(g['date'].isna().sum()),
            'duplicate_source_records': dup_count,
            'total_normalized_rows': tot_rows,
            'total_physical_units': tot_units,
            'distinct_canonical_skus': distinct_skus,
            'distinct_platforms': len(platforms),
            'stale_days_vs_today': max(0, stale_days),
            'quality_flag': "PASS" if not warnings else "WARNINGS_NOTED"
        })
        
    df_brands = pd.DataFrame(brand_rows).sort_values(by='total_units', ascending=False).reset_index(drop=True)
    df_quality = pd.DataFrame(quality_rows).sort_values(by='total_physical_units', ascending=False).reset_index(drop=True)
    
    logger.info(f"Discovered coverage across {len(df_brands)} brands in MySQL.")
    return df_brands, df_quality
