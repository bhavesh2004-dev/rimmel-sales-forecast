"""
Brand & Product Behavior Forensics Module.
Performs rigorous demand pattern classification (Syntetos-Boylan ADI & CV² classification),
computes SKU-level velocity, zero-demand ratios, and generates comprehensive brand behavioral profiles.
Updates sku_master with behavioral attributes.
"""

import os
import sys
import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from multibrand_pipeline.src.db_manager import DBManager

def classify_demand_pattern(adi: float, cv2: float, total_units: float, history_days: int) -> str:
    """
    Standard Syntetos-Boylan demand classification:
    - Thresholds: ADI cutoff = 1.32, CV² cutoff = 0.49
    - Insufficient History: history_days < 30
    - Low Demand: total_units < 10 or (history_days >= 30 and (total_units / history_days) < 0.05)
    - Smooth: ADI < 1.32 and CV² < 0.49
    - Intermittent: ADI >= 1.32 and CV² < 0.49
    - Erratic: ADI < 1.32 and CV² >= 0.49
    - Lumpy: ADI >= 1.32 and CV² >= 0.49
    """
    if history_days < 30:
        return "Insufficient History"
    if total_units < 10 or (total_units / max(history_days, 1)) < 0.05:
        return "Low Demand"
    if adi < 1.32 and cv2 < 0.49:
        return "Smooth"
    elif adi >= 1.32 and cv2 < 0.49:
        return "Intermittent"
    elif adi < 1.32 and cv2 >= 0.49:
        return "Erratic"
    else:
        return "Lumpy"

def analyze_behavior():
    db = DBManager()
    logger.info("Extracting normalized sales for behavioral forensic analysis...")
    
    with db.engine.connect() as conn:
        df = pd.read_sql("""
            SELECT date, brand_id, display_brand_name, canonical_sku, platform_group, units_sold, selling_price, category
            FROM normalized_sales
        """, conn)

    df['date'] = pd.to_datetime(df['date'])
    max_date = df['date'].max()
    min_date = df['date'].min()
    logger.info(f"Loaded {len(df):,} sales rows spanning {min_date.date()} to {max_date.date()}.")

    # 1. Brand Profile Analysis
    brand_profiles = []
    for b_id, g in df.groupby('brand_id'):
        d_name = g['display_brand_name'].iloc[0]
        skus = g['canonical_sku'].nunique()
        tot_units = g['units_sold'].sum()
        b_min_date = g['date'].min()
        b_max_date = g['date'].max()
        days_span = (b_max_date - b_min_date).days + 1
        avg_units_per_day = tot_units / max(days_span, 1)
        
        # Channel share
        chan_units = g.groupby('platform_group')['units_sold'].sum()
        amz_share = (chan_units.get('Amazon', 0) / max(tot_units, 1)) * 100.0
        ebay_share = (chan_units.get('eBay', 0) / max(tot_units, 1)) * 100.0
        web_share = (chan_units.get('Website', 0) / max(tot_units, 1)) * 100.0
        oth_share = (chan_units.get('Other', 0) / max(tot_units, 1)) * 100.0
        
        # Top category
        top_cats = g['category'].value_counts().head(2).index.tolist()
        top_cat_str = ", ".join(top_cats)
        
        brand_profiles.append({
            'brand_id': b_id,
            'display_brand_name': d_name,
            'skus': skus,
            'total_units': int(tot_units),
            'min_date': b_min_date.strftime('%Y-%m-%d'),
            'max_date': b_max_date.strftime('%Y-%m-%d'),
            'days_span': days_span,
            'avg_daily_volume': round(avg_units_per_day, 1),
            'amz_pct': round(amz_share, 1),
            'ebay_pct': round(ebay_share, 1),
            'web_pct': round(web_share, 1),
            'oth_pct': round(oth_share, 1),
            'top_categories': top_cat_str
        })
    df_brand_summary = pd.DataFrame(brand_profiles).sort_values('total_units', ascending=False)
    logger.info("Brand Profiles Summary:\n" + df_brand_summary[['display_brand_name', 'skus', 'total_units', 'days_span', 'avg_daily_volume']].to_string())

    # 2. SKU-Level Diagnostics & Croston / Syntetos-Boylan Metrics
    logger.info("Computing SKU-level demand statistics (ADI, CV², velocity over 7/14/30/60/90 days)...")
    sku_diagnostics = []

    # Daily aggregation per SKU
    daily_sku = df.groupby(['canonical_sku', 'date'])['units_sold'].sum().reset_index()

    for sku, g_sku in daily_sku.groupby('canonical_sku'):
        sku_brand = df[df['canonical_sku'] == sku]['brand_id'].iloc[0]
        sku_cat = df[df['canonical_sku'] == sku]['category'].iloc[0]
        
        # Build complete daily time series for active window
        s_min = g_sku['date'].min()
        s_max = max_date  # evaluate up to latest global date
        all_dates = pd.date_range(s_min, s_max, freq='D')
        
        g_reindexed = g_sku.set_index('date')['units_sold'].reindex(all_dates, fill_value=0).reset_index()
        g_reindexed.columns = ['date', 'units']
        
        tot_days = len(g_reindexed)
        tot_units = g_reindexed['units'].sum()
        non_zero_days = (g_reindexed['units'] > 0).sum()
        zero_days = tot_days - non_zero_days
        zero_demand_ratio = zero_days / max(tot_days, 1)
        
        # ADI (Average Demand Interval = total periods / non-zero demand periods)
        adi = tot_days / max(non_zero_days, 1) if non_zero_days > 0 else 999.0
        
        # Non-zero demand values for CV²
        non_zero_vals = g_reindexed[g_reindexed['units'] > 0]['units']
        if len(non_zero_vals) > 1:
            mean_nz = non_zero_vals.mean()
            std_nz = non_zero_vals.std()
            cv = std_nz / mean_nz if mean_nz > 0 else 0.0
            cv2 = cv ** 2
        else:
            cv2 = 0.0
            
        pattern = classify_demand_pattern(adi, cv2, tot_units, tot_days)
        
        # Rolling velocities (trailing from max_date)
        v7 = g_reindexed[g_reindexed['date'] > (max_date - pd.Timedelta(days=7))]['units'].sum() / 7.0
        v14 = g_reindexed[g_reindexed['date'] > (max_date - pd.Timedelta(days=14))]['units'].sum() / 14.0
        v30 = g_reindexed[g_reindexed['date'] > (max_date - pd.Timedelta(days=30))]['units'].sum() / 30.0
        v60 = g_reindexed[g_reindexed['date'] > (max_date - pd.Timedelta(days=60))]['units'].sum() / 60.0
        v90 = g_reindexed[g_reindexed['date'] > (max_date - pd.Timedelta(days=90))]['units'].sum() / 90.0

        sku_diagnostics.append({
            'canonical_sku': sku,
            'brand_id': sku_brand,
            'category': sku_cat,
            'history_days': tot_days,
            'selling_days': non_zero_days,
            'total_units': int(tot_units),
            'zero_demand_ratio': round(zero_demand_ratio, 4),
            'adi': round(adi, 4),
            'cv2': round(cv2, 4),
            'behavior_class': pattern,
            'v7': round(v7, 2),
            'v14': round(v14, 2),
            'v30': round(v30, 2),
            'v60': round(v60, 2),
            'v90': round(v90, 2)
        })

    df_sku_diag = pd.DataFrame(sku_diagnostics)
    logger.info("Demand Pattern Breakdown:\n" + df_sku_diag['behavior_class'].value_counts().to_string())

    # 3. Update sku_master in MySQL
    logger.info("Updating behavioral metrics in sku_master table...")
    with db.engine.connect() as conn:
        for _, s in df_sku_diag.iterrows():
            conn.execute(text("""
                UPDATE sku_master
                SET adi = :adi,
                    cv2 = :cv2,
                    zero_demand_ratio = :zdr,
                    behavior_class = :bclass
                WHERE canonical_sku = :sku
            """), {
                "adi": float(s['adi']),
                "cv2": float(s['cv2']),
                "zdr": float(s['zero_demand_ratio']),
                "bclass": s['behavior_class'],
                "sku": s['canonical_sku']
            })
        conn.commit()

    logger.info("sku_master table successfully updated with behavioral metrics.")
    return df_brand_summary, df_sku_diag

if __name__ == "__main__":
    analyze_behavior()
