"""
PLATFORM MAPPING MODULE (PHASE 1)
=================================
Normalizes raw multi-platform e-commerce channels into four high-level platform groups:
1. Amazon
2. eBay
3. Website
4. Other (TikTok Shop and Retail Store wholesale dispatches)

Preserves original channel names and assigns granular channel_type without losing identity.
"""
import pandas as pd
from typing import Dict, Tuple

# Exact business mapping dictionary
CHANNEL_TO_PLATFORM_MAP = {
    'Bellas Beauty Amazon - MFN': {
        'platform_group': 'Amazon',
        'channel_type': 'Amazon',
        'description': 'Amazon UK Merchant-Fulfilled storefront (Bellas Beauty)'
    },
    'Mayah Beauty Amazon - MFN': {
        'platform_group': 'Amazon',
        'channel_type': 'Amazon',
        'description': 'Amazon UK Merchant-Fulfilled storefront (Mayah Beauty)'
    },
    'Bellas Beauty Ebay - MFN': {
        'platform_group': 'eBay',
        'channel_type': 'eBay',
        'description': 'eBay UK Merchant-Fulfilled storefront (Bellas Beauty)'
    },
    'Mayah Beauty Ebay - MFN': {
        'platform_group': 'eBay',
        'channel_type': 'eBay',
        'description': 'eBay UK Merchant-Fulfilled storefront (Mayah Beauty)'
    },
    'GLAMBEAUTY Website - MFN': {
        'platform_group': 'Website',
        'channel_type': 'Website',
        'description': 'Direct-to-Consumer e-commerce storefront (GLAMBEAUTY)'
    },
    'Glam TTS - MFN': {
        'platform_group': 'Other',
        'channel_type': 'TikTok',
        'description': 'TikTok Shop storefront (Legacy, active Jan-Jun 2025)'
    },
    'Glam TTS New - MFN': {
        'platform_group': 'Other',
        'channel_type': 'TikTok',
        'description': 'TikTok Shop new account trial (Jul-Aug 2025)'
    },
    'UFK Trading Manual - MFN': {
        'platform_group': 'Other',
        'channel_type': 'Retail Store',
        'description': 'Physical retail / wholesale sample dispatches (Logged manually at £0.00 price)'
    }
}

def apply_platform_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends platform_group and channel_type while preserving original channel column.
    Raises ValueError if any unmapped channel is encountered.
    """
    df_out = df.copy()
    unmapped = set(df_out['channel'].dropna().unique()) - set(CHANNEL_TO_PLATFORM_MAP.keys())
    if unmapped:
        raise ValueError(f"Unmapped channels found in dataset: {unmapped}")
        
    df_out['platform_group'] = df_out['channel'].map(lambda c: CHANNEL_TO_PLATFORM_MAP[c]['platform_group'])
    df_out['channel_type']   = df_out['channel'].map(lambda c: CHANNEL_TO_PLATFORM_MAP[c]['channel_type'])
    return df_out

def build_platform_mapping_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs a comprehensive platform_mapping reference summary table.
    """
    rows = []
    total_units_all = df['units_sold'].sum()
    total_rows_all = len(df)
    
    for ch, meta in CHANNEL_TO_PLATFORM_MAP.items():
        ch_df = df[df['channel'] == ch]
        row_cnt = len(ch_df)
        units = ch_df['units_sold'].sum() if row_cnt > 0 else 0
        orders = ch_df['orders_count'].sum() if row_cnt > 0 else 0
        skus = ch_df['sku'].nunique() if row_cnt > 0 else 0
        min_d = ch_df['date'].min().strftime('%Y-%m-%d') if row_cnt > 0 else 'N/A'
        max_d = ch_df['date'].max().strftime('%Y-%m-%d') if row_cnt > 0 else 'N/A'
        
        rows.append({
            'channel': ch,
            'platform_group': meta['platform_group'],
            'channel_type': meta['channel_type'],
            'row_count': row_cnt,
            'row_share_pct': round((row_cnt / total_rows_all) * 100.0, 2) if total_rows_all > 0 else 0.0,
            'total_units_sold': units,
            'unit_share_pct': round((units / total_units_all) * 100.0, 2) if total_units_all > 0 else 0.0,
            'total_orders': orders,
            'unique_skus': skus,
            'min_date': min_d,
            'max_date': max_d,
            'description': meta['description']
        })
        
    return pd.DataFrame(rows)
