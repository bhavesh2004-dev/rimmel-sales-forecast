"""
Multi-Brand Normalization Module
Standardizes brands, platform channels, canonical SKUs, quantities, and pricing.
Principle: Brand as Data.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def normalize_brand_name(raw_brand: str) -> Tuple[str, str]:
    """
    Returns (brand_id, display_name).
    Example: 'Max Factor' -> ('MAX_FACTOR', 'Max Factor')
             'Rimmel Brand Sales Data...' -> ('RIMMEL', 'Rimmel')
    """
    clean = str(raw_brand).strip()
    if not clean or clean.lower() == 'nan':
        return 'UNKNOWN_BRAND', 'Unknown Brand'
        
    # Check common known tokens or slugify
    upper = clean.upper()
    if 'RIMMEL' in upper:
        return 'RIMMEL', 'Rimmel'
    elif 'MAX' in upper and 'FACTOR' in upper:
        return 'MAX_FACTOR', 'Max Factor'
    elif 'WELEDA' in upper:
        return 'WELEDA', 'Weleda'
    elif 'DELILAH' in upper:
        return 'DELILAH', 'delilah'
    elif 'GEEK' in upper:
        return 'GEEK_GORGEOUS', 'Geek & Gorgeous'
    else:
        # Generic slugify for future brands
        brand_id = re.sub(r'[^A-Z0-9]+', '_', upper).strip('_')
        return brand_id, clean

def map_platform(channel: str, platform_mapping: Dict[str, List[str]]) -> str:
    """
    Maps a raw channel string to one of: Amazon, eBay, Website, Other.
    Prioritizes longer, more specific aliases.
    """
    if not channel or pd.isna(channel):
        return 'Other'
        
    norm_chan = str(channel).lower().strip()
    
    # Sort aliases by length descending to match most specific tokens first
    all_pairs = []
    for plat, aliases in platform_mapping.items():
        for alias in aliases:
            all_pairs.append((len(alias), alias.lower(), plat))
    all_pairs.sort(key=lambda x: x[0], reverse=True)
    
    for _, alias, plat in all_pairs:
        if alias in norm_chan:
            return plat
                
    return 'Other'

def resolve_canonical_sku(raw_sku: str, actual_sku: Optional[str] = None) -> str:
    """
    Resolves physical sellable canonical SKU from raw marketplace listing SKU.
    """
    if actual_sku and not pd.isna(actual_sku) and str(actual_sku).strip() != '' and str(actual_sku).lower() != 'none':
        return str(actual_sku).strip()
        
    s = str(raw_sku).strip()
    # Strip common channel listing suffixes if present
    s_clean = re.sub(r'-(FBA|MFN|AMZ|EBAY)$', '', s, flags=re.IGNORECASE)
    return s_clean

def clean_monetary_value(val: Any) -> float:
    """Strips currency symbols and converts to float."""
    if pd.isna(val) or val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return max(0.0, float(val))
    s = str(val).strip()
    s = re.sub(r'[£$€,\s]', '', s)
    try:
        return max(0.0, float(s))
    except ValueError:
        return 0.0

def normalize_transactions(
    df: pd.DataFrame,
    platform_mapping: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Normalizes ingested transaction DataFrame.
    """
    logger.info(f"Normalizing {len(df):,} raw transaction records...")
    df_out = df.copy()
    
    # 1. Brand Normalization
    brand_tuples = [normalize_brand_name(b) for b in df_out['raw_brand']]
    df_out['brand_id'] = [t[0] for t in brand_tuples]
    df_out['display_brand_name'] = [t[1] for t in brand_tuples]
    
    # 2. Date Normalization
    df_out['date'] = pd.to_datetime(df_out['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    invalid_dates = df_out['date'].isna().sum()
    if invalid_dates > 0:
        logger.warning(f"Dropping {invalid_dates} rows with unparseable dates.")
        df_out = df_out[df_out['date'].notna()].copy()
        
    # 3. Platform Mapping
    df_out['raw_channel'] = df_out['channel'].astype(str)
    df_out['platform_group'] = df_out['channel'].apply(lambda c: map_platform(c, platform_mapping))
    
    # 4. Canonical SKU Resolution
    has_actual = 'actual_sku' in df_out.columns
    df_out['raw_sku'] = df_out['sku'].astype(str)
    df_out['canonical_sku'] = [
        resolve_canonical_sku(r['raw_sku'], r.get('actual_sku', None) if has_actual else None)
        for _, r in df_out.iterrows()
    ]
    
    # 5. Pack Multiplier
    if 'pack_multiplier' in df_out.columns:
        df_out['pack_multiplier'] = pd.to_numeric(df_out['pack_multiplier'], errors='coerce').fillna(1).astype(int)
        df_out['pack_multiplier'] = df_out['pack_multiplier'].apply(lambda m: max(1, m))
    else:
        df_out['pack_multiplier'] = 1
        
    # 6. Physical Units Sold
    df_out['quantity'] = pd.to_numeric(df_out['quantity'], errors='coerce').fillna(0)
    df_out['units_sold'] = df_out['quantity'].apply(lambda q: max(0, float(q)))
    
    # 7. Orders Count
    if 'orders_count' in df_out.columns:
        df_out['orders_count'] = pd.to_numeric(df_out['orders_count'], errors='coerce').fillna(1).apply(lambda o: max(0, int(o)))
    else:
        df_out['orders_count'] = df_out['units_sold'].apply(lambda u: 1 if u > 0 else 0)
        
    # 8. Selling Price
    if 'price' in df_out.columns:
        df_out['selling_price'] = df_out['price'].apply(clean_monetary_value)
    else:
        df_out['selling_price'] = 0.0
        
    # 9. Stock (Keep NaN as unobserved; do NOT replace with zero)
    if 'stock' in df_out.columns:
        df_out['current_stock'] = pd.to_numeric(df_out['stock'], errors='coerce')
    else:
        df_out['current_stock'] = np.nan
        
    # Product title / category
    if 'title' in df_out.columns:
        df_out['product_title'] = df_out['title'].fillna('').astype(str)
    else:
        df_out['product_title'] = df_out['canonical_sku']
        
    if 'category' in df_out.columns:
        df_out['category'] = df_out['category'].fillna('Cosmetics General').astype(str)
    else:
        df_out['category'] = 'Cosmetics General'
        
    logger.info(f"Normalization complete. Retained {len(df_out):,} rows across {df_out['brand_id'].nunique()} brand(s).")
    return df_out
