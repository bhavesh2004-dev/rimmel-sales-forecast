"""
SKU & PRODUCT HIERARCHY MODULE (PHASE 1.1)
==========================================
Resolves the multi-level retail product hierarchy:
Parent Product Family -> Canonical SKU -> Sellable Variant / Raw SKU -> Platform Listing

Preserves original fields:
- raw_parent_id
- raw_sku
- raw_listing_id

Creates resolved fields:
- resolved_parent_id
- canonical_sku
- resolved_listing_id

Creates resolution tracking fields:
- parent_resolution_method ('FROM_RAW_PARENT_ID' or 'RAW_SKU_FALLBACK')
- canonical_sku_source ('ACTUAL_SKU' or 'RAW_SKU_FALLBACK')
- listing_resolution_method ('FROM_RAW_LISTING_ID' or 'RAW_SKU_FALLBACK')
"""
import pandas as pd
import numpy as np

def apply_sku_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies deterministic product hierarchy mappings while preserving original identifiers
    and creating explicit resolution method lineage fields.
    """
    df_out = df.copy()
    
    # 1. Preserve original raw identifiers cleanly trimmed
    df_out['raw_sku'] = df_out['sku'].astype(str).str.strip()
    
    # raw_parent_id
    if 'parent_id' in df_out.columns:
        df_out['raw_parent_id'] = df_out['parent_id'].astype(str).str.strip().replace({'nan': None, 'None': None, '': None})
    else:
        df_out['raw_parent_id'] = None
        
    # raw_listing_id
    if 'listing_id' in df_out.columns:
        df_out['raw_listing_id'] = df_out['listing_id'].astype(str).str.strip().replace({'nan': None, 'None': None, '': None})
    else:
        df_out['raw_listing_id'] = None
        
    # actual_sku
    if 'actual_sku' in df_out.columns:
        clean_actual = df_out['actual_sku'].astype(str).str.strip().replace({'nan': None, 'None': None, '': None})
    else:
        clean_actual = None
        
    # 2. Canonical SKU Resolution
    has_actual = clean_actual.notnull() if clean_actual is not None else pd.Series(False, index=df_out.index)
    df_out['canonical_sku'] = np.where(has_actual, clean_actual, df_out['raw_sku'])
    df_out['canonical_sku_source'] = np.where(has_actual, 'ACTUAL_SKU', 'RAW_SKU_FALLBACK')
    
    # 3. Parent ID Resolution
    has_parent = df_out['raw_parent_id'].notnull()
    df_out['resolved_parent_id'] = np.where(has_parent, df_out['raw_parent_id'], df_out['raw_sku'])
    df_out['parent_resolution_method'] = np.where(has_parent, 'FROM_RAW_PARENT_ID', 'RAW_SKU_FALLBACK')
    
    # 4. Listing ID Resolution
    has_listing = df_out['raw_listing_id'].notnull()
    df_out['resolved_listing_id'] = np.where(has_listing, df_out['raw_listing_id'], df_out['raw_sku'])
    df_out['listing_resolution_method'] = np.where(has_listing, 'FROM_RAW_LISTING_ID', 'RAW_SKU_FALLBACK')
    
    # 5. Fix child_asin: ONLY for Amazon; strictly None for non-Amazon
    if 'child_asin' in df_out.columns:
        clean_child = df_out['child_asin'].astype(str).str.strip().replace({'nan': None, 'None': None, '': None})
        is_amazon = df_out['channel'].str.contains('Amazon', na=False)
        df_out['child_asin'] = np.where(is_amazon, clean_child, None)
    else:
        df_out['child_asin'] = None
        
    return df_out

def build_sku_master(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs catalog master for all unique SKUs with resolution methods and cross-platform presence.
    """
    df_mapped = apply_sku_hierarchy(df)
    
    sku_records = []
    for raw_sku, group in df_mapped.groupby('raw_sku'):
        canonical_sku = group['canonical_sku'].iloc[0]
        canonical_source = group['canonical_sku_source'].iloc[0]
        resolved_parent = group['resolved_parent_id'].iloc[0]
        parent_method = group['parent_resolution_method'].iloc[0]
        raw_parent = group['raw_parent_id'].iloc[0]
        
        category = group['category'].dropna().iloc[0] if group['category'].notnull().any() else 'Uncategorized'
        launch_date = group['launch_date'].dropna().iloc[0] if group['launch_date'].notnull().any() else None
        pack_mult = int(group['pack_multiplier'].iloc[0]) if 'pack_multiplier' in group else 1
        
        channels = set(group['channel'].unique())
        is_amazon = 1 if any('Amazon' in c for c in channels) else 0
        is_ebay = 1 if any('Ebay' in c for c in channels) else 0
        is_website = 1 if any('Website' in c for c in channels) else 0
        is_other = 1 if any(c in ['Glam TTS - MFN', 'Glam TTS New - MFN', 'UFK Trading Manual - MFN'] for c in channels) else 0
        
        tot_units = int(group['units_sold'].sum()) if 'units_sold' in group else int(group['observed_units_sold'].sum())
        tot_orders = int(group['orders_count'].sum())
        avg_price = float(round(group['selling_price'].mean(), 2))
        
        first_sale = group['date'].min()
        last_sale = group['date'].max()
        first_sale_str = first_sale.strftime('%Y-%m-%d') if hasattr(first_sale, 'strftime') else str(first_sale)[:10]
        last_sale_str = last_sale.strftime('%Y-%m-%d') if hasattr(last_sale, 'strftime') else str(last_sale)[:10]
        active_days = group['date'].nunique()
        
        sku_records.append({
            'raw_sku': raw_sku,
            'canonical_sku': canonical_sku,
            'canonical_sku_source': canonical_source,
            'raw_parent_id': raw_parent,
            'resolved_parent_id': resolved_parent,
            'parent_resolution_method': parent_method,
            'category': category,
            'launch_date': launch_date.strftime('%Y-%m-%d') if pd.notnull(launch_date) and hasattr(launch_date, 'strftime') else str(launch_date)[:10],
            'pack_multiplier': pack_mult,
            'total_units_sold': tot_units,
            'total_orders': tot_orders,
            'avg_selling_price': avg_price,
            'first_sale_date': first_sale_str,
            'last_sale_date': last_sale_str,
            'active_days_count': active_days,
            'is_on_amazon': is_amazon,
            'is_on_ebay': is_ebay,
            'is_on_website': is_website,
            'is_on_other': is_other,
            'platforms_count': is_amazon + is_ebay + is_website + is_other,
            'channels_count': len(channels)
        })
        
    return pd.DataFrame(sku_records).sort_values(by='total_units_sold', ascending=False)
