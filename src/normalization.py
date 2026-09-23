"""
DATA NORMALIZATION ENGINE (PHASE 1.1)
=====================================
Applies all cleaning, hierarchy, platform mapping, shared warehouse inventory,
and observation state logic to produce:
1. Transaction-level normalized dataset (with resolved identifiers and non-Amazon child_asin=NULL)
2. True Daily SKU x Platform Observation Layer (Grain: DATE x PLATFORM x CANONICAL_SKU)
"""
import pandas as pd
import numpy as np
from src.platform_mapping import apply_platform_mapping
from src.sku_mapping import apply_sku_hierarchy

def normalize_transactions(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw sales transactions into clean normalized transaction records.
    Preserves all 101,085 source rows with deterministic lineage.
    """
    df = df_raw.copy()
    
    # 1. Platform Mapping
    df = apply_platform_mapping(df)
    
    # 2. SKU & Hierarchy Mapping (with explicit resolution methods and child_asin fix)
    df = apply_sku_hierarchy(df)
    
    # 3. Format Date
    df['date'] = pd.to_datetime(df['date'])
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
    if 'launch_date' in df.columns:
        df['launch_date'] = pd.to_datetime(df['launch_date'])
        df['launch_date_str'] = df['launch_date'].dt.strftime('%Y-%m-%d')
    else:
        df['launch_date_str'] = 'N/A'
        
    # 4. Shared Warehouse Inventory Logic (Vectorized)
    df['has_inventory_signal'] = (
        (df['date'] >= pd.to_datetime('2025-08-01')) & 
        (df['current_stock'].notnull())
    ).astype(int)
    
    df['stockout_flag'] = (
        (df['has_inventory_signal'] == 1) & 
        (df['current_stock'] == 0)
    ).astype(int)
    
    conds_inv = [
        df['date'] < pd.to_datetime('2025-08-01'),
        df['current_stock'].isnull(),
        df['current_stock'] == 0,
        df['current_stock'] < 20
    ]
    choices_inv = [
        'UNMONITORED_HISTORICAL',
        'UNMONITORED',
        'OUT_OF_STOCK',
        'LOW_STOCK'
    ]
    df['inventory_status'] = np.select(conds_inv, choices_inv, default='IN_STOCK')
    
    # 5. Observation State Tagging
    conds_obs = [
        (df['units_sold'] > 0) & (df['stockout_flag'] == 1),
        (df['units_sold'] > 0) & (df['date'] < pd.to_datetime('2025-08-01')),
        (df['units_sold'] > 0),
        (df['units_sold'] == 0) & (df['stockout_flag'] == 1),
        (df['units_sold'] == 0) & (df['has_inventory_signal'] == 1) & (df['current_stock'] > 0),
        (df['units_sold'] == 0)
    ]
    choices_obs = [
        'STOCKOUT_DEMAND_CENSORED',
        'INVENTORY_UNKNOWN',
        'OBSERVED_SALE',
        'STOCKOUT_DEMAND_CENSORED',
        'OBSERVED_ZERO',
        'ACTIVE_NO_TRANSACTION'
    ]
    df['observation_state'] = np.select(conds_obs, choices_obs, default='INSUFFICIENT_EVIDENCE')
    
    df['model_units_sold'] = df['units_sold'].astype(float)
    df['treatment_method'] = 'AS_OBSERVED'
    df['treatment_confidence'] = 1.0
    
    # 6. Quality & Anomaly Flags
    df['is_zero_price_order'] = (df['selling_price'] <= 0).astype(int)
    df['is_pack_multiplier_discrepancy'] = (df['units_sold'] != df['orders_count'] * df['pack_multiplier']).astype(int)
    
    # 7. Final Clean Column Arrangement
    final_cols = [
        'source_row_id',
        'date_str',
        'platform_group',
        'channel',
        'channel_type',
        'raw_sku',
        'canonical_sku',
        'canonical_sku_source',
        'raw_parent_id',
        'resolved_parent_id',
        'parent_resolution_method',
        'raw_listing_id',
        'resolved_listing_id',
        'listing_resolution_method',
        'child_asin',
        'category',
        'pack_multiplier',
        'units_sold',
        'orders_count',
        'selling_price',
        'launch_date_str',
        'current_stock',
        'has_inventory_signal',
        'stockout_flag',
        'inventory_status',
        'observation_state',
        'model_units_sold',
        'treatment_method',
        'treatment_confidence',
        'is_zero_price_order',
        'is_pack_multiplier_discrepancy',
        'buy_box_percentage',
        'amazon_sessions',
        'ebay_promoted_flag',
        'fulfillment_type'
    ]
    
    df_normalized = df[final_cols].rename(columns={
        'date_str': 'date',
        'launch_date_str': 'launch_date',
        'units_sold': 'observed_units_sold'
    })
    
    return df_normalized

def build_daily_sku_platform_layer(df_tx_ml: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs the true Daily SKU x Platform Observation Layer for the ML-eligible period:
    Date Range: 2025-08-01 to 2026-09-10 (406 calendar days)
    Grain: DATE x PLATFORM x CANONICAL_SKU (573,678 rows)
    """
    df = df_tx_ml.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # 1. Aggregate positive transactions to DATE x PLATFORM x CANONICAL_SKU
    daily_sales = df.groupby(['date', 'platform_group', 'canonical_sku']).agg(
        observed_units_sold=('observed_units_sold', 'sum'),
        orders_count=('orders_count', 'sum'),
        selling_price=('selling_price', 'mean'),
        current_stock=('current_stock', 'first'),
        has_inventory_signal=('has_inventory_signal', 'first'),
        stockout_flag=('stockout_flag', 'first'),
        raw_sku=('raw_sku', 'first'),
        canonical_sku_source=('canonical_sku_source', 'first'),
        resolved_parent_id=('resolved_parent_id', 'first'),
        category=('category', 'first'),
        pack_multiplier=('pack_multiplier', 'first'),
        launch_date=('launch_date', 'first')
    ).reset_index()
    
    # 2. Extract SKU metadata
    sku_meta = df.groupby('canonical_sku').agg(
        category=('category', 'first'),
        resolved_parent_id=('resolved_parent_id', 'first'),
        canonical_sku_source=('canonical_sku_source', 'first'),
        launch_date=('launch_date', 'first'),
        pack_multiplier=('pack_multiplier', 'first'),
        total_lifetime_units=('observed_units_sold', 'sum'),
        lifetime_active_days=('date', 'nunique')
    ).reset_index()
    
    # Check TTS only on Other
    other_channels = df[df['platform_group'] == 'Other'].groupby('canonical_sku')['channel'].unique().to_dict()
    def is_tts_only(c_sku):
        chans = other_channels.get(c_sku, [])
        return 1 if any('TTS' in c for c in chans) and not any('UFK' in c for c in chans) else 0
    sku_meta['is_other_tts_only'] = sku_meta['canonical_sku'].map(is_tts_only)
    
    # 3. Extract platform active windows per canonical SKU
    platform_sku_meta = df.groupby(['platform_group', 'canonical_sku']).agg(
        first_platform_date=('date', 'min'),
        last_platform_date=('date', 'max'),
        platform_units=('observed_units_sold', 'sum'),
        platform_active_days=('date', 'nunique')
    ).reset_index()
    
    # 4. Extract shared warehouse daily stock per SKU from any channel that reported stock
    daily_stock = df.dropna(subset=['current_stock']).groupby(['date', 'canonical_sku'])['current_stock'].first().reset_index()
    daily_stock = daily_stock.rename(columns={'current_stock': 'warehouse_stock_on_date'})
    
    # 5. Build full calendar grid (406 days) for all active (platform, canonical_sku) pairs
    all_dates = pd.date_range('2025-08-01', '2026-09-10', freq='D')
    pairs = platform_sku_meta[['platform_group', 'canonical_sku']].drop_duplicates()
    
    grid_rows = []
    for _, p_row in pairs.iterrows():
        p_grp = p_row['platform_group']
        c_sku = p_row['canonical_sku']
        for d in all_dates:
            grid_rows.append((d, p_grp, c_sku))
            
    grid_df = pd.DataFrame(grid_rows, columns=['date', 'platform_group', 'canonical_sku'])
    grid_df = grid_df.sort_values(by=['canonical_sku', 'platform_group', 'date']).reset_index(drop=True)
    
    # Merge daily sales
    grid_df = grid_df.merge(
        daily_sales[['date', 'platform_group', 'canonical_sku', 'observed_units_sold', 'orders_count', 'selling_price', 'current_stock', 'has_inventory_signal', 'stockout_flag']],
        on=['date', 'platform_group', 'canonical_sku'],
        how='left'
    )
    
    # Merge metadata
    grid_df = grid_df.merge(sku_meta, on='canonical_sku', how='left')
    grid_df = grid_df.merge(platform_sku_meta, on=['platform_group', 'canonical_sku'], how='left')
    grid_df = grid_df.merge(daily_stock, on=['date', 'canonical_sku'], how='left')
    
    # Consolidate inventory
    grid_df['current_stock'] = grid_df['current_stock'].fillna(grid_df['warehouse_stock_on_date'])
    grid_df['current_stock'] = grid_df.groupby('canonical_sku')['current_stock'].ffill().bfill()
    grid_df['has_inventory_signal'] = grid_df['current_stock'].notnull().astype(int)
    grid_df['stockout_flag'] = ((grid_df['has_inventory_signal'] == 1) & (grid_df['current_stock'] == 0)).astype(int)
    
    # 6. Classify Observation States (Sales are NEVER labeled inactive/discontinued)
    def classify_state(row):
        d = row['date']
        l_date = pd.to_datetime(row['launch_date']) if pd.notnull(row['launch_date']) and row['launch_date'] != 'N/A' else None
        first_p = pd.to_datetime(row['first_platform_date'])
        units = row['observed_units_sold']
        stock = row['current_stock']
        platform = row['platform_group']
        
        # 1. If sales actually occurred:
        if pd.notnull(units) and units > 0:
            if row['stockout_flag'] == 1:
                return 'STOCKOUT_DEMAND_CENSORED'
            return 'OBSERVED_SALE'
            
        # 2. Pre-launch
        if l_date and d < l_date:
            return 'PRE_LAUNCH'
            
        # 3. Post-discontinuation for TikTok
        if platform == 'Other' and d > pd.to_datetime('2025-08-08') and row['is_other_tts_only'] == 1:
            return 'POST_DISCONTINUATION'
            
        # 4. Platform inactive prior to first listing on this platform
        if d < first_p:
            return 'PLATFORM_INACTIVE'
            
        # 5. Stockout / Demand censored (no sales occurred and stock == 0)
        if row['stockout_flag'] == 1 or (pd.notnull(stock) and stock == 0):
            return 'STOCKOUT_DEMAND_CENSORED'
            
        # 6. Insufficient evidence for very sparse products
        if row['total_lifetime_units'] < 5 and row['platform_active_days'] <= 2 and pd.isnull(units):
            return 'INSUFFICIENT_EVIDENCE'
            
        # 7. Active on platform and stock > 0 -> true zero demand
        if pd.notnull(stock) and stock > 0:
            return 'OBSERVED_ZERO'
            
        return 'INSUFFICIENT_EVIDENCE'
        
    grid_df['observation_state'] = grid_df.apply(classify_state, axis=1)
    
    # 7. Model Units Sold and Treatment Policies
    def assign_treatment(row):
        st = row['observation_state']
        units = row['observed_units_sold']
        if st == 'OBSERVED_SALE':
            return units, 'AS_OBSERVED', 1.0
        elif st == 'OBSERVED_ZERO':
            return 0.0, 'ZERO', 1.0
        elif st == 'STOCKOUT_DEMAND_CENSORED':
            if pd.notnull(units) and units > 0:
                return units, 'CENSORED_DEMAND', 0.5
            else:
                return np.nan, 'LEAVE_UNAVAILABLE', 0.0
        elif st in ['PRE_LAUNCH', 'POST_DISCONTINUATION', 'PLATFORM_INACTIVE']:
            return np.nan, 'NOT_APPLICABLE', 0.0
        else: # INSUFFICIENT_EVIDENCE / DATA_CAPTURE_GAP
            return np.nan, 'PRESERVE_UNCERTAINTY', 0.0
            
    treatment_results = grid_df.apply(assign_treatment, axis=1)
    grid_df['model_units_sold'] = [t[0] for t in treatment_results]
    grid_df['treatment_method'] = [t[1] for t in treatment_results]
    grid_df['treatment_confidence'] = [t[2] for t in treatment_results]
    
    # Fill observed_units_sold = 0 for OBSERVED_ZERO
    grid_df['observed_units_sold'] = np.where(grid_df['observation_state'] == 'OBSERVED_ZERO', 0, grid_df['observed_units_sold'])
    grid_df['orders_count'] = np.where(grid_df['observation_state'] == 'OBSERVED_ZERO', 0, grid_df['orders_count'])
    
    # Final date formatting
    grid_df['date_str'] = grid_df['date'].dt.strftime('%Y-%m-%d')
    grid_df['launch_date_str'] = pd.to_datetime(grid_df['launch_date']).dt.strftime('%Y-%m-%d')
    
    final_cols = [
        'date_str',
        'platform_group',
        'canonical_sku',
        'canonical_sku_source',
        'resolved_parent_id',
        'category',
        'pack_multiplier',
        'observed_units_sold',
        'orders_count',
        'selling_price',
        'current_stock',
        'has_inventory_signal',
        'stockout_flag',
        'observation_state',
        'model_units_sold',
        'treatment_method',
        'treatment_confidence',
        'launch_date_str'
    ]
    
    df_daily_obs = grid_df[final_cols].rename(columns={
        'date_str': 'date',
        'launch_date_str': 'launch_date'
    })
    
    return df_daily_obs
