"""
OBSERVATION STATE MODULE (PHASE 1)
==================================
Establishes the 10-state observation framework for retail demand forecasting:
1. OBSERVED_SALE
2. OBSERVED_ZERO
3. ACTIVE_NO_TRANSACTION
4. PRE_LAUNCH
5. POST_DISCONTINUATION
6. PLATFORM_INACTIVE
7. STOCKOUT / DEMAND_CENSORED
8. INVENTORY_UNKNOWN
9. DATA_CAPTURE_GAP
10. INSUFFICIENT_EVIDENCE
"""
import pandas as pd
import numpy as np

OBSERVATION_STATE_DEFINITIONS = [
    {
        'state_name': 'OBSERVED_SALE',
        'definition': 'An explicit commercial transaction was recorded with units_sold > 0.',
        'evidence_criteria': 'units_sold > 0 in raw transaction log.',
        'modeling_implication': 'Ground-truth sales demand. Included in historical velocity features and loss functions.',
        'imputation_recommendation': 'AS_OBSERVED (No imputation required).'
    },
    {
        'state_name': 'OBSERVED_ZERO',
        'definition': 'Product was explicitly logged with zero sales on an active channel while stock was available.',
        'evidence_criteria': 'Explicit daily feed row with units_sold == 0 and current_stock > 0.',
        'modeling_implication': 'True zero demand signal. Informs intermittency and velocity deceleration.',
        'imputation_recommendation': 'ZERO (Preserve exact zero).'
    },
    {
        'state_name': 'ACTIVE_NO_TRANSACTION',
        'definition': 'Product listing is actively published and in stock, but zero transactions occurred on this date.',
        'evidence_criteria': 'Date falls between launch_date and active operational period; stock > 0; absent from raw daily log.',
        'modeling_implication': 'Represents true zero consumer demand for active SKU on this platform.',
        'imputation_recommendation': 'ZERO (Valid zero demand).'
    },
    {
        'state_name': 'PRE_LAUNCH',
        'definition': 'Date is prior to the product official launch_date or first operational listing.',
        'evidence_criteria': 'date < launch_date.',
        'modeling_implication': 'Product was not available in the market. Must NOT be treated as zero demand.',
        'imputation_recommendation': 'EXCLUDE_FROM_GRID (Do not impute).'
    },
    {
        'state_name': 'POST_DISCONTINUATION',
        'definition': 'Date is after the verified discontinuation of a channel (e.g. TikTok Shop after 2025-06-16).',
        'evidence_criteria': 'date > discontinuation_date on specific channel.',
        'modeling_implication': 'Channel is decommissioned. No inventory replenishment should be projected.',
        'imputation_recommendation': 'EXCLUDE_FROM_GRID (Do not impute).'
    },
    {
        'state_name': 'PLATFORM_INACTIVE',
        'definition': 'Product is not listed or commercially offered on this specific platform.',
        'evidence_criteria': 'SKU never listed or sold on this platform throughout its lifecycle.',
        'modeling_implication': 'Platform is not an operational sales vector for this SKU.',
        'imputation_recommendation': 'EXCLUDE_FROM_GRID (Do not impute).'
    },
    {
        'state_name': 'STOCKOUT / DEMAND_CENSORED',
        'definition': 'Central warehouse inventory was exhausted (current_stock == 0). Sales were constrained by supply.',
        'evidence_criteria': 'has_inventory_signal == 1 and current_stock == 0.',
        'modeling_implication': 'Observed sales under-represent true unconstrained consumer demand. Must be flagged in loss functions.',
        'imputation_recommendation': 'CENSORED_DEMAND_TREATMENT (Use latent demand estimation or exclude from velocity denominator).'
    },
    {
        'state_name': 'INVENTORY_UNKNOWN',
        'definition': 'Inventory tracking was not active (e.g. Jan 1 - Jul 31, 2025). Stockout status cannot be verified.',
        'evidence_criteria': 'date < 2025-08-01 (current_stock is null).',
        'modeling_implication': 'Sales occurred, but whether stockouts constrained demand is unobservable.',
        'imputation_recommendation': 'AS_OBSERVED_UNMONITORED (Retain sales; exclude from inventory-dependent models).'
    },
    {
        'state_name': 'DATA_CAPTURE_GAP',
        'definition': 'Channel data feed temporarily dropped or failed to record active transactions.',
        'evidence_criteria': 'Entire channel or catalog recorded zero transactions during a known operational day.',
        'modeling_implication': 'Data pipeline interruption, not market behavior.',
        'imputation_recommendation': 'INTERPOLATION / RECENT_LOCAL_MEAN (Where verified).'
    },
    {
        'state_name': 'INSUFFICIENT_EVIDENCE',
        'definition': 'SKU has fewer than 5 lifetime transactions or highly erratic sparse records.',
        'evidence_criteria': 'Lifetime active days < 5 or lifetime units sold < 10.',
        'modeling_implication': 'Insufficient statistical history for standalone ML time-series modeling.',
        'imputation_recommendation': 'CATEGORY_HIERARCHICAL_PRIOR (Fall back to category baseline).'
    }
]

def build_observation_states_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs the observation_states reference table and calculates empirical counts in the dataset.
    """
    df_states = pd.DataFrame(OBSERVATION_STATE_DEFINITIONS)
    
    # Calculate dataset counts
    # In transactional raw dataset, all rows have units_sold > 0
    # In-stock vs Stockout vs Inventory Unknown
    counts = {}
    for item in OBSERVATION_STATE_DEFINITIONS:
        s = item['state_name']
        if s == 'OBSERVED_SALE':
            # Rows with inventory known > 0 and units > 0
            cnt = len(df[(df['units_sold'] > 0) & (df['date'] >= '2025-08-01') & (df['current_stock'] > 0)])
        elif s == 'INVENTORY_UNKNOWN':
            # Rows in pre-Aug 2025 period
            cnt = len(df[df['date'] < '2025-08-01'])
        elif s == 'STOCKOUT / DEMAND_CENSORED':
            # Rows where stock was 0 but units were sold (or near stockout)
            cnt = len(df[(df['date'] >= '2025-08-01') & (df['current_stock'] == 0)])
        else:
            cnt = 0
            
        counts[s] = cnt
        
    df_states['count_in_dataset'] = df_states['state_name'].map(counts).fillna(0).astype(int)
    total_rows = len(df)
    df_states['percentage'] = df_states['count_in_dataset'].map(lambda c: f"{(c / total_rows) * 100:.2f}%" if c > 0 else "Grid State")
    
    return df_states
